use std::sync::Arc;

use ha_ndarray::{Axes, AxisRange, Range, Shape};

use crate::error::{Error, Result};
use crate::schema::{self, Layout, TensorViewShape};
use crate::tensor::{Tensor, TensorElement, TensorFileEntry};
use crate::traits::{
    BoxFuture, TensorArray, TensorGeometry, TensorRead, TensorTransform, TensorViewSemantics,
    TensorWrite,
};

use crate::{stream, validate};

#[derive(Clone)]
pub struct TensorView<'t, FE, T> {
    tensor: &'t Tensor<FE, T>,
    base_offset: i64,
    axes: Vec<AxisContrib>,
    shape: TensorViewShape,
}

#[derive(Clone)]
pub(crate) enum AxisContrib {
    Stride(i64),
    Gather(Arc<[i64]>),
}

impl<'t, FE, T> TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    pub(crate) fn new_identity(tensor: &'t Tensor<FE, T>) -> Self {
        let shape: ha_ndarray::Shape = tensor.schema().shape().to_vec().into();
        Self {
            tensor,
            base_offset: 0,
            axes: tensor
                .schema()
                .strides()
                .iter()
                .map(|&s| AxisContrib::Stride(s as i64))
                .collect(),
            shape: TensorViewShape::from(shape),
        }
    }

    pub fn flat_offset(&self, coord: &[u64]) -> Result<i64> {
        if coord.len() != self.axes.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        let mut k: i64 = self.base_offset;
        for (c, axis) in coord.iter().zip(self.axes.iter()) {
            k += match axis {
                AxisContrib::Stride(s) => (*c as i64) * s,
                AxisContrib::Gather(offsets) => {
                    let i = usize::try_from(*c)
                        .map_err(|_| Error::InvalidCoord("coord overflows usize".to_string()))?;
                    *offsets.get(i).ok_or_else(|| {
                        Error::InvalidCoord("coord out of bounds for gather".to_string())
                    })?
                }
            };
        }

        Ok(k)
    }

    fn is_c_contiguous(&self) -> bool {
        if self.axes.len() != self.shape.len() {
            return false;
        }
        let Ok(expected) = schema::contiguous_strides(&self.shape) else {
            return false;
        };
        self.axes
            .iter()
            .zip(expected.iter())
            .all(|pair| match pair {
                (AxisContrib::Stride(s), e) => *s >= 0 && *s as usize == *e,
                _ => false,
            })
    }

    pub fn view_encoder(&self) -> stream::TensorViewEncoder<'_, 't, FE, T> {
        stream::TensorViewEncoder::new(self)
    }

    fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        validate::validate_coord(&self.shape, coord)?;
        let k = self.flat_offset(coord)?;
        if k < 0 {
            return Err(Error::InvalidCoord("negative linear offset".to_string()));
        }
        let k = k as u64;
        let base_coord: Vec<u64> = self
            .tensor
            .strides()
            .iter()
            .zip(self.tensor.shape().iter())
            .map(|(stride, dim)| (k / *stride as u64) % *dim as u64)
            .collect();
        validate::validate_coord(self.tensor.shape(), &base_coord)?;
        Ok(base_coord)
    }

    pub(crate) fn tensor(&self) -> &'t Tensor<FE, T> {
        self.tensor
    }
}

pub fn default_permutation(ndim: usize, permutation: Option<Axes>) -> Result<Vec<usize>> {
    let axes: Vec<usize> = permutation
        .map(|axes| axes.into_iter().collect())
        .unwrap_or_else(|| (0..ndim).rev().collect());

    if axes.len() != ndim {
        return Err(Error::InvalidLayout(
            "transpose permutation rank must match tensor rank".to_string(),
        ));
    }

    Ok(axes)
}

impl<'t, FE, T> TensorGeometry for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type DType = T;

    fn dtype(&self) -> Self::DType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'t, FE, T> TensorViewSemantics for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn is_base_tensor(&self) -> bool {
        self.base_offset == 0
            && self.shape.as_slice() == self.tensor.shape()
            && self.is_c_contiguous()
    }

    fn supports_write_through(&self) -> bool {
        !self
            .axes
            .iter()
            .any(|a| matches!(a, AxisContrib::Gather(_)))
    }
}

impl<'t, FE, T> TensorTransform for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = shape.iter().product();
        if old_size != new_size {
            return Err(Error::InvalidLayout(
                "reshape requires an equal number of elements".to_string(),
            ));
        }
        if !self.is_c_contiguous() {
            return Err(Error::Unsupported(
                "reshape requires a C-contiguous view; copy the tensor before reshaping \
                a transposed, flip, step-strided, or gather-sliced view"
                    .to_string(),
            ));
        }
        let strides = schema::contiguous_strides(&shape)?;
        Ok(Self {
            tensor: self.tensor,
            base_offset: self.base_offset,
            axes: strides
                .iter()
                .map(|&s| AxisContrib::Stride(s as i64))
                .collect(),
            shape,
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        if range.len() != self.shape.len() {
            return Err(Error::InvalidLayout(
                "slice range rank must match tensor rank".to_string(),
            ));
        }

        let mut new_axes: Vec<AxisContrib> = Vec::with_capacity(self.axes.len());
        let mut new_base_offset = self.base_offset;
        let mut new_shape = Shape::with_capacity(self.axes.len());

        for (axis_index, (bound, dim)) in range.iter().zip(self.shape.iter()).enumerate() {
            let current = &self.axes[axis_index];

            match bound {
                AxisRange::At(i) => {
                    if *i >= *dim {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }
                    new_base_offset += match current {
                        AxisContrib::Stride(s) => (*i as i64) * s,
                        AxisContrib::Gather(g) => *g.get(*i).ok_or_else(|| {
                            Error::InvalidLayout(format!(
                                "slice bound at axis {axis_index} out of bounds for gather"
                            ))
                        })?,
                    };
                }
                AxisRange::In(start, stop, step) => {
                    if *step == 0 || *start > *stop || *stop > *dim {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }

                    let extent = if start == stop {
                        0
                    } else {
                        (stop - start).div_ceil(*step)
                    };

                    match current {
                        AxisContrib::Stride(s) => {
                            new_base_offset += (*start as i64) * s;
                            let new_s = s * (*step as i64);
                            new_axes.push(AxisContrib::Stride(new_s));
                        }
                        AxisContrib::Gather(g) => {
                            let offsets = (0..extent)
                                .map(|c| {
                                    g.get(start + c * step).copied().ok_or_else(|| {
                                        Error::InvalidLayout(format!(
                                            "slice bound at axis {axis_index} out of bounds for gather"
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<i64>>>()?;
                            new_axes.push(AxisContrib::Gather(offsets.into()));
                        }
                    }
                    new_shape.push(extent);
                }
                AxisRange::Of(indices) => {
                    if indices.iter().any(|index| *index >= *dim) {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }

                    let offsets = indices
                        .iter()
                        .map(|idx| {
                            Ok(match current {
                                AxisContrib::Stride(s) => (*idx as i64) * s,
                                AxisContrib::Gather(g) => *g.get(*idx).ok_or_else(|| {
                                    Error::InvalidLayout(format!(
                                        "slice bound at axis {axis_index} out of bounds for gather"
                                    ))
                                })?,
                            })
                        })
                        .collect::<Result<Vec<i64>>>()?;

                    new_axes.push(AxisContrib::Gather(offsets.into()));
                    new_shape.push(indices.len());
                }
            }
        }

        Ok(Self {
            tensor: self.tensor,
            base_offset: new_base_offset,
            axes: new_axes,
            shape: new_shape,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        let permutation = default_permutation(self.shape.len(), permutation)?;

        if permutation.len() != self.axes.len() {
            return Err(Error::InvalidLayout(
                "transpose permutation rank must match tensor rank".to_string(),
            ));
        }

        let mut seen = vec![false; self.axes.len()];
        let mut axes = Vec::with_capacity(self.axes.len());
        let mut shape = Shape::with_capacity(self.axes.len());

        for permuted_axis in permutation {
            if permuted_axis >= self.axes.len() || seen[permuted_axis] {
                return Err(Error::InvalidLayout(
                    "transpose permutation must be a valid axis permutation".to_string(),
                ));
            }

            seen[permuted_axis] = true;
            axes.push(self.axes[permuted_axis].clone());
            shape.push(self.shape[permuted_axis]);
        }

        Ok(Self {
            tensor: self.tensor,
            base_offset: self.base_offset,
            axes,
            shape,
        })
    }
}

impl<'t, FE, T> TensorRead for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let base_coord = self.resolve_base_coord(coord)?;
            self.tensor.read_value(&base_coord).await
        })
    }
}

impl<'t, FE, T> TensorWrite for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn write_value<'a>(
        &'a self,
        coord: &'a [u64],
        value: Self::DType,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let base_coord = self.resolve_base_coord(coord)?;
            self.tensor.write_value(&base_coord, value).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::{Path, PathBuf};

    use b_table::Node;
    use destream::{de, en};
    use freqfs::Cache;
    use ha_ndarray::{AxisRange, axes, range, shape};
    use safecast::as_type;

    use crate::{DType, Error, Layout, Tensor, TensorSchema};

    use super::*;

    #[derive(Clone, Debug)]
    enum TestFE {
        Node(Node<u64>),
        F32(Vec<f32>),
        Text(String),
    }

    impl<'en> en::ToStream<'en> for TestFE {
        fn to_stream<E: en::Encoder<'en>>(
            &'en self,
            encoder: E,
        ) -> std::result::Result<E::Ok, E::Error> {
            match self {
                Self::Node(node) => node.to_stream(encoder),
                Self::F32(values) => values.to_stream(encoder),
                Self::Text(text) => text.to_stream(encoder),
            }
        }
    }

    // Only ever read via a concrete `AsType` target (mirrors `src/lib.rs::sparse_lifecycle_tests`).
    impl de::FromStream for TestFE {
        type Context = ();

        async fn from_stream<D: de::Decoder>(
            _: (),
            _decoder: &mut D,
        ) -> std::result::Result<Self, D::Error> {
            Err(de::Error::custom(
                "TestFE does not support generic decoding; read via a concrete AsType target",
            ))
        }
    }

    as_type!(TestFE, Node, Node<u64>);
    as_type!(TestFE, F32, Vec<f32>);
    as_type!(TestFE, Text, String);

    fn unique_tmp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        path.push(format!("fensor_view_tests_{name}_{unique}"));
        path
    }

    fn open_dir(root: &Path) -> io::Result<freqfs::DirLock<TestFE>> {
        let cache = Cache::<TestFE>::new(1_000_000, None);
        cache.load(root.to_path_buf())
    }

    async fn new_dir(name: &str) -> (PathBuf, freqfs::DirLock<TestFE>) {
        let root = unique_tmp_dir(name);
        tokio::fs::create_dir(&root).await.expect("create tmp dir");
        let dir = open_dir(&root).expect("load tmp dir");
        (root, dir)
    }

    async fn cleanup(root: &Path) {
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    async fn create_dense(
        name: &str,
        shape: Shape,
        max_capacity: usize,
    ) -> (PathBuf, Tensor<TestFE, f32>) {
        let (root, dir) = new_dir(name).await;
        let schema = TensorSchema::new(DType::F32, shape).expect("schema");
        let tensor = Tensor::<TestFE, f32>::create(dir, schema, Layout::Dense, max_capacity)
            .await
            .expect("create dense");
        (root, tensor)
    }

    // -- flat_offset correctness ---------------------------------------------------

    #[tokio::test]
    async fn flat_offset_identity() {
        // [4,5,6], strides [30,6,1], coord [2,1,3] → k = 2*30 + 1*6 + 3*1 = 69
        let (root, tensor) = create_dense("flat_offset_identity", shape![4, 5, 6], 1000).await;
        let view = tensor.view();
        assert_eq!(view.flat_offset(&[2, 1, 3]).expect("offset"), 69);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn flat_offset_reshape_rank_reducing() {
        // [2,3,4]→[6,4]: new strides [4,1], coord [3,2] → k = 3*4 + 2*1 = 14
        let (root, tensor) =
            create_dense("flat_offset_reshape_rank_reducing", shape![2, 3, 4], 1000).await;
        let view = tensor.view();
        let reshaped = view.reshape(shape![6, 4]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[3, 2]).expect("offset"), 14);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn flat_offset_reshape_rank_increasing() {
        // [6,4]→[2,3,4]: new strides [12,4,1], coord [1,0,2] → k = 1*12 + 0*4 + 2*1 = 14
        let (root, tensor) =
            create_dense("flat_offset_reshape_rank_increasing", shape![6, 4], 1000).await;
        let view = tensor.view();
        let reshaped = view.reshape(shape![2, 3, 4]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[1, 0, 2]).expect("offset"), 14);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn flat_offset_reshape_same_rank() {
        // [6,4]→[4,6]: new strides [6,1], coord [2,3] → k = 2*6 + 3*1 = 15
        let (root, tensor) =
            create_dense("flat_offset_reshape_same_rank", shape![6, 4], 1000).await;
        let view = tensor.view();
        let reshaped = view.reshape(shape![4, 6]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[2, 3]).expect("offset"), 15);
        cleanup(&root).await;
    }

    // -- slice / transpose flat_offset correctness --------------------------------

    #[tokio::test]
    async fn slice_then_resolve() {
        // [4,5,6], strides [30,6,1]
        // At(2): base_offset += 2*30 = 60, axis dropped
        // In(1,5,2): base_offset += 1*6 = 6, axis → Stride(12), extent 2
        // Of([0,3,4]): Gather([0,3,4]), extent 3
        // flat_offset([1,2]) = 66 + 1*12 + Gather[2]=4 = 82
        let (root, tensor) = create_dense("slice_then_resolve", shape![4, 5, 6], 1000).await;
        let view = tensor.view();
        let r = range![
            AxisRange::At(2),
            AxisRange::In(1, 5, 2),
            AxisRange::Of(shape![0, 3, 4])
        ];
        let sliced = view.slice(r).expect("slice");
        assert_eq!(sliced.shape(), &[2, 3]);
        assert_eq!(sliced.flat_offset(&[1, 2]).expect("offset"), 82);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn transpose_arbitrary_permutation() {
        // [3,4,5], strides [20,5,1], perm [2,0,1] → axes [Stride(1),Stride(20),Stride(5)]
        // flat_offset([1,2,3]) = 1*1 + 2*20 + 3*5 = 56
        let (root, tensor) =
            create_dense("transpose_arbitrary_permutation", shape![3, 4, 5], 1000).await;
        let view = tensor.view();
        let transposed = view.transpose(Some(axes![2, 0, 1])).expect("transpose");
        assert_eq!(transposed.flat_offset(&[1, 2, 3]).expect("offset"), 56);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn compose_transpose_and_slice() {
        // [3,4,5], transpose [2,0,1] → axes [Stride(1),Stride(20),Stride(5)], shape [5,3,4]
        // In(1,5,2) on axis 0 (Stride(1)): base_offset+=1, Stride(2), extent 2
        // At(2) on axis 1 (Stride(20)): base_offset+=40, dropped
        // In(0,4,2) on axis 2 (Stride(5)): base_offset+=0, Stride(10), extent 2
        // base_offset=41, axes=[Stride(2),Stride(10)], shape=[2,2]
        // flat_offset([1,1]) = 41 + 2 + 10 = 53
        let (root, tensor) =
            create_dense("compose_transpose_and_slice", shape![3, 4, 5], 1000).await;
        let view = tensor.view();
        let transposed = view.transpose(Some(axes![2, 0, 1])).expect("transpose");
        let r = range![
            AxisRange::In(1, 5, 2),
            AxisRange::At(2),
            AxisRange::In(0, 4, 2)
        ];
        let sliced = transposed.slice(r).expect("slice");
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.flat_offset(&[1, 1]).expect("offset"), 53);
        cleanup(&root).await;
    }

    // -- invalid reshape chain guards ---------------------------------------------

    #[tokio::test]
    async fn transpose_then_reshape_rejected() {
        // transposed [3,4]: axes=[Stride(1),Stride(4)]; contiguous_strides([4,3])=[3,1]; 1≠3
        let (root, tensor) =
            create_dense("transpose_then_reshape_rejected", shape![3, 4], 1000).await;
        let view = tensor.view();
        let transposed = view.transpose(Some(axes![1, 0])).expect("transpose");
        assert!(matches!(
            transposed.reshape(shape![12]),
            Err(Error::Unsupported(_))
        ));
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn strided_slice_then_reshape_rejected() {
        // In(0,6,2) on axis 0 of [6,4]: axes=[Stride(8),Stride(1)]; contiguous_strides([3,4])=[4,1]; 8≠4
        let (root, tensor) =
            create_dense("strided_slice_then_reshape_rejected", shape![6, 4], 1000).await;
        let view = tensor.view();
        let r = range![AxisRange::In(0, 6, 2), AxisRange::In(0, 4, 1)];
        let sliced = view.slice(r).expect("slice");
        assert!(matches!(
            sliced.reshape(shape![12]),
            Err(Error::Unsupported(_))
        ));
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn gather_slice_then_reshape_rejected() {
        // Of([0,2,4]) on axis 0: Gather axis present → rejected
        let (root, tensor) =
            create_dense("gather_slice_then_reshape_rejected", shape![6, 4], 1000).await;
        let view = tensor.view();
        let r = range![AxisRange::Of(shape![0, 2, 4]), AxisRange::In(0, 4, 1)];
        let sliced = view.slice(r).expect("slice");
        assert!(matches!(
            sliced.reshape(shape![12]),
            Err(Error::Unsupported(_))
        ));
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn at_slice_non_first_axis_then_reshape_rejected() {
        // At(0) on axis 1 of [3,4]: axes=[Stride(4)]; contiguous_strides([3])=[1]; 4≠1
        let (root, tensor) = create_dense(
            "at_slice_non_first_axis_then_reshape_rejected",
            shape![3, 4],
            1000,
        )
        .await;
        let view = tensor.view();
        let r = range![AxisRange::In(0, 3, 1), AxisRange::At(0)];
        let sliced = view.slice(r).expect("slice");
        assert!(matches!(
            sliced.reshape(shape![3]),
            Err(Error::Unsupported(_))
        ));
        cleanup(&root).await;
    }

    // -- valid reshape chains -----------------------------------------------------

    #[tokio::test]
    async fn step1_slice_then_reshape_valid() {
        // In(1,4,1) on axis 0 of [6,4]: strides unchanged [4,1], contiguous for [3,4] ✓
        // base_offset=4; reshape [3,4]→[12]: axes=[Stride(1)]
        // flat_offset([5]) = 4 + 5*1 = 9
        let (root, tensor) =
            create_dense("step1_slice_then_reshape_valid", shape![6, 4], 1000).await;
        let view = tensor.view();
        let r = range![AxisRange::In(1, 4, 1), AxisRange::In(0, 4, 1)];
        let sliced = view.slice(r).expect("slice");
        let reshaped = sliced.reshape(shape![12]).expect("reshape must succeed");
        assert_eq!(reshaped.flat_offset(&[5]).expect("offset"), 9);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn at_first_axis_then_reshape_valid() {
        // At(1) on axis 0 of [3,4]: base_offset=4, axes=[Stride(1)], contiguous for [4] ✓
        // reshape [4]→[2,2]: axes=[Stride(2),Stride(1)]
        // flat_offset([1,1]) = 4 + 1*2 + 1*1 = 7
        let (root, tensor) =
            create_dense("at_first_axis_then_reshape_valid", shape![3, 4], 1000).await;
        let view = tensor.view();
        let r = range![AxisRange::At(1), AxisRange::In(0, 4, 1)];
        let sliced = view.slice(r).expect("slice");
        let reshaped = sliced.reshape(shape![2, 2]).expect("reshape must succeed");
        assert_eq!(reshaped.flat_offset(&[1, 1]).expect("offset"), 7);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn reshape_then_slice_flat_offset() {
        // reshape [6,4]→[2,3,4]: axes=[Stride(12),Stride(4),Stride(1)]
        // slice In(0,2,1) / In(0,2,1) / In(0,4,1); shape=[2,2,4]
        // flat_offset([1,1,2]) = 0 + 1*12 + 1*4 + 2*1 = 18
        let (root, tensor) =
            create_dense("reshape_then_slice_flat_offset", shape![6, 4], 1000).await;
        let view = tensor.view();
        let reshaped = view.reshape(shape![2, 3, 4]).expect("reshape");
        let r = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 4, 1)
        ];
        let sliced = reshaped.slice(r).expect("slice");
        assert_eq!(sliced.shape(), &[2, 2, 4]);
        assert_eq!(sliced.flat_offset(&[1, 1, 2]).expect("offset"), 18);
        cleanup(&root).await;
    }
}
