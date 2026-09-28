//! Bounded last-axis Fourier evaluation. Each pack contains complete axis groups;
//! backend workspace is bounded by the capped transform size. Nested transforms
//! may recompute groups; there is no result cache or implicit persistence.

use std::collections::BTreeMap;

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{ArrayAccess, NDArrayFourier, NDArrayRead, NDArrayTransform, Number};

use crate::expression::{self, Batch, Expression, MAX_BATCH_ELEMENTS};
use crate::mapping::CoordinateMap;
use crate::request::{self, Axis, BatchRequest};
use crate::schema::Coord;
use crate::slice::Slice;
use crate::{
    Axes, BoxFuture, Error, Layout, Range, Result, Shape, SparseElementStream, Strides,
    TensorElement, TensorFourier, TensorGeometry, TensorRead, TensorTransform, TensorViewSemantics,
    ValueBlockStream,
};

mod sealed {
    pub trait Sealed {}
}

/// Sealed backend Fourier operation; numerical work belongs to ha-ndarray.
pub trait FourierOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    const NAME: &'static str;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>>;
}

/// Unnormalized forward Fourier transform.
#[derive(Clone, Copy, Debug)]
pub struct Fft;

/// Unnormalized inverse Fourier transform.
#[derive(Clone, Copy, Debug)]
pub struct Ifft;

impl sealed::Sealed for Fft {}

impl sealed::Sealed for Ifft {}

impl<T: TensorElement> FourierOp<T> for Fft
where
    ArrayAccess<'static, T>: NDArrayFourier<DType = T, Platform = ha_ndarray::Platform>,
    <ArrayAccess<'static, T> as NDArrayFourier>::Output: Into<ha_ndarray::Accessor<'static, T>>,
    T: ha_ndarray::Complex,
{
    const NAME: &'static str = "fft";

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.fft()?))
    }
}

impl<T: TensorElement> FourierOp<T> for Ifft
where
    ArrayAccess<'static, T>: NDArrayFourier<DType = T, Platform = ha_ndarray::Platform>,
    <ArrayAccess<'static, T> as NDArrayFourier>::Output: Into<ha_ndarray::Accessor<'static, T>>,
    T: ha_ndarray::Complex,
{
    const NAME: &'static str = "ifft";

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.ifft()?))
    }
}

/// Lazy, read-only Fourier view with bounded complete-group evaluation.
///
/// Output transforms change coordinate mapping, never the source transform axis.
/// Even one requested frequency reads its complete input group. Sparse groups
/// retain union support, including supported numerical zeros.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorFourier};
/// async fn real<F: TensorFileEntry<f32>>(t: &Tensor<F, f32>) {
///     t.view().fft().await.unwrap();
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorFourier, TensorWrite};
/// fn writable<V: TensorWrite>(_: &V) {}
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     writable(&t.view().fft().await.unwrap());
/// }
/// ```
///
/// ```
/// use fensor::{complex::Complex32, Result, Tensor, TensorFileEntry, TensorFourier,
///     TensorTransform, TensorMatrixUnaryComplex};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) -> Result<()> {
///     let view = t.view().fft().await?;
///     let _ = view.clone().flip(0)?.mh().await?;
///     Ok(())
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex64, Tensor, TensorArray, TensorFileEntry, TensorFourier};
/// fn storage<V: TensorArray>(_: &V) {}
/// async fn example<F: TensorFileEntry<Complex64>>(t: &Tensor<F, Complex64>) {
///     storage(&t.view().ifft().await.unwrap());
/// }
/// ```
pub struct FourierView<Source, Op> {
    source: Source,
    op: Op,
    // Rank-sized original geometry and output mapping.
    shape: Shape,
    strides: Strides,
    mapping: CoordinateMap,
}

impl<S: Clone, O: Clone> Clone for FourierView<S, O> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            op: self.op.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            mapping: self.mapping.clone(),
        }
    }
}

fn validate_axes(shape: &[u64], count: usize, operation: &str) -> Result<()> {
    if shape.len() < count || shape.contains(&0) {
        return Err(Error::InvalidLayout(format!(
            "{operation} requires rank >= {count} and nonempty dimensions: {shape:?}"
        )));
    }
    crate::schema::contiguous_strides(shape)?;

    for &len in &shape[shape.len() - count..] {
        if len > MAX_BATCH_ELEMENTS as u64 {
            return Err(Error::Unsupported(format!(
                "{operation} axis length {len} exceeds limit {MAX_BATCH_ELEMENTS}"
            )));
        }
    }
    Ok(())
}

impl<S: Expression + Clone, O: FourierOp<S::DType>> FourierView<S, O>
where
    S::DType: TensorElement,
{
    fn new(source: &S, op: O) -> Result<Self> {
        validate_axes(source.shape(), 1, O::NAME)?;
        let shape = Shape::from_slice(source.shape());
        let strides = crate::schema::contiguous_strides(&shape)?;
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        Ok(Self {
            source: source.clone(),
            op,
            shape,
            strides,
            mapping,
        })
    }
}

impl<E: Expression + Clone> TensorFourier for E
where
    E::DType: TensorElement,
    Fft: FourierOp<E::DType>,
    Ifft: FourierOp<E::DType>,
{
    type FftOutput = FourierView<Self, Fft>;
    type IfftOutput = FourierView<Self, Ifft>;

    fn fft(&self) -> BoxFuture<'_, Result<Self::FftOutput>> {
        Box::pin(async move { FourierView::new(self, Fft) })
    }

    fn ifft(&self) -> BoxFuture<'_, Result<Self::IfftOutput>> {
        Box::pin(async move { FourierView::new(self, Ifft) })
    }
}

/// Unnormalized forward transform of the final two axes, preserving batch axes.
/// Both axis lengths must fit the execution limit; nested groups may be recomputed.
pub async fn fft2<E>(source: &E) -> Result<FourierView<FourierView<E, Fft>, Fft>>
where
    E: Expression + Clone,
    E::DType: TensorElement,
    Fft: FourierOp<E::DType>,
{
    validate_axes(source.shape(), 2, "fft2")?;
    let mut axes: Axes = (0..source.ndim()).collect();
    axes.swap(source.ndim() - 2, source.ndim() - 1);
    let first = FourierView::new(source, Fft)?.transpose(Some(axes.clone()))?;
    FourierView::new(&first, Fft)?.transpose(Some(axes))
}

/// Unnormalized inverse transform of the final two axes.
/// Inverse-after-forward scales by the product of the final two dimensions.
pub async fn ifft2<E>(source: &E) -> Result<FourierView<FourierView<E, Ifft>, Ifft>>
where
    E: Expression + TensorTransform + Clone,
    E::DType: TensorElement,
    Ifft: FourierOp<E::DType>,
{
    validate_axes(source.shape(), 2, "ifft2")?;
    let mut axes: Axes = (0..source.ndim()).collect();
    axes.swap(source.ndim() - 2, source.ndim() - 1);
    let transposed = source.clone().transpose(Some(axes.clone()))?;
    let first = FourierView::new(&transposed, Ifft)?.transpose(Some(axes))?;
    FourierView::new(&first, Ifft)
}

impl<S: TensorGeometry, O: Send + Sync> TensorGeometry for FourierView<S, O> {
    type DType = S::DType;

    fn dtype(&self) -> crate::NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.mapping.shape
    }

    fn layout(&self) -> Layout {
        match self.source.layout() {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: None },
        }
    }
}

impl<S: TensorGeometry, O: Send + Sync> TensorViewSemantics for FourierView<S, O> {
    fn is_base_tensor(&self) -> bool {
        false
    }

    fn supports_write_through(&self) -> bool {
        false
    }
}

// Group keys and scatter entries contain at most one entry per requested output.
type Groups = BTreeMap<Coord, Vec<(usize, usize)>>;

impl<S, O> FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
{
    fn groups(&self, request: &BatchRequest) -> Result<Groups> {
        let mut cursor = request.cursor(self.shape())?;
        let mut coord = Coord::new();
        let mut mapped = Coord::new();
        let mut groups = Groups::new();
        let mut destination = 0;
        while cursor.next_into(&mut coord) {
            self.mapping
                .resolve_into(&coord, &self.shape, &self.strides, &mut mapped)?;
            let frequency = mapped.pop().expect("validated Fourier rank") as usize;
            groups
                .entry(mapped.clone())
                .or_default()
                .push((destination, frequency));
            destination += 1;
        }
        Ok(groups)
    }
}

impl<S, O> Expression for FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
{
    fn build<'a>(&'a self, request: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let groups = self.groups(request)?;
            // Axis length is checked before narrowing. Each pack and output vector
            // holds at most MAX_BATCH_ELEMENTS elements, independently of tensor size.
            let width = *self.shape.last().expect("validated Fourier rank") as usize;
            let mut groups = groups.into_iter().peekable();
            let mut values = vec![S::DType::ZERO; request.len()];
            let mut support =
                matches!(self.layout(), Layout::Sparse { .. }).then(|| vec![0; request.len()]);

            while groups.peek().is_some() {
                let pack: Vec<_> = groups.by_ref().take(MAX_BATCH_ELEMENTS / width).collect();
                let rectangles = pack
                    .iter()
                    .map(|(prefix, _)| {
                        let axes = prefix
                            .iter()
                            .map(|&i| Axis::range(i, 1))
                            .chain(std::iter::once(Axis::range(0, width as u64)))
                            .collect();
                        Slice::new(self.source.shape(), axes)?.rectangle()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let input = BatchRequest::rectangles(rectangles)?;
                let mut batch = expression::evaluate_batch(&self.source, &input).await?;
                if let Some(mask) = &batch.support {
                    for (value, &supported) in batch.values.iter_mut().zip(mask) {
                        if supported == 0 {
                            *value = S::DType::ZERO;
                        }
                    }
                }
                let array = expression::batch_array(batch.values)?
                    .reshape(ha_ndarray::shape![pack.len(), width])?;
                let transformed = self
                    .op
                    .apply(ArrayAccess::from(array))?
                    .buffer()?
                    .to_slice()?
                    .into_vec();
                if transformed.len() != input.len() {
                    return Err(Error::InvalidLayout(format!(
                        "{} output: expected {} elements, got {}",
                        O::NAME,
                        input.len(),
                        transformed.len()
                    )));
                }

                for (group, (_, destinations)) in pack.into_iter().enumerate() {
                    let start = group * width;
                    let populated = batch
                        .support
                        .as_ref()
                        .is_none_or(|mask| mask[start..start + width].iter().any(|&v| v != 0));
                    for (destination, frequency) in destinations {
                        if populated {
                            values[destination] = transformed[start + frequency];
                        }
                        if let Some(mask) = &mut support {
                            mask[destination] = u8::from(populated);
                        }
                    }
                }
            }

            Ok(Batch {
                array: expression::batch_array(values)?,
                support,
            })
        })
    }
}

impl<S, O> TensorRead for FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
{
    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, Self::DType>> {
        expression::coordinate_blocks(self)
    }

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Ok(
                expression::evaluate_batch(self, &BatchRequest::point(coord))
                    .await?
                    .values[0],
            )
        })
    }

    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        let coords = request::linear_requests(self.shape())?;

        Ok(expression::ordered_batches(self, coords)
            .map_ok(|(_, batch)| batch.values)
            .boxed())
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let coords = crate::traits::sparse_coords(self, range, requested_order)?;

            Ok(
                expression::ordered_batches(self, request::explicit_requests(coords))
                    .and_then(move |(coords, values)| async move {
                        Ok(futures::stream::iter(expression::sparse_elements(
                            coords,
                            values,
                            self.shape(),
                        )?))
                    })
                    .try_flatten()
                    .boxed(),
            )
        })
    }
}

impl<S, O: Send + Sync> TensorTransform for FourierView<S, O>
where
    S: TensorGeometry,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.reshape(shape)?,
            ..self
        })
    }

    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.broadcast(shape)?,
            ..self
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.slice(range)?,
            ..self
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.transpose(permutation)?,
            ..self
        })
    }

    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.flip(axis)?,
            ..self
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.squeeze(axes)?,
            ..self
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.unsqueeze(axes)?,
            ..self
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex64;
    use crate::test_support::{cleanup, fixture};

    #[tokio::test]
    async fn complete_groups_pack_without_dense_support_masks() {
        let width = 7;
        let count = MAX_BATCH_ELEMENTS / width + 1;
        let (root, tensor) = fixture::source(
            "fft_packs",
            smallvec::smallvec![count as u64, width as u64],
            Layout::Dense,
            31,
            2048,
            std::iter::repeat_n(Complex64::ONE, count * width),
        )
        .await;
        tensor.sync().await.unwrap();
        fn files(path: &std::path::Path) -> std::collections::BTreeSet<std::path::PathBuf> {
            let mut paths = std::collections::BTreeSet::new();
            for entry in std::fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    paths.extend(files(&path));
                } else {
                    paths.insert(path);
                }
            }
            paths
        }
        let before = files(&root);
        let view = tensor.view().fft().await.unwrap();
        let coords = (0..count)
            .map(|i| vec![i as u64, 0])
            .chain([vec![0, 0], vec![0, 1]])
            .collect();
        let request = BatchRequest::explicit(coords).unwrap();

        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                let batch = expression::evaluate_batch(&view, &request).await.unwrap();
                assert!(batch.support.is_none());
                assert!(
                    batch.values[..count + 1]
                        .iter()
                        .all(|&v| v == Complex64::new(width as f64, 0.))
                );
                let ku = 32. * width as f64 * f64::EPSILON / 2.;
                assert!(batch.values[count + 1].norm() <= ku / (1. - ku) * width as f64);
                crate::read_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    assert_eq!(m.slice_requests, 3); // Outer request plus two complete packs.
                    assert_eq!(m.requested, count * width); // Duplicate outputs share source reads.
                });
            })
            .await;
        assert!(
            Expression::preferred_requests(&view, view.shape())
                .unwrap()
                .is_none()
        );
        assert_eq!(
            files(&root),
            before,
            "evaluation must not create result storage"
        );
        cleanup(&root).await;
    }

    #[test]
    fn invalid_axes_fail_before_execution() {
        for shape in [&[][..], &[0][..]] {
            assert!(matches!(
                validate_axes(shape, 1, "fft"),
                Err(Error::InvalidLayout(_))
            ));
        }
        assert!(validate_axes(&[MAX_BATCH_ELEMENTS as u64], 1, "fft").is_ok());
        assert!(matches!(
            validate_axes(&[MAX_BATCH_ELEMENTS as u64 + 1], 1, "fft"),
            Err(Error::Unsupported(_))
        ));
        assert!(validate_axes(&[u64::MAX, 2], 1, "fft").is_err());
    }
}
