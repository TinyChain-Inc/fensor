use std::sync::Arc;

use ha_ndarray::{Axes, AxisRange, Range, Shape, Strides};

use crate::{AxisContribSchema, Error, Result, TensorSchema, ViewSchema, schema};

#[derive(Clone)]
pub struct TensorView {
    base_rank: usize,
    base_offset: i64,
    axes: Vec<AxisContrib>,
}

#[derive(Clone)]
pub(crate) enum AxisContrib {
    Stride(i64),
    Gather(Arc<[i64]>),
}

impl TensorView {
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    pub fn base_rank(&self) -> usize {
        self.base_rank
    }

    pub fn identity(schema: &TensorSchema) -> Self {
        Self {
            base_rank: schema.rank(),
            base_offset: 0,
            axes: schema
                .strides()
                .iter()
                .map(|&s| AxisContrib::Stride(s as i64))
                .collect(),
        }
    }

    pub fn transpose(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.len() != self.axes.len() {
            return Err(Error::InvalidLayout(
                "transpose permutation rank must match tensor rank".to_string(),
            ));
        }

        let mut seen = vec![false; self.axes.len()];
        let mut axes = Vec::with_capacity(self.axes.len());

        for permuted_axis in permutation {
            if *permuted_axis >= self.axes.len() || seen[*permuted_axis] {
                return Err(Error::InvalidLayout(
                    "transpose permutation must be a valid axis permutation".to_string(),
                ));
            }

            seen[*permuted_axis] = true;
            axes.push(self.axes[*permuted_axis].clone());
        }

        Ok(Self {
            base_rank: self.base_rank,
            base_offset: self.base_offset,
            axes,
        })
    }

    pub fn slice(&self, shape: &[usize], range: &Range) -> Result<(Self, Shape, Strides)> {
        if range.len() != shape.len() {
            return Err(Error::InvalidLayout(
                "slice range rank must match tensor rank".to_string(),
            ));
        }

        let mut new_axes: Vec<AxisContrib> = Vec::with_capacity(self.axes.len());
        let mut new_base_offset = self.base_offset;
        let mut new_shape = Shape::with_capacity(self.axes.len());
        let mut new_strides = Strides::with_capacity(self.axes.len());

        for (axis_index, (bound, dim)) in range.iter().zip(shape.iter()).enumerate() {
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
                            new_strides.push(new_s.unsigned_abs() as usize);
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
                            new_strides.push(0);
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
                    new_strides.push(0);
                }
            }
        }

        Ok((
            Self {
                base_rank: self.base_rank,
                base_offset: new_base_offset,
                axes: new_axes,
            },
            new_shape,
            new_strides,
        ))
    }

    pub fn to_schema(&self) -> Result<ViewSchema> {
        Ok(ViewSchema {
            base_rank: self.base_rank,
            base_offset: self.base_offset,
            axes: self
                .axes
                .iter()
                .map(|a| match a {
                    AxisContrib::Stride(s) => AxisContribSchema::Stride(*s),
                    AxisContrib::Gather(g) => {
                        AxisContribSchema::Gather(g.iter().copied().collect())
                    }
                })
                .collect(),
        })
    }

    pub fn from_schema(schema: &ViewSchema) -> Result<Self> {
        let axes = schema
            .axes
            .iter()
            .map(|a| match a {
                AxisContribSchema::Stride(s) => Ok(AxisContrib::Stride(*s)),
                AxisContribSchema::Gather(g) => Ok(AxisContrib::Gather(
                    g.iter().copied().collect::<Vec<_>>().into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            base_rank: schema.base_rank,
            base_offset: schema.base_offset,
            axes,
        })
    }

    pub fn reshape(&self, current_shape: &[usize], new_shape: &[usize]) -> Result<Self> {
        if !self.is_c_contiguous(current_shape) {
            return Err(Error::Unsupported(
                "reshape requires a C-contiguous view; copy the tensor before reshaping \
                a transposed, flip, step-strided, or gather-sliced view"
                    .to_string(),
            ));
        }
        Ok(Self {
            base_rank: self.base_rank,
            base_offset: self.base_offset,
            axes: schema::contiguous_strides(new_shape)
                .iter()
                .map(|&s| AxisContrib::Stride(s as i64))
                .collect(),
        })
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

    fn is_c_contiguous(&self, shape: &[usize]) -> bool {
        if self.axes.len() != shape.len() {
            return false;
        }
        let expected = schema::contiguous_strides(shape);
        self.axes
            .iter()
            .zip(expected.iter())
            .all(|pair| match pair {
                (AxisContrib::Stride(s), e) => *s >= 0 && *s as usize == *e,
                _ => false,
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ha_ndarray::{range, shape};

    fn schema(shape: &[usize]) -> TensorSchema {
        TensorSchema::dense(
            shape.iter().copied().collect(),
            shape.iter().copied().collect(),
        )
        .expect("schema")
    }

    // -- flat_offset correctness ---------------------------------------------------

    #[test]
    fn flat_offset_identity() {
        // [4,5,6], strides [30,6,1], coord [2,1,3] → k = 2*30 + 1*6 + 3*1 = 69
        let schema = schema(&[4, 5, 6]);
        let view = TensorView::identity(&schema);
        assert_eq!(view.flat_offset(&[2, 1, 3]).expect("offset"), 69);
    }

    #[test]
    fn flat_offset_reshape_rank_reducing() {
        // [2,3,4]→[6,4]: new strides [4,1], coord [3,2] → k = 3*4 + 2*1 = 14
        let schema = schema(&[2, 3, 4]);
        let view = TensorView::identity(&schema);
        let reshaped = view.reshape(schema.shape(), &[6, 4]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[3, 2]).expect("offset"), 14);
    }

    #[test]
    fn flat_offset_reshape_rank_increasing() {
        // [6,4]→[2,3,4]: new strides [12,4,1], coord [1,0,2] → k = 1*12 + 0*4 + 2*1 = 14
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let reshaped = view.reshape(schema.shape(), &[2, 3, 4]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[1, 0, 2]).expect("offset"), 14);
    }

    #[test]
    fn flat_offset_reshape_same_rank() {
        // [6,4]→[4,6]: new strides [6,1], coord [2,3] → k = 2*6 + 3*1 = 15
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let reshaped = view.reshape(schema.shape(), &[4, 6]).expect("reshape");
        assert_eq!(reshaped.flat_offset(&[2, 3]).expect("offset"), 15);
    }

    // -- slice / transpose flat_offset correctness --------------------------------

    #[test]
    fn slice_then_resolve() {
        // [4,5,6], strides [30,6,1]
        // At(2): base_offset += 2*30 = 60, axis dropped
        // In(1,5,2): base_offset += 1*6 = 6, axis → Stride(12), extent 2
        // Of([0,3,4]): Gather([0,3,4]), extent 3
        // flat_offset([1,2]) = 66 + 1*12 + Gather[2]=4 = 82
        let schema = schema(&[4, 5, 6]);
        let view = TensorView::identity(&schema);
        let range = range![
            AxisRange::At(2),
            AxisRange::In(1, 5, 2),
            AxisRange::Of(shape![0, 3, 4])
        ];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        assert_eq!(new_shape.as_slice(), &[2, 3]);
        assert_eq!(sliced.flat_offset(&[1, 2]).expect("offset"), 82);
    }

    #[test]
    fn transpose_arbitrary_permutation() {
        // [3,4,5], strides [20,5,1], perm [2,0,1] → axes [Stride(1),Stride(20),Stride(5)]
        // flat_offset([1,2,3]) = 1*1 + 2*20 + 3*5 = 56
        let schema = schema(&[3, 4, 5]);
        let view = TensorView::identity(&schema);
        let transposed = view.transpose(&[2, 0, 1]).expect("transpose");
        assert_eq!(transposed.flat_offset(&[1, 2, 3]).expect("offset"), 56);
    }

    #[test]
    fn compose_transpose_and_slice() {
        // [3,4,5], transpose [2,0,1] → axes [Stride(1),Stride(20),Stride(5)], shape [5,3,4]
        // In(1,5,2) on axis 0 (Stride(1)): base_offset+=1, Stride(2), extent 2
        // At(2) on axis 1 (Stride(20)): base_offset+=40, dropped
        // In(0,4,2) on axis 2 (Stride(5)): base_offset+=0, Stride(10), extent 2
        // base_offset=41, axes=[Stride(2),Stride(10)], shape=[2,2]
        // flat_offset([1,1]) = 41 + 2 + 10 = 53
        let schema = schema(&[3, 4, 5]);
        let view = TensorView::identity(&schema);
        let transposed = view.transpose(&[2, 0, 1]).expect("transpose");
        let shape: Shape = shape![5, 3, 4];
        let range: Range = range![
            AxisRange::In(1, 5, 2),
            AxisRange::At(2),
            AxisRange::In(0, 4, 2)
        ];
        let (sliced, new_shape, _) = transposed.slice(&shape, &range).expect("slice");
        assert_eq!(new_shape.as_slice(), &[2, 2]);
        assert_eq!(sliced.flat_offset(&[1, 1]).expect("offset"), 53);
    }

    // -- schema roundtrip ---------------------------------------------------------

    #[test]
    fn view_schema_roundtrip() {
        // [5,4,3], strides [12,3,1], perm [2,0,1] → axes [Stride(1),Stride(12),Stride(3)]
        // flat_offset([1,2,3]) = 1*1 + 2*12 + 3*3 = 34
        let schema = schema(&[5, 4, 3]);
        let identity = TensorView::identity(&schema);
        let transposed = identity.transpose(&[2, 0, 1]).expect("transpose");
        let view_schema = transposed.to_schema().expect("schema");
        let restored = TensorView::from_schema(&view_schema).expect("restore");
        assert_eq!(restored.flat_offset(&[1, 2, 3]).expect("coord"), 34);
    }

    #[test]
    fn view_schema_roundtrip_with_reshape() {
        // ViewSchema carries only base_rank, base_offset, axes — no shape/strides duplication
        // [6,4]→[2,3,4]: axes [Stride(12),Stride(4),Stride(1)], coord [1,0,2] → k=14
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let reshaped = view.reshape(schema.shape(), &[2, 3, 4]).expect("reshape");
        let view_schema = reshaped.to_schema().expect("to_schema");
        let restored = TensorView::from_schema(&view_schema).expect("from_schema");
        assert_eq!(restored.flat_offset(&[1, 0, 2]).expect("offset"), 14);
    }

    // -- invalid reshape chain guards ---------------------------------------------

    #[test]
    fn transpose_then_reshape_rejected() {
        // transposed [3,4]: axes=[Stride(1),Stride(4)]; contiguous_strides([4,3])=[3,1]; 1≠3
        let schema = schema(&[3, 4]);
        let view = TensorView::identity(&schema);
        let transposed = view.transpose(&[1, 0]).expect("transpose");
        assert!(matches!(
            transposed.reshape(&[4, 3], &[12]),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn strided_slice_then_reshape_rejected() {
        // In(0,6,2) on axis 0 of [6,4]: axes=[Stride(8),Stride(1)]; contiguous_strides([3,4])=[4,1]; 8≠4
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let range = range![AxisRange::In(0, 6, 2), AxisRange::In(0, 4, 1)];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        assert!(matches!(
            sliced.reshape(&new_shape, &[12]),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn gather_slice_then_reshape_rejected() {
        // Of([0,2,4]) on axis 0: Gather axis present → rejected
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let range = range![AxisRange::Of(shape![0, 2, 4]), AxisRange::In(0, 4, 1)];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        assert!(matches!(
            sliced.reshape(&new_shape, &[12]),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn at_slice_non_first_axis_then_reshape_rejected() {
        // At(0) on axis 1 of [3,4]: axes=[Stride(4)]; contiguous_strides([3])=[1]; 4≠1
        let schema = schema(&[3, 4]);
        let view = TensorView::identity(&schema);
        let range = range![AxisRange::In(0, 3, 1), AxisRange::At(0)];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        assert!(matches!(
            sliced.reshape(&new_shape, &[3]),
            Err(Error::Unsupported(_))
        ));
    }

    // -- valid reshape chains -----------------------------------------------------

    #[test]
    fn step1_slice_then_reshape_valid() {
        // In(1,4,1) on axis 0 of [6,4]: strides unchanged [4,1], contiguous for [3,4] ✓
        // base_offset=4; reshape [3,4]→[12]: axes=[Stride(1)]
        // flat_offset([5]) = 4 + 5*1 = 9
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let range = range![AxisRange::In(1, 4, 1), AxisRange::In(0, 4, 1)];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        let reshaped = sliced
            .reshape(&new_shape, &[12])
            .expect("reshape must succeed");
        assert_eq!(reshaped.flat_offset(&[5]).expect("offset"), 9);
    }

    #[test]
    fn at_first_axis_then_reshape_valid() {
        // At(1) on axis 0 of [3,4]: base_offset=4, axes=[Stride(1)], contiguous for [4] ✓
        // reshape [4]→[2,2]: axes=[Stride(2),Stride(1)]
        // flat_offset([1,1]) = 4 + 1*2 + 1*1 = 7
        let schema = schema(&[3, 4]);
        let view = TensorView::identity(&schema);
        let range = range![AxisRange::At(1), AxisRange::In(0, 4, 1)];
        let (sliced, new_shape, _) = view.slice(schema.shape(), &range).expect("slice");
        let reshaped = sliced
            .reshape(&new_shape, &[2, 2])
            .expect("reshape must succeed");
        assert_eq!(reshaped.flat_offset(&[1, 1]).expect("offset"), 7);
    }

    #[test]
    fn reshape_then_slice_flat_offset() {
        // reshape [6,4]→[2,3,4]: axes=[Stride(12),Stride(4),Stride(1)]
        // slice In(0,2,1) / In(0,2,1) / In(0,4,1); shape=[2,2,4]
        // flat_offset([1,1,2]) = 0 + 1*12 + 1*4 + 2*1 = 18
        let schema = schema(&[6, 4]);
        let view = TensorView::identity(&schema);
        let reshaped_shape = [2usize, 3, 4];
        let reshaped = view
            .reshape(schema.shape(), &reshaped_shape)
            .expect("reshape");
        let range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 4, 1)
        ];
        let (sliced, new_shape, _) = reshaped.slice(&reshaped_shape, &range).expect("slice");
        assert_eq!(new_shape.as_slice(), &[2, 2, 4]);
        assert_eq!(sliced.flat_offset(&[1, 1, 2]).expect("offset"), 18);
    }
}
