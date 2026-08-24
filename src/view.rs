#[cfg(test)]
mod tests;

use std::{iter, sync::Arc};

use ha_ndarray::{Axes, AxisRange, Range, Shape};
use smallvec::SmallVec;

use crate::error::{Error, Result};
use crate::schema::{self, Layout, TensorViewShape};
use crate::tensor::{Tensor, TensorElement, TensorFileEntry};
use crate::traits::{
    BoxFuture, TensorArray, TensorGeometry, TensorRead, TensorTransform, TensorViewSemantics,
    TensorWrite,
};

use crate::{PORTABLE_INLINE_RANK, stream, validate};

#[derive(Clone)]
pub struct TensorView<'t, FE, T> {
    tensor: &'t Tensor<FE, T>,
    base_offset: i64,
    axes: SmallVec<[AxisContrib; PORTABLE_INLINE_RANK]>,
    shape: TensorViewShape,
}

#[derive(Clone)]
pub(crate) enum AxisContrib {
    Stride(i64),
    Gather(Arc<[i64]>),
    Broadcast(i64),
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
                AxisContrib::Broadcast(constant) => *constant,
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

fn slice_bound_at(
    current: &AxisContrib,
    dim: usize,
    index: usize,
    axis_index: usize,
) -> Result<i64> {
    if index >= dim {
        return Err(Error::InvalidLayout(format!(
            "slice bound at axis {axis_index} is out of bounds"
        )));
    }
    match current {
        AxisContrib::Stride(s) => Ok((index as i64) * s),
        AxisContrib::Broadcast(c) => Ok(*c),
        AxisContrib::Gather(g) => g.get(index).copied().ok_or_else(|| {
            Error::InvalidLayout(format!(
                "slice bound at axis {axis_index} out of bounds for gather"
            ))
        }),
    }
}

fn slice_bound_in(
    current: &AxisContrib,
    dim: usize,
    start: usize,
    stop: usize,
    step: usize,
    axis_index: usize,
) -> Result<(i64, AxisContrib, usize)> {
    if step == 0 || start > stop || stop > dim {
        return Err(Error::InvalidLayout(format!(
            "slice bound at axis {axis_index} is out of bounds"
        )));
    }
    let extent = if start == stop {
        0
    } else {
        (stop - start).div_ceil(step)
    };

    let (offset_delta, axis) = match current {
        AxisContrib::Stride(s) => ((start as i64) * s, AxisContrib::Stride(s * (step as i64))),
        AxisContrib::Broadcast(c) => (0, AxisContrib::Broadcast(*c)),
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
            (0, AxisContrib::Gather(offsets.into()))
        }
    };

    Ok((offset_delta, axis, extent))
}

fn slice_bound_of(
    current: &AxisContrib,
    dim: usize,
    indices: &[usize],
    axis_index: usize,
) -> Result<AxisContrib> {
    if indices.iter().any(|index| *index >= dim) {
        return Err(Error::InvalidLayout(format!(
            "slice bound at axis {axis_index} is out of bounds"
        )));
    }
    match current {
        AxisContrib::Broadcast(c) => Ok(AxisContrib::Broadcast(*c)),
        AxisContrib::Stride(s) => {
            let offsets: Vec<i64> = indices.iter().map(|idx| (*idx as i64) * s).collect();
            Ok(AxisContrib::Gather(offsets.into()))
        }
        AxisContrib::Gather(g) => {
            let offsets = indices
                .iter()
                .map(|idx| {
                    g.get(*idx).copied().ok_or_else(|| {
                        Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} out of bounds for gather"
                        ))
                    })
                })
                .collect::<Result<Vec<i64>>>()?;
            Ok(AxisContrib::Gather(offsets.into()))
        }
    }
}

fn flip_axis_contrib(current: &AxisContrib, dim: usize) -> (i64, AxisContrib) {
    match current {
        AxisContrib::Stride(s) => (((dim as i64) - 1) * s, AxisContrib::Stride(-s)),
        AxisContrib::Broadcast(c) => (0, AxisContrib::Broadcast(*c)),
        AxisContrib::Gather(g) => (0, AxisContrib::Gather(g.iter().rev().copied().collect())),
    }
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
            .any(|a| matches!(a, AxisContrib::Gather(_) | AxisContrib::Broadcast(_)))
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

    fn broadcast(self, shape: TensorViewShape) -> Result<Self> {
        if shape.len() < self.shape.len() {
            return Err(Error::InvalidLayout(format!(
                "cannot broadcast shape {:?} to a lower rank shape {:?}",
                self.shape, shape
            )));
        }

        let rank_diff = shape.len() - self.shape.len();
        let mut axes = SmallVec::with_capacity(shape.len());
        axes.extend(iter::repeat_n(AxisContrib::Broadcast(0), rank_diff));

        for (axis_index, (&old_dim, &new_dim)) in
            self.shape.iter().zip(shape[rank_diff..].iter()).enumerate()
        {
            if old_dim != 1 && old_dim != new_dim {
                return Err(Error::InvalidLayout(format!(
                    "cannot broadcast axis {axis_index} from dimension {old_dim} to {new_dim}"
                )));
            }

            let axis_contrib = if old_dim == 1usize {
                AxisContrib::Broadcast(match &self.axes[axis_index] {
                    AxisContrib::Stride(_) => 0i64,
                    AxisContrib::Broadcast(c) => *c,
                    AxisContrib::Gather(g) => *g.first().ok_or_else(|| {
                        Error::InvalidLayout(format!("axis {axis_index} has an empty gather table"))
                    })?,
                })
            } else {
                self.axes[axis_index].clone()
            };

            axes.push(axis_contrib);
        }

        Ok(Self {
            tensor: self.tensor,
            base_offset: self.base_offset,
            axes,
            shape,
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        if range.len() != self.shape.len() {
            return Err(Error::InvalidLayout(
                "slice range rank must match tensor rank".to_string(),
            ));
        }

        let mut new_axes = SmallVec::with_capacity(self.axes.len());
        let mut new_base_offset = self.base_offset;
        let mut new_shape = Shape::with_capacity(self.axes.len());

        for (axis_index, (bound, &dim)) in range.iter().zip(self.shape.iter()).enumerate() {
            let current = &self.axes[axis_index];

            match bound {
                AxisRange::At(i) => {
                    new_base_offset += slice_bound_at(current, dim, *i, axis_index)?;
                }
                AxisRange::In(start, stop, step) => {
                    let (offset_delta, axis, extent) =
                        slice_bound_in(current, dim, *start, *stop, *step, axis_index)?;
                    new_base_offset += offset_delta;
                    new_axes.push(axis);
                    new_shape.push(extent);
                }
                AxisRange::Of(indices) => {
                    new_axes.push(slice_bound_of(current, dim, indices, axis_index)?);
                    new_shape.push(indices.len());
                }
            }
        }

        if new_shape.contains(&0) {
            return Err(Error::InvalidLayout(
                "slice produced a zero-extent axis".to_string(),
            ));
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
        let mut axes = SmallVec::with_capacity(self.axes.len());
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

    fn flip(self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() {
            return Err(Error::InvalidLayout(format!(
                "flip axis {axis} is out of bounds for rank {}",
                self.shape.len()
            )));
        }

        let dim = self.shape[axis];
        let (offset_delta, new_axis) = flip_axis_contrib(&self.axes[axis], dim);

        let mut axes = self.axes;
        axes[axis] = new_axis;

        Ok(Self {
            tensor: self.tensor,
            base_offset: self.base_offset + offset_delta,
            axes,
            shape: self.shape,
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        let ndim = self.shape.len();

        if axes.is_empty() {
            return Err(Error::InvalidLayout(
                "squeeze requires a non-empty list of axes".to_string(),
            ));
        }
        if axes.len() == ndim {
            return Err(Error::InvalidLayout(
                "squeeze cannot remove every axis; rank-0 tensors are not supported".to_string(),
            ));
        }

        let mut remove = vec![false; ndim];
        for &axis in axes.iter() {
            if axis >= ndim {
                return Err(Error::InvalidLayout(format!(
                    "squeeze axis {axis} is out of bounds for rank {ndim}"
                )));
            }
            if remove[axis] {
                return Err(Error::InvalidLayout(format!(
                    "squeeze axis {axis} specified more than once"
                )));
            }
            if self.shape[axis] != 1 {
                return Err(Error::InvalidLayout(format!(
                    "cannot squeeze axis {axis} with dimension {}",
                    self.shape[axis]
                )));
            }
            remove[axis] = true;
        }

        let mut base_offset = self.base_offset;
        let mut axes_out = SmallVec::with_capacity(ndim - axes.len());
        let mut shape_out = Shape::with_capacity(ndim - axes.len());

        for (i, &removed) in remove.iter().enumerate().take(ndim) {
            if removed {
                base_offset += slice_bound_at(&self.axes[i], 1, 0, i)?;
            } else {
                axes_out.push(self.axes[i].clone());
                shape_out.push(self.shape[i]);
            }
        }

        Ok(Self {
            tensor: self.tensor,
            base_offset,
            axes: axes_out,
            shape: shape_out,
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        let old_ndim = self.shape.len();

        if axes.is_empty() {
            return Err(Error::InvalidLayout(
                "unsqueeze requires a non-empty list of axes".to_string(),
            ));
        }

        let mut insert_before = vec![false; old_ndim];
        for &axis in axes.iter() {
            if axis >= old_ndim {
                return Err(Error::InvalidLayout(format!(
                    "unsqueeze axis {axis} is out of bounds for rank {old_ndim}"
                )));
            }
            if insert_before[axis] {
                return Err(Error::InvalidLayout(format!(
                    "unsqueeze axis {axis} specified more than once"
                )));
            }
            insert_before[axis] = true;
        }

        let new_ndim = old_ndim + axes.len();
        let mut layout: SmallVec<[Option<usize>; PORTABLE_INLINE_RANK]> =
            SmallVec::with_capacity(new_ndim);
        for (i, &inserted) in insert_before.iter().enumerate().take(old_ndim) {
            if inserted {
                layout.push(None);
            }
            layout.push(Some(i));
        }

        let shape_out: Shape = layout
            .iter()
            .map(|slot| match slot {
                None => 1,
                Some(i) => self.shape[*i],
            })
            .collect();

        let contiguous = schema::contiguous_strides(&shape_out)?;

        let axes_out: SmallVec<[AxisContrib; PORTABLE_INLINE_RANK]> = layout
            .iter()
            .enumerate()
            .map(|(pos, slot)| match slot {
                None => AxisContrib::Stride(contiguous[pos] as i64),
                Some(i) => self.axes[*i].clone(),
            })
            .collect();

        Ok(Self {
            tensor: self.tensor,
            base_offset: self.base_offset,
            axes: axes_out,
            shape: shape_out,
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
            if !self.supports_write_through() {
                return Err(Error::Unsupported(
                    "this view does not support write-through to the base tensor".to_string(),
                ));
            }
            let base_coord = self.resolve_base_coord(coord)?;
            self.tensor.write_value(&base_coord, value).await
        })
    }
}
