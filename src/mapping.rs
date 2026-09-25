//! Coordinate mapping shared by geometric storage and reduction views.

use std::{iter, sync::Arc};

use ha_ndarray::{Axes, AxisRange, Range, Shape};
use smallvec::SmallVec;

use crate::schema::{self, TensorViewShape};
use crate::{Error, PORTABLE_INLINE_RANK, Result, validate};

#[derive(Clone)]
pub(crate) struct CoordinateMap {
    pub base_offset: i64,
    pub axes: SmallVec<[AxisContrib; PORTABLE_INLINE_RANK]>,
    pub shape: TensorViewShape,
}

#[derive(Clone)]
pub(crate) enum AxisContrib {
    Stride(i64),
    Gather(GatherOffsets),
    Broadcast(i64),
}

/// A separable forward slice: base coordinates are fixed or advance on one axis.
/// This metadata is rank-sized; unsupported affine reshapes retain logical reads.
pub(crate) struct StorageSlice {
    pub(crate) origins: Vec<usize>,
    pub(crate) axes: Vec<(usize, usize)>,
}

impl StorageSlice {
    pub(crate) fn identity(shape: &[usize]) -> Self {
        Self {
            origins: vec![0; shape.len()],
            axes: (0..shape.len()).map(|i| (i, 1)).collect(),
        }
    }

    pub(crate) fn bounds(&self, regions: &[(usize, usize)]) -> Option<Vec<(usize, usize)>> {
        for (base, (&origin, &(lo, hi))) in self.origins.iter().zip(regions).enumerate() {
            if !self.axes.iter().any(|(axis, _)| *axis == base) && !(lo..hi).contains(&origin) {
                return None;
            }
        }

        Some(
            self.axes
                .iter()
                .map(|&(base, step)| {
                    let (lo, hi) = regions[base];
                    let origin = self.origins[base];
                    (
                        lo.saturating_sub(origin).div_ceil(step),
                        hi.saturating_sub(origin).div_ceil(step),
                    )
                })
                .collect(),
        )
    }
}

/// Caller-selection-sized metadata. Slices and reversals share the table.
#[derive(Clone)]
pub(crate) struct GatherOffsets {
    offsets: Arc<[i64]>,
    start: usize,
    step: usize,
    reversed: bool,
    len: usize,
}

impl From<Vec<i64>> for GatherOffsets {
    fn from(offsets: Vec<i64>) -> Self {
        Self {
            len: offsets.len(),
            offsets: offsets.into(),
            start: 0,
            step: 1,
            reversed: false,
        }
    }
}

impl GatherOffsets {
    fn index(&self, index: usize) -> Option<usize> {
        if index >= self.len {
            return None;
        }

        let delta = index.checked_mul(self.step)?;
        if self.reversed {
            self.start.checked_sub(delta)
        } else {
            self.start.checked_add(delta)
        }
    }

    fn get(&self, index: usize) -> Option<&i64> {
        self.offsets.get(self.index(index)?)
    }

    fn first(&self) -> Option<&i64> {
        self.get(0)
    }

    fn slice(&self, start: usize, step: usize, len: usize) -> Result<Self> {
        let start = if len == 0 {
            self.start
        } else {
            self.index(start)
                .ok_or_else(|| Error::InvalidLayout("gather slice out of bounds".into()))?
        };
        let step = if len <= 1 {
            1
        } else {
            self.step
                .checked_mul(step)
                .ok_or_else(|| Error::InvalidLayout("gather stride overflow".into()))?
        };
        Ok(Self {
            offsets: self.offsets.clone(),
            start,
            step,
            reversed: self.reversed,
            len,
        })
    }

    fn flipped(&self) -> Self {
        let start = if self.len == 0 {
            self.start
        } else {
            // Valid descriptors address only entries in the shared table.
            self.index(self.len - 1).expect("valid gather descriptor")
        };
        Self {
            start,
            reversed: !self.reversed,
            ..self.clone()
        }
    }
}

impl CoordinateMap {
    pub(crate) fn storage_slice(
        &self,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Option<StorageSlice>> {
        let mut origins = Vec::new();
        self.resolve_into(&vec![0; self.shape.len()], shape, strides, &mut origins)?;
        let origins: Vec<_> = origins.into_iter().map(|v| v as usize).collect();
        let mut axes: Vec<(usize, usize)> = Vec::with_capacity(self.axes.len());

        for (contribution, &dim) in self.axes.iter().zip(&self.shape) {
            let AxisContrib::Stride(stride) = contribution else {
                return Ok(None);
            };
            let Ok(stride) = usize::try_from(*stride) else {
                return Ok(None);
            };
            if stride == 0 {
                return Ok(None);
            }

            let base = (0..shape.len()).rev().find(|&base| {
                !axes.iter().any(|(axis, _)| *axis == base)
                    && stride.is_multiple_of(strides[base])
                    && (stride / strides[base])
                        .checked_mul(dim - 1)
                        .and_then(|delta| origins[base].checked_add(delta))
                        .is_some_and(|last| last < shape[base])
            });
            let Some(base) = base else {
                return Ok(None);
            };
            axes.push((base, stride / strides[base]));
        }

        Ok(Some(StorageSlice { origins, axes }))
    }

    pub fn identity(shape: Shape, strides: &[usize]) -> Self {
        Self {
            base_offset: 0,
            axes: strides
                .iter()
                .map(|s| AxisContrib::Stride(*s as i64))
                .collect(),
            shape,
        }
    }

    pub fn resolve(&self, coord: &[u64], shape: &[usize], strides: &[usize]) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        self.resolve_into(coord, shape, strides, &mut out)?;
        Ok(out)
    }

    pub fn is_identity(&self, shape: &[usize], strides: &[usize]) -> bool {
        self.base_offset == 0 && self.shape.as_slice() == shape && self.axes.len() == strides.len()
            && self.axes.iter().zip(strides).all(|(axis,stride)| matches!(axis, AxisContrib::Stride(s) if i64::try_from(*stride).ok() == Some(*s)))
    }

    /// Structural affine description; gather tables deliberately retain cursor mapping.
    pub(crate) fn affine(&self) -> Result<Option<(i128, Vec<i128>)>> {
        if self.axes.len() != self.shape.len() {
            return Err(Error::InvalidCoord("mapping rank mismatch".into()));
        }

        let mut offset = self.base_offset as i128;
        let mut strides = Vec::with_capacity(self.axes.len());

        for axis in &self.axes {
            match axis {
                AxisContrib::Stride(stride) => strides.push(*stride as i128),
                AxisContrib::Broadcast(constant) => {
                    offset = offset
                        .checked_add(*constant as i128)
                        .ok_or_else(|| Error::InvalidCoord("mapping offset overflow".into()))?;
                    strides.push(0);
                }
                AxisContrib::Gather(_) => return Ok(None),
            }
        }

        Ok(Some((offset, strides)))
    }

    pub fn resolve_into(
        &self,
        coord: &[u64],
        shape: &[usize],
        strides: &[usize],
        out: &mut Vec<u64>,
    ) -> Result<()> {
        validate::validate_coord(&self.shape, coord)?;
        let offset = u64::try_from(self.flat_offset(coord)?)
            .map_err(|_| Error::InvalidCoord("negative linear offset".into()))?;
        let size = shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
            .ok_or_else(|| Error::InvalidCoord("mapping size overflow".into()))?;
        if offset >= size || strides.len() != shape.len() || strides.contains(&0) {
            return Err(Error::InvalidCoord("mapped offset out of bounds".into()));
        }
        out.clear();
        out.extend(
            strides
                .iter()
                .zip(shape)
                .map(|(s, d)| (offset / *s as u64) % *d as u64),
        );
        Ok(())
    }

    pub fn flat_offset(&self, coord: &[u64]) -> Result<i64> {
        if coord.len() != self.axes.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        let mut k: i64 = self.base_offset;

        for (c, axis) in coord.iter().zip(self.axes.iter()) {
            let delta = match axis {
                AxisContrib::Stride(s) => i64::try_from(*c)
                    .ok()
                    .and_then(|c| c.checked_mul(*s))
                    .ok_or_else(|| Error::InvalidCoord("mapping offset overflow".into()))?,
                AxisContrib::Broadcast(constant) => *constant,
                AxisContrib::Gather(offsets) => {
                    let i = usize::try_from(*c)
                        .map_err(|_| Error::InvalidCoord("coord overflows usize".into()))?;
                    *offsets.get(i).ok_or_else(|| {
                        Error::InvalidCoord("coord out of bounds for gather".into())
                    })?
                }
            };
            k = k
                .checked_add(delta)
                .ok_or_else(|| Error::InvalidCoord("mapping offset overflow".into()))?;
        }

        Ok(k)
    }

    pub fn is_c_contiguous(&self) -> bool {
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

    pub(crate) fn reshape(self, shape: Shape) -> Result<Self> {
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
            base_offset: self.base_offset,
            axes: strides
                .iter()
                .map(|&s| AxisContrib::Stride(s as i64))
                .collect(),
            shape,
        })
    }

    pub(crate) fn broadcast(self, shape: TensorViewShape) -> Result<Self> {
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
            base_offset: self.base_offset,
            axes,
            shape,
        })
    }

    pub(crate) fn slice(self, range: Range) -> Result<Self> {
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
            base_offset: new_base_offset,
            axes: new_axes,
            shape: new_shape,
        })
    }

    pub(crate) fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
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
            base_offset: self.base_offset,
            axes,
            shape,
        })
    }

    pub(crate) fn flip(self, axis: usize) -> Result<Self> {
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
            base_offset: self.base_offset + offset_delta,
            axes,
            shape: self.shape,
        })
    }

    pub(crate) fn squeeze(self, axes: Axes) -> Result<Self> {
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
            base_offset,
            axes: axes_out,
            shape: shape_out,
        })
    }

    pub(crate) fn unsqueeze(self, axes: Axes) -> Result<Self> {
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
            base_offset: self.base_offset,
            axes: axes_out,
            shape: shape_out,
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
        AxisContrib::Gather(g) => (0, AxisContrib::Gather(g.slice(start, step, extent)?)),
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
            // A new explicit selection may allocate one offset per supplied index.
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
        AxisContrib::Gather(g) => (0, AxisContrib::Gather(g.flipped())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_transforms_share_the_original_table() {
        let original = GatherOffsets::from(vec![9, 2, 9, 4, 7, 3]);
        let sliced = original.slice(1, 2, 3).unwrap();
        let flipped = sliced.flipped();
        let nested = flipped.slice(1, 1, 2).unwrap();

        for view in [&sliced, &flipped, &nested] {
            assert!(Arc::ptr_eq(&original.offsets, &view.offsets));
        }
        assert_eq!(
            (0..3).map(|i| *flipped.get(i).unwrap()).collect::<Vec<_>>(),
            [3, 4, 2]
        );
        assert_eq!(
            (0..2).map(|i| *nested.get(i).unwrap()).collect::<Vec<_>>(),
            [4, 2]
        );
        assert_eq!(*original.get(0).unwrap(), *original.get(2).unwrap());
        assert_eq!(
            *nested
                .slice(1, usize::MAX, 1)
                .unwrap()
                .flipped()
                .get(0)
                .unwrap(),
            2
        );
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    #[test]
    fn identity_is_structural_and_mapping_reuses_scratch() {
        let shape = ha_ndarray::shape![3, 4];
        let strides = schema::contiguous_strides(&shape).unwrap();
        let map = CoordinateMap::identity(shape.clone(), &strides);
        assert!(map.is_identity(&shape, &strides));
        let flipped = map.clone().flip(1).unwrap();
        assert!(!flipped.is_identity(&shape, &strides));
        assert!(
            flipped
                .clone()
                .flip(1)
                .unwrap()
                .is_identity(&shape, &strides)
        );
        let mut out = Vec::with_capacity(2);
        let ptr = out.as_ptr();

        for row in 0..3 {
            for col in 0..4 {
                flipped
                    .resolve_into(&[row, col], &shape, &strides, &mut out)
                    .unwrap();
                assert_eq!(out, vec![row, 3 - col]);
                assert_eq!(out.as_ptr(), ptr);
            }
        }

        let mut bad = map;
        bad.base_offset = 12;
        assert!(
            bad.resolve_into(&[0, 0], &shape, &strides, &mut out)
                .is_err()
        );
        bad.base_offset = i64::MAX;
        assert!(
            bad.resolve_into(&[2, 3], &shape, &strides, &mut out)
                .is_err()
        );
    }
}
