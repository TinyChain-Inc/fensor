//! Coordinate mapping shared by geometric storage and reduction views.

use std::{iter, sync::Arc};

use smallvec::SmallVec;

use crate::schema::Coord;
use crate::{Axes, AxisRange, Error, PORTABLE_INLINE_RANK, Range, Result, Shape, schema, validate};

#[derive(Clone)]
pub(crate) struct CoordinateMap {
    pub base_offset: i128,
    // Large descriptors stay heap-backed; their gather metadata is shared.
    pub axes: Vec<AxisContrib>,
    pub shape: Shape,
}

#[derive(Clone)]
pub(crate) enum AxisContrib {
    Stride(i128),
    Gather(GatherOffsets),
    Broadcast(i128),
}

/// A separable forward slice: base coordinates are fixed or advance on one axis.
/// This metadata is rank-sized; unsupported affine reshapes retain logical reads.
pub(crate) struct StorageSlice {
    pub(crate) origins: Coord,
    pub(crate) axes: Vec<(usize, u64)>,
}

impl StorageSlice {
    pub(crate) fn identity(shape: &[u64]) -> Self {
        Self {
            origins: Coord::from_elem(0, shape.len()),
            axes: (0..shape.len()).map(|i| (i, 1)).collect(),
        }
    }

    pub(crate) fn bounds(&self, regions: &[(u64, u64)]) -> Option<Vec<(u64, u64)>> {
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
    offsets: Arc<[i128]>,
    start: usize,
    step: usize,
    reversed: bool,
    len: usize,
}

impl From<Vec<i128>> for GatherOffsets {
    fn from(offsets: Vec<i128>) -> Self {
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

    fn get(&self, index: usize) -> Option<&i128> {
        self.offsets.get(self.index(index)?)
    }

    fn first(&self) -> Option<&i128> {
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
        shape: &[u64],
        strides: &[u64],
    ) -> Result<Option<StorageSlice>> {
        let mut origins = Coord::new();
        self.resolve_into(
            &Coord::from_elem(0, self.shape.len()),
            shape,
            strides,
            &mut origins,
        )?;

        let mut axes: Vec<(usize, u64)> = Vec::with_capacity(self.axes.len());

        for (contribution, &dim) in self.axes.iter().zip(&self.shape) {
            let AxisContrib::Stride(stride) = contribution else {
                return Ok(None);
            };
            let Ok(stride) = u64::try_from(*stride) else {
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

    pub fn identity(shape: Shape, strides: &[u64]) -> Self {
        Self {
            base_offset: 0,
            axes: strides
                .iter()
                .map(|s| AxisContrib::Stride(*s as i128))
                .collect(),
            shape,
        }
    }

    pub fn resolve(&self, coord: &[u64], shape: &[u64], strides: &[u64]) -> Result<Vec<u64>> {
        let mut out = Coord::new();
        self.resolve_into(coord, shape, strides, &mut out)?;
        Ok(out.into_vec())
    }

    pub fn is_identity(&self, shape: &[u64], strides: &[u64]) -> bool {
        self.base_offset == 0 && self.shape.as_slice() == shape && self.axes.len() == strides.len()
            && self.axes.iter().zip(strides).all(|(axis,stride)| matches!(axis, AxisContrib::Stride(s) if i128::from(*stride) == *s))
    }

    /// Structural affine description; gather tables deliberately retain cursor mapping.
    pub(crate) fn affine(&self) -> Result<Option<(i128, Vec<i128>)>> {
        if self.axes.len() != self.shape.len() {
            return Err(Error::InvalidCoord("mapping rank mismatch".into()));
        }

        let mut offset = self.base_offset;
        let mut strides = Vec::with_capacity(self.axes.len());

        for axis in &self.axes {
            match axis {
                AxisContrib::Stride(stride) => strides.push(*stride),
                AxisContrib::Broadcast(constant) => {
                    offset = offset
                        .checked_add(*constant)
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
        shape: &[u64],
        strides: &[u64],
        out: &mut Coord,
    ) -> Result<()> {
        validate::validate_coord(&self.shape, coord)?;
        let offset = u64::try_from(self.flat_offset(coord)?)
            .map_err(|_| Error::InvalidCoord("negative linear offset".into()))?;
        let size = crate::schema::checked_product(shape)?;
        if offset >= size || strides.len() != shape.len() || strides.contains(&0) {
            return Err(Error::InvalidCoord("mapped offset out of bounds".into()));
        }
        out.clear();
        out.extend(strides.iter().zip(shape).map(|(s, d)| (offset / *s) % *d));
        Ok(())
    }

    pub fn flat_offset(&self, coord: &[u64]) -> Result<i128> {
        if coord.len() != self.axes.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        let mut k: i128 = self.base_offset;

        for (c, axis) in coord.iter().zip(self.axes.iter()) {
            let delta = match axis {
                AxisContrib::Stride(s) => i128::from(*c)
                    .checked_mul(*s)
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
                (AxisContrib::Stride(s), e) => *s == i128::from(*e),
                _ => false,
            })
    }

    pub(crate) fn reshape(self, shape: Shape) -> Result<Self> {
        let old_size = schema::checked_product(&self.shape)?;
        let new_size = schema::checked_product(&shape)?;
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
                .map(|&s| AxisContrib::Stride(s as i128))
                .collect(),
            shape,
        })
    }

    pub(crate) fn broadcast(self, shape: Shape) -> Result<Self> {
        schema::validate_shape_dims(&shape)?;
        schema::checked_product(&shape)?;
        if shape.len() < self.shape.len() {
            return Err(Error::InvalidLayout(format!(
                "cannot broadcast shape {:?} to a lower rank shape {:?}",
                self.shape, shape
            )));
        }

        let rank_diff = shape.len() - self.shape.len();
        let mut axes = Vec::with_capacity(shape.len());
        axes.extend(iter::repeat_n(AxisContrib::Broadcast(0), rank_diff));

        for (axis_index, (&old_dim, &new_dim)) in
            self.shape.iter().zip(shape[rank_diff..].iter()).enumerate()
        {
            if old_dim != 1 && old_dim != new_dim {
                return Err(Error::InvalidLayout(format!(
                    "cannot broadcast axis {axis_index} from dimension {old_dim} to {new_dim}"
                )));
            }

            let axis_contrib = if old_dim == 1u64 {
                AxisContrib::Broadcast(match &self.axes[axis_index] {
                    AxisContrib::Stride(_) => 0i128,
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

        let mut new_axes = Vec::with_capacity(self.axes.len());
        let mut new_base_offset = self.base_offset;
        let mut new_shape = Shape::with_capacity(self.axes.len());

        for (axis_index, (bound, &dim)) in range.iter().zip(self.shape.iter()).enumerate() {
            let current = &self.axes[axis_index];

            match bound {
                AxisRange::At(i) => {
                    new_base_offset = new_base_offset
                        .checked_add(slice_bound_at(current, dim, *i, axis_index)?)
                        .ok_or_else(mapping_overflow)?;
                }
                AxisRange::In(start, stop, step) => {
                    let (offset_delta, axis, extent) =
                        slice_bound_in(current, dim, *start, *stop, *step, axis_index)?;
                    new_base_offset = new_base_offset
                        .checked_add(offset_delta)
                        .ok_or_else(mapping_overflow)?;
                    new_axes.push(axis);
                    new_shape.push(extent);
                }
                AxisRange::Of(indices) => {
                    new_axes.push(slice_bound_of(current, dim, indices, axis_index)?);
                    new_shape.push(indices.len() as u64);
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

        let mut seen: SmallVec<[bool; PORTABLE_INLINE_RANK]> =
            SmallVec::from_elem(false, self.axes.len());
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
        let (offset_delta, new_axis) = flip_axis_contrib(&self.axes[axis], dim)?;

        let mut axes = self.axes;
        axes[axis] = new_axis;

        Ok(Self {
            base_offset: self
                .base_offset
                .checked_add(offset_delta)
                .ok_or_else(mapping_overflow)?,
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

        let mut remove: SmallVec<[bool; PORTABLE_INLINE_RANK]> = SmallVec::from_elem(false, ndim);

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
        let mut axes_out = Vec::with_capacity(ndim - axes.len());
        let mut shape_out = Shape::with_capacity(ndim - axes.len());

        for (i, &removed) in remove.iter().enumerate().take(ndim) {
            if removed {
                base_offset = base_offset
                    .checked_add(slice_bound_at(&self.axes[i], 1, 0, i)?)
                    .ok_or_else(mapping_overflow)?;
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

        let mut insert_before: SmallVec<[bool; PORTABLE_INLINE_RANK]> =
            SmallVec::from_elem(false, old_ndim);

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

        let axes_out: Vec<AxisContrib> = layout
            .iter()
            .enumerate()
            .map(|(pos, slot)| match slot {
                None => AxisContrib::Stride(contiguous[pos] as i128),
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

pub fn default_permutation(ndim: usize, permutation: Option<Axes>) -> Result<Axes> {
    let axes: Axes = permutation.unwrap_or_else(|| (0..ndim).rev().collect());

    if axes.len() != ndim {
        return Err(Error::InvalidLayout(
            "transpose permutation rank must match tensor rank".to_string(),
        ));
    }

    Ok(axes)
}

fn slice_bound_at(current: &AxisContrib, dim: u64, index: u64, axis_index: usize) -> Result<i128> {
    if index >= dim {
        return Err(Error::InvalidLayout(format!(
            "slice bound at axis {axis_index} is out of bounds"
        )));
    }
    match current {
        AxisContrib::Stride(s) => (index as i128).checked_mul(*s).ok_or_else(mapping_overflow),
        AxisContrib::Broadcast(c) => Ok(*c),
        AxisContrib::Gather(g) => g
            .get(
                usize::try_from(index)
                    .map_err(|_| Error::InvalidCoord("gather index exceeds usize".into()))?,
            )
            .copied()
            .ok_or_else(|| {
                Error::InvalidLayout(format!(
                    "slice bound at axis {axis_index} out of bounds for gather"
                ))
            }),
    }
}

fn slice_bound_in(
    current: &AxisContrib,
    dim: u64,
    start: u64,
    stop: u64,
    step: u64,
    axis_index: usize,
) -> Result<(i128, AxisContrib, u64)> {
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
        AxisContrib::Stride(s) => (
            (start as i128)
                .checked_mul(*s)
                .ok_or_else(mapping_overflow)?,
            AxisContrib::Stride(s.checked_mul(step as i128).ok_or_else(mapping_overflow)?),
        ),
        AxisContrib::Broadcast(c) => (0, AxisContrib::Broadcast(*c)),
        AxisContrib::Gather(g) => (
            0,
            AxisContrib::Gather(
                g.slice(
                    usize::try_from(start)
                        .map_err(|_| Error::InvalidLayout("gather start exceeds usize".into()))?,
                    usize::try_from(step)
                        .map_err(|_| Error::InvalidLayout("gather step exceeds usize".into()))?,
                    usize::try_from(extent)
                        .map_err(|_| Error::InvalidLayout("gather extent exceeds usize".into()))?,
                )?,
            ),
        ),
    };

    Ok((offset_delta, axis, extent))
}

fn slice_bound_of(
    current: &AxisContrib,
    dim: u64,
    indices: &[u64],
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
            let offsets: Vec<i128> = indices
                .iter()
                .map(|idx| (*idx as i128).checked_mul(*s).ok_or_else(mapping_overflow))
                .collect::<Result<_>>()?;
            Ok(AxisContrib::Gather(offsets.into()))
        }
        AxisContrib::Gather(g) => {
            let offsets =
                indices
                    .iter()
                    .map(|idx| {
                        g.get(usize::try_from(*idx).map_err(|_| {
                            Error::InvalidCoord("gather index exceeds usize".into())
                        })?)
                        .copied()
                        .ok_or_else(|| {
                            Error::InvalidLayout(format!(
                                "slice bound at axis {axis_index} out of bounds for gather"
                            ))
                        })
                    })
                    .collect::<Result<Vec<i128>>>()?;
            Ok(AxisContrib::Gather(offsets.into()))
        }
    }
}

fn mapping_overflow() -> Error {
    Error::InvalidLayout("signed logical mapping overflow".into())
}

fn flip_axis_contrib(current: &AxisContrib, dim: u64) -> Result<(i128, AxisContrib)> {
    Ok(match current {
        AxisContrib::Stride(s) => (
            (i128::from(dim) - 1)
                .checked_mul(*s)
                .ok_or_else(mapping_overflow)?,
            AxisContrib::Stride(s.checked_neg().ok_or_else(mapping_overflow)?),
        ),
        AxisContrib::Broadcast(c) => (0, AxisContrib::Broadcast(*c)),
        AxisContrib::Gather(g) => (0, AxisContrib::Gather(g.flipped())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_mapping_overflow_is_a_structured_error() {
        let map = CoordinateMap {
            base_offset: i128::MAX,
            axes: vec![AxisContrib::Stride(i128::MAX)],
            shape: smallvec::smallvec![3],
        };
        assert!(matches!(map.flat_offset(&[2]), Err(Error::InvalidCoord(_))));
        assert!(matches!(map.clone().flip(0), Err(Error::InvalidLayout(_))));
        assert!(matches!(
            map.clone()
                .slice(smallvec::smallvec![AxisRange::In(0, 3, 2)]),
            Err(Error::InvalidLayout(_))
        ));
        assert!(matches!(
            map.slice(smallvec::smallvec![AxisRange::Of(vec![2])]),
            Err(Error::InvalidLayout(_))
        ));
    }

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
        let mut out = Coord::with_capacity(2);
        let ptr = out.as_ptr();

        for row in 0..3 {
            for col in 0..4 {
                flipped
                    .resolve_into(&[row, col], &shape, &strides, &mut out)
                    .unwrap();
                assert_eq!(out.as_slice(), &[row, 3 - col]);
                assert_eq!(out.as_ptr(), ptr);
            }
        }

        let mut bad = map;
        bad.base_offset = 12;
        assert!(
            bad.resolve_into(&[0, 0], &shape, &strides, &mut out)
                .is_err()
        );
        bad.base_offset = i128::MAX;
        assert!(
            bad.resolve_into(&[2, 3], &shape, &strides, &mut out)
                .is_err()
        );
    }
}
