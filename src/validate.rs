use ha_ndarray::{AxisRange, Range, Shape};

use crate::{Error, Result};

pub(crate) fn validate_coord(shape: &[usize], coord: &[u64]) -> Result<()> {
    if coord.len() != shape.len() {
        return Err(Error::InvalidCoord(
            "incorrect number of coordinates".to_string(),
        ));
    }

    for (i, (c, dim)) in coord.iter().zip(shape.iter()).enumerate() {
        let c = usize::try_from(*c)
            .map_err(|_| Error::InvalidCoord(format!("coordinate at axis {i} overflows usize")))?;
        if c >= *dim {
            return Err(Error::InvalidCoord(format!(
                "coordinate at axis {i} is out of bounds"
            )));
        }
    }

    Ok(())
}

pub(crate) fn ensure_offset_in_bounds(offset: usize, block_len: usize) -> Result<()> {
    if offset < block_len {
        Ok(())
    } else {
        Err(Error::InvalidLayout(
            "block offset out of bounds".to_string(),
        ))
    }
}

pub fn matmul_output_shape(left: &[usize], right: &[usize]) -> Result<Shape> {
    if left.len() < 2 || right.len() < 2 {
        return Err(Error::InvalidLayout(format!(
            "invalid dimensions for matrix multiply: left={left:?}, right={right:?} (expected rank >= 2)"
        )));
    }

    for shape in [left, right] {
        crate::schema::validate_shape_dims(shape)?;
        shape
            .iter()
            .try_fold(1usize, |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| Error::InvalidLayout("matrix input size overflow".into()))?;
    }

    let l_inner = left[left.len() - 1];
    let r_inner = right[right.len() - 2];
    if l_inner != r_inner {
        return Err(Error::InvalidLayout(format!(
            "invalid dimensions for matrix multiply: left={left:?}, right={right:?} (inner dimensions {l_inner} and {r_inner} do not match)"
        )));
    }

    let l_batch = &left[..left.len() - 2];
    let r_batch = &right[..right.len() - 2];
    if l_batch != r_batch {
        return Err(Error::InvalidLayout(format!(
            "invalid batch dimensions for matrix multiply: left={left:?}, right={right:?}"
        )));
    }

    let mut out = Shape::with_capacity(l_batch.len() + 2);
    out.extend(l_batch.iter().copied());
    out.push(left[left.len() - 2]);
    out.push(right[right.len() - 1]);
    out.iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
        .ok_or_else(|| Error::InvalidLayout("matrix output size overflow".into()))?;

    Ok(out)
}

/// Validate ranges without expanding interval axes into coordinate buffers.
pub(crate) fn iter_range_coords(shape: &[usize], range: &Range) -> Result<RangeCoords> {
    if range.len() != shape.len() {
        return Err(Error::InvalidLayout(
            "range rank must match tensor rank".into(),
        ));
    }

    let mut lengths = Vec::with_capacity(range.len());

    for (axis, (bound, &dim)) in range.iter().zip(shape).enumerate() {
        let invalid =
            || Error::InvalidLayout(format!("range bound at axis {axis} is out of bounds"));
        let len = match bound {
            AxisRange::At(i) if *i < dim => 1,
            AxisRange::In(start, stop, step) if *step > 0 && start <= stop && *stop <= dim => {
                (stop - start).div_ceil(*step)
            }
            AxisRange::Of(indices) if indices.iter().all(|&i| i < dim) => indices.len(),
            _ => return Err(invalid()),
        };
        lengths.push(len);
    }

    let remaining = lengths
        .iter()
        .try_fold(1usize, |n, &len| n.checked_mul(len))
        .ok_or_else(|| Error::InvalidLayout("range size overflow".into()))?;
    Ok(RangeCoords {
        range: range.clone(),
        coord: vec![0; range.len()],
        lengths,
        remaining,
    })
}

pub(crate) fn full_range(shape: &[usize]) -> Range {
    shape.iter().map(|&dim| AxisRange::In(0, dim, 1)).collect()
}

// Rank-sized traversal state plus caller-sized explicit selections, if supplied.
// Interval axes are descriptors; no range-sized coordinate table is constructed.
pub(crate) struct RangeCoords {
    range: Range,
    lengths: Vec<usize>,
    coord: Vec<usize>,
    remaining: usize,
}

// Checked cardinality allows write-length validation without visiting coordinates.
impl ExactSizeIterator for RangeCoords {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl Iterator for RangeCoords {
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let out = self
            .coord
            .iter()
            .zip(&self.range)
            .map(|(&i, bound)| match bound {
                AxisRange::At(at) => *at as u64,
                AxisRange::In(start, _, step) => (start + i * step) as u64,
                AxisRange::Of(indices) => indices[i] as u64,
            })
            .collect();
        self.remaining -= 1;
        if self.remaining > 0 {
            for axis in (0..self.coord.len()).rev() {
                self.coord[axis] += 1;
                if self.coord[axis] < self.lengths[axis] {
                    break;
                }

                self.coord[axis] = 0;
            }
        }

        Some(out)
    }
}
