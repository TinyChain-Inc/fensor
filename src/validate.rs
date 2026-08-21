use crate::{Error, Result};
use ha_ndarray::{AxisRange, Range, Shape};

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
    Ok(out)
}

fn axis_range_indices(axis_range: &AxisRange, dim: usize, axis: usize) -> Result<Vec<u64>> {
    let out_of_bounds =
        || Error::InvalidLayout(format!("range bound at axis {axis} is out of bounds"));

    match axis_range {
        AxisRange::At(i) => {
            if *i >= dim {
                return Err(out_of_bounds());
            }
            Ok(vec![*i as u64])
        }
        AxisRange::In(start, stop, step) => {
            if *step == 0 || start > stop || *stop > dim {
                return Err(out_of_bounds());
            }
            Ok((*start..*stop).step_by(*step).map(|i| i as u64).collect())
        }
        AxisRange::Of(indices) => {
            if indices.iter().any(|&i| i >= dim) {
                return Err(out_of_bounds());
            }
            Ok(indices.iter().map(|&i| i as u64).collect())
        }
    }
}

/// Row-major (last axis fastest) coordinate enumeration over a `Range`, validated against `shape`.
pub(crate) fn iter_range_coords(shape: &[usize], range: &Range) -> Result<RangeCoords> {
    if range.len() != shape.len() {
        return Err(Error::InvalidLayout(format!(
            "range has {} axes but tensor has {} dimensions",
            range.len(),
            shape.len()
        )));
    }

    let axis_indices = range
        .iter()
        .zip(shape.iter())
        .enumerate()
        .map(|(axis, (axis_range, &dim))| axis_range_indices(axis_range, dim, axis))
        .collect::<Result<Vec<Vec<u64>>>>()?;

    Ok(RangeCoords::new(axis_indices))
}

/// `AxisRange::In(0, dim, 1)` per axis -- the full-tensor range.
pub(crate) fn full_range(shape: &[usize]) -> Range {
    shape.iter().map(|&dim| AxisRange::In(0, dim, 1)).collect()
}

pub(crate) struct RangeCoords {
    axis_indices: Vec<Vec<u64>>,
    coord: Vec<usize>,
    remaining: usize,
}

impl RangeCoords {
    fn new(axis_indices: Vec<Vec<u64>>) -> Self {
        let remaining = axis_indices.iter().map(Vec::len).product();
        let ndim = axis_indices.len();
        Self {
            axis_indices,
            coord: vec![0; ndim],
            remaining,
        }
    }
}

impl Iterator for RangeCoords {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let out: Vec<u64> = self
            .coord
            .iter()
            .zip(&self.axis_indices)
            .map(|(&i, axis)| axis[i])
            .collect();
        self.remaining -= 1;

        if self.remaining > 0 {
            let mut axis = self.coord.len() - 1;
            loop {
                self.coord[axis] += 1;
                if self.coord[axis] < self.axis_indices[axis].len() {
                    break;
                }
                self.coord[axis] = 0;
                axis -= 1;
            }
        }

        Some(out)
    }
}
