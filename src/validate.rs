use crate::schema::Coord;
use crate::{AxisRange, Error, Range, Result, Shape};
pub(crate) fn validate_coord(shape: &[u64], coord: &[u64]) -> Result<()> {
    if coord.len() != shape.len() {
        return Err(Error::InvalidCoord(
            "incorrect number of coordinates".to_string(),
        ));
    }

    for (i, (c, dim)) in coord.iter().zip(shape.iter()).enumerate() {
        if c >= dim {
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

pub fn matmul_output_shape(left: &[u64], right: &[u64]) -> Result<Shape> {
    if left.len() < 2 || right.len() < 2 {
        return Err(Error::InvalidLayout(format!(
            "invalid dimensions for matrix multiply: left={left:?}, right={right:?} (expected rank >= 2)"
        )));
    }

    for shape in [left, right] {
        crate::schema::validate_shape_dims(shape)?;
        crate::schema::checked_product(shape)
            .map_err(|_| Error::InvalidLayout("matrix input size overflow".into()))?;
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
    crate::schema::checked_product(&out)
        .map_err(|_| Error::InvalidLayout("matrix output size overflow".into()))?;

    Ok(out)
}

/// Validate ranges without expanding interval axes into coordinate buffers.
pub(crate) fn iter_range_coords(shape: &[u64], range: &Range) -> Result<RangeCoords> {
    if range.len() != shape.len() {
        return Err(Error::InvalidLayout(
            "range rank must match tensor rank".into(),
        ));
    }

    let mut lengths = crate::Shape::with_capacity(range.len());

    for (axis, (bound, &dim)) in range.iter().zip(shape).enumerate() {
        let invalid =
            || Error::InvalidLayout(format!("range bound at axis {axis} is out of bounds"));
        let len = match bound {
            AxisRange::At(i) if *i < dim => 1,
            AxisRange::In(start, stop, step) if *step > 0 && start <= stop && *stop <= dim => {
                (stop - start).div_ceil(*step)
            }
            AxisRange::Of(indices) if indices.iter().all(|&i| i < dim) => indices.len() as u64,
            _ => return Err(invalid()),
        };
        lengths.push(len);
    }

    let remaining = if lengths.contains(&0) {
        0
    } else {
        crate::schema::checked_product(&lengths)
            .map_err(|_| Error::InvalidLayout("range size overflow".into()))?
    };
    Ok(RangeCoords {
        range: range.clone(),
        coord: Coord::from_elem(0, range.len()),
        lengths,
        remaining,
    })
}

pub(crate) fn full_range(shape: &[u64]) -> Range {
    shape.iter().map(|&dim| AxisRange::In(0, dim, 1)).collect()
}

// Rank-sized traversal state plus caller-sized explicit selections, if supplied.
// Interval axes are descriptors; no range-sized coordinate table is constructed.
pub(crate) struct RangeCoords {
    range: Range,
    lengths: crate::Shape,
    coord: Coord,
    remaining: u64,
}

// Checked cardinality allows write-length validation without visiting coordinates.
impl RangeCoords {
    pub(crate) fn remaining(&self) -> u64 {
        self.remaining
    }
}

impl Iterator for RangeCoords {
    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(n) => (n, Some(n)),
            Err(_) => (usize::MAX, None),
        }
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
                AxisRange::At(at) => *at,
                AxisRange::In(start, _, step) => start + i * step,
                AxisRange::Of(indices) => indices[i as usize],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_cardinality_handles_empty_selections_and_overflow() {
        let long = u64::MAX / 2;
        let selection: Range =
            smallvec::smallvec![AxisRange::In(0, long, 1), AxisRange::Of(vec![0, 1, 0])];
        assert!(matches!(
            iter_range_coords(&[long, 2], &selection),
            Err(Error::InvalidLayout(_))
        ));

        let mut empty = selection;
        empty.push(AxisRange::Of(Vec::new()));
        let mut coords = iter_range_coords(&[long, 2, 1], &empty).unwrap();
        assert_eq!(coords.remaining(), 0);
        assert_eq!(coords.next(), None);
        assert_eq!(coords.next(), None);
    }
}
