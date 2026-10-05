use crate::schema::Coord;
use crate::{Axes, AxisRange, Error, Range, Result, Shape};
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

/// Common broadcast shape, aligning trailing axes without narrowing logical dimensions.
/// This describes geometry only; operands still need explicit broadcast transforms.
pub fn broadcast_shape(left: &[u64], right: &[u64]) -> Result<Shape> {
    let mut shape = Shape::from_elem(1, left.len().max(right.len()));
    let rank = shape.len();
    for i in 0..rank {
        let l = left.len().checked_sub(i + 1).map(|a| left[a]).unwrap_or(1);
        let r = right
            .len()
            .checked_sub(i + 1)
            .map(|a| right[a])
            .unwrap_or(1);
        if l != r && l != 1 && r != 1 {
            return Err(Error::InvalidSchema(
                "incompatible tensor broadcast shapes".into(),
            ));
        }
        shape[rank - i - 1] = l.max(r);
    }
    Ok(shape)
}

/// Operand shapes with common matrix batch axes, preserving their final two axes.
/// Contraction dimensions are validated by matrix multiplication itself.
pub fn matmul_broadcast_shapes(left: &[u64], right: &[u64]) -> Result<(Shape, Shape)> {
    if left.len() < 2 || right.len() < 2 {
        return Err(Error::InvalidSchema(
            "matmul requires rank at least two".into(),
        ));
    }
    let mut left_shape = broadcast_shape(&left[..left.len() - 2], &right[..right.len() - 2])?;
    let mut right_shape = left_shape.clone();
    left_shape.extend_from_slice(&left[left.len() - 2..]);
    right_shape.extend_from_slice(&right[right.len() - 2..]);
    Ok((left_shape, right_shape))
}

/// Sorted, unique reduction axes, validated against the input rank.
pub fn reduction_axes(rank: usize, mut axes: Axes) -> Result<Axes> {
    axes.sort_unstable();
    axes.dedup();
    if axes.iter().any(|&axis| axis >= rank) {
        return Err(Error::InvalidLayout("reduction axis out of bounds".into()));
    }
    Ok(axes)
}

/// Axes to sum with retained dimensions before reshaping to a broadcast source shape.
pub fn broadcast_reduce_axes(source: &[u64], target: &[u64]) -> Result<Axes> {
    if target.len() > source.len() {
        return Err(Error::InvalidSchema(
            "broadcast reduction increases rank".into(),
        ));
    }
    let prefix = source.len() - target.len();
    let mut axes: Axes = (0..prefix).collect();
    for (axis, (&source, &target)) in source[prefix..].iter().zip(target).enumerate() {
        if target == 1 && source != 1 {
            axes.push(prefix + axis);
        } else if source != target {
            return Err(Error::InvalidSchema(
                "invalid broadcast reduction target".into(),
            ));
        }
    }
    Ok(axes)
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
    fn broadcast_geometry_preserves_logical_width_and_explicit_matrix_contract() {
        for (left, right, expected) in [
            (vec![], vec![2, 3], vec![2, 3]),
            (vec![2, 1, 4], vec![3, 4], vec![2, 3, 4]),
            (vec![u64::MAX, 1], vec![1, 2], vec![u64::MAX, 2]),
        ] {
            assert_eq!(broadcast_shape(&left, &right).unwrap().as_slice(), expected);
            assert_eq!(broadcast_shape(&right, &left).unwrap().as_slice(), expected);
        }
        assert!(broadcast_shape(&[2, 3], &[4, 3]).is_err());
        let (left, right) = matmul_broadcast_shapes(&[2, 1, 3, 4], &[5, 4, 6]).unwrap();
        assert_eq!(left.as_slice(), &[2, 5, 3, 4]);
        assert_eq!(right.as_slice(), &[2, 5, 4, 6]);
        assert_eq!(
            matmul_output_shape(&left, &right).unwrap().as_slice(),
            &[2, 5, 3, 6]
        );
        assert!(matmul_output_shape(&[2, 1, 3, 4], &[5, 4, 6]).is_err());
        assert!(matmul_broadcast_shapes(&[3], &[3, 4]).is_err());
        assert!(matmul_broadcast_shapes(&[2, 3, 4], &[5, 4, 6]).is_err());
    }

    #[test]
    fn reduction_geometry_normalizes_axes_and_reverses_broadcasting() {
        for (rank, axes, expected) in [
            (0, vec![], vec![]),
            (3, vec![2, 0, 2], vec![0, 2]),
            (3, vec![], vec![]),
        ] {
            assert_eq!(
                reduction_axes(rank, axes.into()).unwrap().as_slice(),
                expected
            );
        }
        assert!(reduction_axes(3, vec![3, 0, 3].into()).is_err());
        for (source, target, expected) in [
            (vec![2, 3, 4], vec![1, 4], vec![0, 1]),
            (vec![2, 3], vec![2, 3], vec![]),
            (vec![2, 3], vec![], vec![0, 1]),
            (vec![u64::MAX, 3], vec![1, 3], vec![0]),
        ] {
            assert_eq!(
                broadcast_reduce_axes(&source, &target).unwrap().as_slice(),
                expected
            );
        }
        assert!(broadcast_reduce_axes(&[2, 3], &[2, 2]).is_err());
        assert!(broadcast_reduce_axes(&[2, 3], &[1, 2, 3]).is_err());
    }

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
