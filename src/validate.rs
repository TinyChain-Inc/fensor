use crate::{Error, Result};
use ha_ndarray::Shape;

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
