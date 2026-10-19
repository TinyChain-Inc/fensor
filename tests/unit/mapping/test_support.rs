use super::{AxisContrib, CoordinateMap};
use crate::Shape;

pub(crate) fn affine(offset: i128, stride: i128, shape: Shape) -> CoordinateMap {
    CoordinateMap {
        base_offset: offset,
        axes: vec![AxisContrib::Stride(stride)],
        shape,
    }
}

pub(crate) fn set_offset(mapping: &mut CoordinateMap, offset: i128) {
    mapping.base_offset = offset;
}
