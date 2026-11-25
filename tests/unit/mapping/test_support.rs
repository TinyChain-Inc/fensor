use crate::Shape;
use super::{AxisContrib, CoordinateMap};

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
