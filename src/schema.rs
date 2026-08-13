use std::io;

use b_table::{IndexSchema, Schema};
use ha_ndarray::{Shape, Strides};
use smallvec::SmallVec;

use crate::{Error, Result as FResult};

pub const PORTABLE_INLINE_RANK: usize = 8;
pub type TensorShape = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub fn try_parse(value: &str) -> Option<Self> {
        match value {
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Layout {
    Dense,
    Sparse { axis: Option<usize> },
}

pub(crate) fn validate_shape_dims<T>(shape: &[T]) -> FResult<()>
where
    T: Copy + PartialEq + From<u8>,
{
    if shape.is_empty() {
        return Err(Error::InvalidSchema(
            "tensor shape cannot be empty".to_string(),
        ));
    }

    if shape.iter().any(|dim| *dim == T::from(0u8)) {
        return Err(Error::InvalidSchema(
            "tensor shape dimensions must be non-zero".to_string(),
        ));
    }

    Ok(())
}

pub fn contiguous_strides(shape: &[usize]) -> FResult<Strides> {
    validate_shape_dims(shape)?;

    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];

    for i in (0..ndim).rev() {
        if i + 1 < ndim {
            strides[i] = strides[i + 1]
                .checked_mul(shape[i + 1])
                .ok_or_else(|| Error::InvalidSchema("Dimension overflows stride".to_string()))?;
        }
    }

    Ok(strides.into())
}

pub fn greedy_block_shape(shape: &[usize], max_capacity: usize) -> FResult<Shape> {
    validate_shape_dims(shape)?;

    if max_capacity == 0 {
        return Err(Error::InvalidSchema(
            "max block capacity must be non-zero".to_string(),
        ));
    }

    let ndim = shape.len();
    let mut block_shape = vec![1usize; ndim];
    let mut remaining = max_capacity;

    for i in (0..ndim).rev() {
        let take = shape[i].min(remaining);
        block_shape[i] = take;
        remaining /= take;
    }

    Ok(block_shape.into())
}

fn checked_product(shape: &[usize]) -> FResult<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim as u64))
        .ok_or_else(|| Error::InvalidSchema("shape element count overflows usize".to_string()))
}

/// Row-major (C-order) coordinate walk over `shape`, used to materialize or
/// reconstruct a tensor view's data as a flat sequence for wire transfer.
pub(crate) struct RowMajorCoords {
    shape: Vec<usize>,
    coord: Vec<u64>,
    remaining: u64,
}

pub(crate) fn row_major_coords(shape: &[usize]) -> FResult<RowMajorCoords> {
    let remaining = if shape.is_empty() {
        0
    } else {
        checked_product(shape)?
    };

    Ok(RowMajorCoords {
        shape: shape.to_vec(),
        coord: vec![0u64; shape.len()],
        remaining,
    })
}

impl Iterator for RowMajorCoords {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let out = self.coord.clone();
        self.remaining -= 1;

        if self.remaining > 0 {
            let mut axis = self.shape.len() - 1;
            loop {
                self.coord[axis] += 1;
                if (self.coord[axis] as usize) < self.shape[axis] {
                    break;
                }
                self.coord[axis] = 0;
                if axis == 0 {
                    break;
                }
                axis -= 1;
            }
        }

        Some(out)
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SparseIndexSchema {
    columns: Vec<String>,
}

impl SparseIndexSchema {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }
}

impl b_table::BTreeSchema for SparseIndexSchema {
    type Error = io::Error;
    type Value = u64;

    fn block_size(&self) -> usize {
        4096
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    fn order(&self) -> usize {
        16
    }

    fn validate_key(
        &self,
        key: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse index key length",
            ))
        }
    }
}

impl IndexSchema for SparseIndexSchema {
    type Id = String;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SparseTableSchema {
    primary: SparseIndexSchema,
    auxiliary: Vec<(String, SparseIndexSchema)>,
}

impl Default for SparseTableSchema {
    fn default() -> Self {
        Self {
            primary: SparseIndexSchema::new(vec![
                "coord".to_string(),
                "block_offset".to_string(),
                "block_id".to_string(),
            ]),
            auxiliary: vec![],
        }
    }
}

impl Schema for SparseTableSchema {
    type Id = String;
    type Error = io::Error;
    type Value = u64;
    type Index = SparseIndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns()[0..2]
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns()[2..]
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, Self::Index)] {
        &self.auxiliary
    }

    fn validate_key(
        &self,
        key: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if key.len() == 2 {
            Ok(key)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse table key length",
            ))
        }
    }

    fn validate_values(
        &self,
        values: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if values.len() == 1 {
            Ok(values)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse table value length",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_rank_three() {
        // [4,5,6]: strides[2]=1, strides[1]=1*6=6, strides[0]=6*5=30
        assert_eq!(
            contiguous_strides(&[4, 5, 6]).expect("strides"),
            Strides::from_vec(vec![30, 6, 1])
        );
    }

    #[test]
    fn contiguous_strides_rank_two() {
        // [2,3]: strides[1]=1, strides[0]=1*3=3
        assert_eq!(
            contiguous_strides(&[2, 3]).expect("strides"),
            Strides::from_vec(vec![3, 1])
        );
    }

    #[test]
    fn contiguous_strides_single_dim() {
        assert_eq!(
            contiguous_strides(&[7]).expect("strides"),
            Strides::from_vec(vec![1])
        );
    }

    #[test]
    fn contiguous_strides_all_ones() {
        // [1,1,1]: every stride collapses to 1
        assert_eq!(
            contiguous_strides(&[1, 1, 1]).expect("strides"),
            Strides::from_vec(vec![1, 1, 1])
        );
    }

    #[test]
    fn contiguous_strides_leading_one() {
        // [1,5,6]: a leading 1 dim doesn't perturb trailing strides
        assert_eq!(
            contiguous_strides(&[1, 5, 6]).expect("strides"),
            Strides::from_vec(vec![30, 6, 1])
        );
    }

    #[test]
    fn contiguous_strides_trailing_one() {
        // [4,5,1]: a trailing 1 dim collapses the last two strides to 1
        assert_eq!(
            contiguous_strides(&[4, 5, 1]).expect("strides"),
            Strides::from_vec(vec![5, 1, 1])
        );
    }

    #[test]
    fn contiguous_strides_empty_shape() {
        // validate_shape_dims rejects an empty shape before any stride math runs
        let err = contiguous_strides(&[]).expect_err("empty shape should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn contiguous_strides_zero_dim() {
        // validate_shape_dims rejects any zero-sized dimension, so the old
        // "0 propagates as a stride" behavior is no longer reachable
        let err = contiguous_strides(&[3, 0, 4]).expect_err("zero dimension should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn contiguous_strides_overflow() {
        // strides[1] = shape[2] = usize::MAX (no overflow: 1 * MAX);
        // strides[0] = strides[1] * shape[1] = MAX * 2 -> overflows usize
        let err = contiguous_strides(&[3, 2, usize::MAX]).expect_err("overflow should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn greedy_block_shape_capacity_covers_whole_tensor() {
        // cap=1000 exceeds 4*5*6=120, so the whole tensor fits in one block
        assert_eq!(
            greedy_block_shape(&[4, 5, 6], 1000).expect("block shape"),
            Shape::from_vec(vec![4, 5, 6])
        );
    }

    #[test]
    fn greedy_block_shape_capacity_equals_total() {
        // cap=6 exactly equals 2*3, so the whole tensor still fits
        assert_eq!(
            greedy_block_shape(&[2, 3], 6).expect("block shape"),
            Shape::from_vec(vec![2, 3])
        );
    }

    #[test]
    fn greedy_block_shape_partial_fit_limits_outer_axis() {
        // axis 2: take=min(6,50)=6, remaining=50/6=8
        // axis 1: take=min(5,8)=5, remaining=8/5=1
        // axis 0: take=min(4,1)=1
        assert_eq!(
            greedy_block_shape(&[4, 5, 6], 50).expect("block shape"),
            Shape::from_vec(vec![1, 5, 6])
        );
    }

    #[test]
    fn greedy_block_shape_rank_two_partial() {
        // axis 1: take=min(7,5)=5, remaining=5/5=1
        // axis 0: take=min(3,1)=1
        assert_eq!(
            greedy_block_shape(&[3, 7], 5).expect("block shape"),
            Shape::from_vec(vec![1, 5])
        );
    }

    #[test]
    fn greedy_block_shape_single_dim() {
        // cap needn't evenly divide the axis: take=min(10,3)=3
        assert_eq!(
            greedy_block_shape(&[10], 3).expect("block shape"),
            Shape::from_vec(vec![3])
        );
    }

    #[test]
    fn greedy_block_shape_capacity_one() {
        // the smallest feasible capacity forces every axis down to 1
        assert_eq!(
            greedy_block_shape(&[4, 5, 6], 1).expect("block shape"),
            Shape::from_vec(vec![1, 1, 1])
        );
    }

    #[test]
    fn greedy_block_shape_zero_capacity() {
        // no block can hold 0 elements; block_shape entries must be non-zero
        let err = greedy_block_shape(&[4, 5, 6], 0).expect_err("zero capacity should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn greedy_block_shape_empty_shape() {
        // validate_shape_dims rejects an empty shape, same as contiguous_strides
        let err = greedy_block_shape(&[], 10).expect_err("empty shape should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn greedy_block_shape_zero_dim() {
        // validate_shape_dims rejects any zero-sized dimension, same as contiguous_strides
        let err =
            greedy_block_shape(&[3, 0, 4], 10).expect_err("zero dimension should be rejected");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }
}
