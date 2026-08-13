use ha_ndarray::{Shape, Strides};

use crate::schema::{contiguous_strides, greedy_block_shape, validate_shape_dims};
use crate::{DType, Error, Layout, Result as FResult};

pub type TensorShape = Shape;
pub type TensorStrides = Strides;

pub(crate) type BlockShape = Shape;
pub(crate) type BlockStrides = Strides;

pub(crate) type StorageShape = Shape;
pub(crate) type StorageStrides = Strides;

pub const MAX_BLOCK_CAPACITY: usize = 4096;

/// Base tensor identity: dtype + fixed logical shape + fixed contiguous
/// strides. Held by `Tensor`, never mutated after creation -- the *current*
/// (possibly-transformed) shape/strides live on `TensorView` instead.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorSchema {
    dtype: DType,
    shape: TensorShape,
    strides: TensorStrides,
}

impl TensorSchema {
    pub fn new(dtype: DType, shape: Shape) -> FResult<Self> {
        validate_shape_dims(shape.as_slice())?;
        let strides = contiguous_strides(shape.as_slice())?;
        Ok(Self {
            dtype,
            shape,
            strides,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct BlockSchema {
    pub(crate) shape: BlockShape,
    pub(crate) strides: BlockStrides,
}

impl BlockSchema {
    pub(crate) fn new(tensor_shape: &[usize], max_capacity: usize) -> FResult<Self> {
        let shape = greedy_block_shape(tensor_shape, max_capacity)?;
        let strides = contiguous_strides(shape.as_slice())?;
        Ok(Self { shape, strides })
    }
}

/// Held by `DenseStorage`/`SparseStorage` INSTEAD OF `TensorSchema`.
/// `shape`/`strides` are the block-grid dimensions/strides (ceil(tensor_dim /
/// block_dim) per axis), precomputed once at construction -- never recomputed
/// per call.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct StorageSchema {
    pub(crate) shape: StorageShape,
    pub(crate) layout: Layout,
    pub(crate) strides: StorageStrides,
    pub(crate) block_schema: BlockSchema,
}

impl StorageSchema {
    /// Creation path: run the greedy algorithm to pick a block shape.
    pub(crate) fn new(
        tensor_shape: &[usize],
        layout: Layout,
        max_capacity: usize,
    ) -> FResult<Self> {
        let block_schema = BlockSchema::new(tensor_shape, max_capacity)?;
        Self::from_block_schema(tensor_shape, layout, block_schema)
    }

    /// Load path: block shape already known (persisted) -- no greedy run needed.
    pub(crate) fn from_block_shape(
        tensor_shape: &[usize],
        layout: Layout,
        block_shape: Shape,
    ) -> FResult<Self> {
        let strides = contiguous_strides(block_shape.as_slice())?;
        Self::from_block_schema(
            tensor_shape,
            layout,
            BlockSchema {
                shape: block_shape,
                strides,
            },
        )
    }

    fn from_block_schema(
        tensor_shape: &[usize],
        layout: Layout,
        block_schema: BlockSchema,
    ) -> FResult<Self> {
        if let Layout::Sparse { axis: Some(axis) } = layout
            && axis >= tensor_shape.len()
        {
            return Err(Error::InvalidSchema(
                "sparse axis hint out of bounds".to_string(),
            ));
        }

        let shape: StorageShape = tensor_shape
            .iter()
            .zip(block_schema.shape.iter())
            .map(|(dim, block_dim)| dim.div_ceil(*block_dim))
            .collect();
        let strides = contiguous_strides(shape.as_slice())?;
        Ok(Self {
            shape,
            layout,
            strides,
            block_schema,
        })
    }
}

/// Physical location of a coordinate: which block, and the offset within it.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) struct BlockPosition {
    pub(crate) block_id: u64,
    pub(crate) offset_in_block: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_schema_new_computes_strides() {
        let schema = TensorSchema::new(DType::F32, vec![4usize, 5, 6].into()).expect("schema");
        assert_eq!(schema.shape().as_slice(), &[4, 5, 6]);
        assert_eq!(schema.strides().as_slice(), &[30, 6, 1]);
    }

    #[test]
    fn tensor_schema_new_rejects_empty_shape() {
        let err = TensorSchema::new(DType::F32, Shape::new()).expect_err("empty shape rejected");
        assert!(matches!(err, crate::Error::InvalidSchema(_)));
    }

    #[test]
    fn block_schema_new_covers_whole_tensor() {
        let block = BlockSchema::new(&[4, 5, 6], 1000).expect("block schema");
        assert_eq!(block.shape.as_slice(), &[4, 5, 6]);
        assert_eq!(block.strides.as_slice(), &[30, 6, 1]);
    }

    #[test]
    fn storage_schema_new_evenly_dividing() {
        // tensor [4, 6], capacity large enough for block [4, 6] -> grid [1, 1]
        let storage = StorageSchema::new(&[4, 6], Layout::Dense, 1000).expect("storage schema");
        assert_eq!(storage.shape.as_slice(), &[1, 1]);
        assert_eq!(storage.block_schema.shape.as_slice(), &[4, 6]);
    }

    #[test]
    fn storage_schema_new_ceiling_remainder() {
        // tensor [10], capacity 3 -> block [3], grid ceil(10/3) = 4
        let storage = StorageSchema::new(&[10], Layout::Dense, 3).expect("storage schema");
        assert_eq!(storage.block_schema.shape.as_slice(), &[3]);
        assert_eq!(storage.shape.as_slice(), &[4]);
        assert_eq!(storage.strides.as_slice(), &[1]);
    }

    #[test]
    fn storage_schema_from_block_shape_matches_new() {
        let via_new = StorageSchema::new(&[10], Layout::Dense, 3).expect("via new");
        let via_block_shape =
            StorageSchema::from_block_shape(&[10], Layout::Dense, vec![3usize].into())
                .expect("via from_block_shape");
        assert_eq!(via_new, via_block_shape);
    }

    #[test]
    fn storage_schema_new_rejects_zero_capacity() {
        let err =
            StorageSchema::new(&[4, 5, 6], Layout::Dense, 0).expect_err("zero capacity rejected");
        assert!(matches!(err, crate::Error::InvalidSchema(_)));
    }

    #[test]
    fn storage_schema_new_rejects_empty_shape() {
        let err = StorageSchema::new(&[], Layout::Dense, 10).expect_err("empty shape rejected");
        assert!(matches!(err, crate::Error::InvalidSchema(_)));
    }
}
