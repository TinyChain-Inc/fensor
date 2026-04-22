use std::future::Future;
use std::pin::Pin;

use ha_ndarray::{Axes, Range, Shape};

use crate::schema::{Layout, TensorSchema};
use crate::{Error, Result};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A minimal ndarray-like semantic surface for filesystem-backed tensors.
pub trait Tensor: Send + Sync {
    type DType: Copy + Send + Sync + 'static;

    fn schema(&self) -> &TensorSchema;

    fn dtype(&self) -> Self::DType;

    fn shape(&self) -> &[usize] {
        &self.schema().shape
    }

    fn layout(&self) -> &Layout {
        &self.schema().layout
    }

    fn strides(&self) -> &[usize] {
        &self.schema().strides
    }

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Async value reads aligned with ndarray coordinate semantics.
pub trait TensorRead: Tensor {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>>;

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        _range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<Vec<(Vec<u64>, Self::DType)>>> {
        let base_order = (0..self.ndim()).collect::<Vec<_>>();
        let requested_order = requested_order.into_iter().collect::<Vec<_>>();

        Box::pin(async move {
            Err(Error::UnsupportedSparseIterationOrder {
                requested_order,
                base_order,
                hint: "materialize or perform external sort for incompatible order".to_string(),
            })
        })
    }
}

/// Async value writes aligned with ndarray coordinate semantics.
pub trait TensorWrite: Tensor {
    fn write_value<'a>(&'a self, coord: &'a [u64], value: Self::DType)
    -> BoxFuture<'a, Result<()>>;
}

/// Transform-style ndarray operations (metadata/view level).
pub trait TensorTransform: Tensor + Sized {
    fn reshape(self, shape: Shape) -> Result<Self>;

    fn slice(self, range: Range) -> Result<Self>;

    fn transpose(self, permutation: Option<Axes>) -> Result<Self>;
}

/// Async block-level storage primitives used by higher-level tensor accessors.
pub trait TensorBlockStore: Send + Sync {
    type Block: Clone + Send + Sync + 'static;

    fn read_block<'a>(&'a self, block_id: u64) -> BoxFuture<'a, Result<Option<Self::Block>>>;

    fn write_block<'a>(&'a self, block_id: u64, block: Self::Block) -> BoxFuture<'a, Result<()>>;
}

/// Sparse index access primitives for layouts backed by `b-table`.
pub trait TensorSparseIndex: Send + Sync {
    fn lookup_block_id<'a>(&'a self, key: &'a [u64]) -> BoxFuture<'a, Result<Option<u64>>>;

    fn upsert_block_id<'a>(&'a self, key: Vec<u64>, block_id: u64) -> BoxFuture<'a, Result<()>>;
}
