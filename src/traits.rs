use std::future::Future;
use std::pin::Pin;

use ha_ndarray::{Axes, Range, Shape};

use crate::schema::{DType, Layout};
use crate::validate;
use crate::{Error, Result, TensorSchema};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Minimal shape/dtype surface shared by base tensors AND their views.
pub trait TensorGeometry: Send + Sync {
    type DType: Copy + Send + Sync + 'static;

    fn dtype(&self) -> Self::DType;

    fn layout(&self) -> Layout;

    fn shape(&self) -> &[usize];

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Adds the persistent, storage-backed schema -- implemented ONLY by base tensors.
pub trait TensorArray: TensorGeometry {
    fn schema(&self) -> &TensorSchema;

    fn strides(&self) -> &[usize];

    fn schema_dtype(&self) -> DType {
        self.schema().dtype()
    }
}

type OrderedSparseElements<ET> = Vec<(Vec<u64>, ET)>;

/// Async value reads aligned with ndarray coordinate semantics.
pub trait TensorRead: TensorGeometry {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>>;

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        _range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<OrderedSparseElements<Self::DType>>> {
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

/// Bulk/contiguous read semantics for tensor backends.
pub trait TensorReadBulk: TensorRead {
    fn read_values<'a>(&'a self, _range: Range) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "bulk read is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn read_all<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "read_all is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Async value writes aligned with ndarray coordinate semantics.
pub trait TensorWrite: TensorGeometry {
    fn write_value<'a>(&'a self, coord: &'a [u64], value: Self::DType)
    -> BoxFuture<'a, Result<()>>;
}

/// Bulk/contiguous write semantics for tensor backends.
pub trait TensorWriteBulk: TensorWrite {
    fn write_values<'a>(
        &'a self,
        _range: Range,
        _values: Vec<Self::DType>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "bulk write is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn write_tensor<'a, T>(&'a self, _other: &'a T) -> BoxFuture<'a, Result<()>>
    where
        T: TensorRead<DType = Self::DType> + Sync + ?Sized,
    {
        Box::pin(async move {
            Err(Error::Unsupported(
                "write_tensor is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn fill<'a>(&'a self, _value: Self::DType) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "fill is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Transform-style ndarray operations (metadata/view level).
pub trait TensorTransform: TensorGeometry + Sized {
    fn reshape(self, shape: Shape) -> Result<Self>;

    fn broadcast(self, _shape: Shape) -> Result<Self> {
        Err(Error::Unsupported(
            "broadcast is not implemented for this tensor backend".to_string(),
        ))
    }

    fn flip(self, _axis: usize) -> Result<Self> {
        Err(Error::Unsupported(
            "flip is not implemented for this tensor backend".to_string(),
        ))
    }

    fn slice(self, range: Range) -> Result<Self>;

    fn squeeze(self, _axes: Axes) -> Result<Self> {
        Err(Error::Unsupported(
            "squeeze is not implemented for this tensor backend".to_string(),
        ))
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self>;

    fn unsqueeze(self, _axes: Axes) -> Result<Self> {
        Err(Error::Unsupported(
            "unsqueeze is not implemented for this tensor backend".to_string(),
        ))
    }
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

    fn delete_row<'a>(&'a self, _key: Vec<u64>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "delete_row is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Base/view capability contract, aligned with v1 writeability semantics.
pub trait TensorViewSemantics: TensorGeometry {
    fn is_base_tensor(&self) -> bool {
        true
    }

    fn supports_write_through(&self) -> bool {
        false
    }
}

/// Unary tensor math operations.
pub trait TensorUnary: TensorArray + Sized {
    fn exp<'a>(&'a self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "exp is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn ln<'a>(&'a self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "ln is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn round<'a>(&'a self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "round is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Elementwise tensor math operations.
pub trait TensorMath: TensorArray + Sized {
    fn add<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "add is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn div<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "div is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn log<'a>(&'a self, _base: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "log is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn mul<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "mul is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn pow<'a>(&'a self, _exp: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "pow is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn sub<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "sub is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn rem<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "rem is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Elementwise tensor math operations with scalar arguments.
pub trait TensorMathScalar: TensorArray + Sized {
    fn add_scalar<'a>(&'a self, _rhs: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "add_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn div_scalar<'a>(&'a self, _rhs: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "div_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn log_scalar<'a>(&'a self, _base: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "log_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn mul_scalar<'a>(&'a self, _rhs: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "mul_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn pow_scalar<'a>(&'a self, _exp: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "pow_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn rem_scalar<'a>(&'a self, _rhs: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "rem_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn sub_scalar<'a>(&'a self, _rhs: Self::DType) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "sub_scalar is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Axis-wise tensor reductions.
pub trait TensorReduce: TensorArray + Sized {
    fn max<'a>(&'a self, _axes: Axes, _keepdims: bool) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "max reduction is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn min<'a>(&'a self, _axes: Axes, _keepdims: bool) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "min reduction is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn product<'a>(&'a self, _axes: Axes, _keepdims: bool) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "product reduction is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn sum<'a>(&'a self, _axes: Axes, _keepdims: bool) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "sum reduction is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Scalar tensor reductions.
pub trait TensorReduceAll: TensorArray {
    fn max_all<'a>(&'a self) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "max_all is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn min_all<'a>(&'a self) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "min_all is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn product_all<'a>(&'a self) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "product_all is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn sum_all<'a>(&'a self) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "sum_all is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Boolean scalar tensor reductions.
pub trait TensorReduceBoolean: TensorArray {
    fn all<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "all is not implemented for this tensor backend".to_string(),
            ))
        })
    }

    fn any<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "any is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}

/// Matrix/tensor contraction operations.
pub trait TensorMatMul: TensorArray + Sized {
    fn matmul_output_shape(&self, rhs: &Self) -> Result<Shape> {
        validate::matmul_output_shape(self.shape(), rhs.shape())
    }

    fn matmul<'a>(&'a self, _rhs: &'a Self) -> BoxFuture<'a, Result<Self>> {
        Box::pin(async move {
            Err(Error::Unsupported(
                "matmul is not implemented for this tensor backend".to_string(),
            ))
        })
    }
}
