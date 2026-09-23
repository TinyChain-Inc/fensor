pub mod binary;
mod error;
mod expression;
mod metadata;
pub mod scalar;
mod schema;
mod selection;
mod tensor;
mod traits;
pub mod unary;
mod validate;
mod view;

pub use binary::BinaryView;
pub use error::{Error, Result};
pub use metadata::TensorMetadata;
pub use number_general::NumberType;
pub use schema::{
    Layout, SparseIndexSchema, SparseTableSchema, TensorSchema, TensorShape, contiguous_strides,
};
pub use selection::WhereView;
pub use traits::{
    BoxFuture, SparseElementStream, TensorAbs, TensorArray, TensorBlockStore, TensorBoolean,
    TensorBooleanScalar, TensorCast, TensorCompare, TensorCompareScalar, TensorGeometry,
    TensorMatMul, TensorMath, TensorMathScalar, TensorNumeric, TensorRead, TensorReadBulk,
    TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorSparseIndex, TensorTransform,
    TensorTrig, TensorUnary, TensorUnaryBoolean, TensorViewSemantics, TensorWhere, TensorWrite,
    TensorWriteBulk, ValueBlockStream,
};

pub use tensor::{Tensor, TensorElement, TensorFileEntry};
pub use unary::UnaryView;
pub use view::TensorView;

pub(crate) const PORTABLE_INLINE_RANK: usize = 8;
