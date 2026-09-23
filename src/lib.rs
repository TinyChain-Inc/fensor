//! Filesystem-backed tensors with bounded execution.
//!
//! Tensor execution must not allocate values, coordinates, support masks, or
//! partial-result collections proportional to total tensor size, output size, or
//! reduction-group size. Coordinates must be generated lazily and consumed in
//! bounded batches. Each collection must have an identifiable bound. Only the outer
//! consumer may introduce concurrent batches.
//!
//! Rank-sized metadata and caller-supplied values or explicit index selections are
//! separate, documented memory costs. They must not justify expanding implicit
//! ranges or collecting execution results. Filesystem/cache metadata remains subject
//! to its own limits; this contract is not a total-process memory guarantee.
//!
//! Execution batches are limited to 4096 elements independently of storage block
//! capacity. Read streams and `Tensor::copy_from` consume bounded batches; callers
//! explicitly collect streams when they want a whole in-memory result.
//! `TensorWriteBulk` consumes caller-owned buffers without collecting coordinates.
//! Arbitrary gather selections retain input-sized metadata; slicing and flipping
//! an existing gather share its table. No whole-result collection or compaction
//! convenience API is provided.

pub mod binary;
mod error;
mod expression;
mod mapping;
mod metadata;
pub mod reduce;
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
pub use reduce::ReduceView;
pub use schema::{
    Layout, SparseIndexSchema, SparseTableSchema, TensorSchema, TensorShape, contiguous_strides,
};
pub use selection::WhereView;
pub use traits::{
    BoxFuture, SparseElementStream, TensorAbs, TensorArray, TensorBlockStore, TensorBoolean,
    TensorBooleanScalar, TensorCast, TensorCompare, TensorCompareScalar, TensorGeometry,
    TensorMatMul, TensorMath, TensorMathScalar, TensorNumeric, TensorRead, TensorReduce,
    TensorReduceAll, TensorReduceBoolean, TensorSparseIndex, TensorTransform, TensorTrig,
    TensorUnary, TensorUnaryBoolean, TensorViewSemantics, TensorWhere, TensorWrite,
    TensorWriteBulk, ValueBlockStream,
};

pub use tensor::{Tensor, TensorElement, TensorFileEntry};
pub use unary::UnaryView;
pub use view::TensorView;

pub(crate) const PORTABLE_INLINE_RANK: usize = 8;
