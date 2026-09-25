//! Filesystem-backed tensors with lazy, bounded execution.
//!
//! Chaining operations constructs views; awaiting construction does not compute
//! or persist intermediates. Consumers evaluate bounded batches in memory.
//! Persistence requires an explicit write or [`Tensor::copy_from`]. Source-cache
//! spill/reload is separate from result persistence.
//!
//! Execution never collects a whole tensor, output, or reduction group. Batches
//! obey the [execution limit](https://github.com/TinyChain-Inc/fensor/blob/main/DESIGN.md#bound-and-policy-constants);
//! rank metadata, caller-owned values/selections,
//! filesystem/cache metadata, and independent consumers are separate costs.
//! Computed views are read-only, and streams are live rather than snapshots.
//!
//! Logical geometry uses [`Shape`] and [`Strides`] with `u64` elements; [`Axes`]
//! contains `usize` axis identifiers. [`Range`] uses fensor's [`AxisRange`].
//! [`TensorGeometry::size`] checks cardinality overflow and returns `Result<u64>`.
//! Only bounded buffer dimensions are narrowed for backend evaluation.
//!
//! ha-ndarray owns numerical parallelism; fensor owns bounded async concurrency.
//! The outer consumer uses CPU-limited `buffered` for row-major/sparse/boolean
//! delivery and `buffer_unordered` for coordinate streams and numeric terminals.
//! Synchronous backend work runs on the polling thread and may use backend workers;
//! `async move` does not create CPU parallelism. Inner evaluation adds no buffering.
//! Copying overlaps one sequential update and one lookahead with `try_join!`.
//!
//! Sparse support survives intermediate zeros. Numeric terminals accumulate in
//! completion order under the backend aggregate contract, with no input-order
//! error precedence. Boolean terminals retain logical short-circuit boundaries.
//! Dropping consumption cancels pending evaluation. Copy failures can leave partial
//! storage; call [`Tensor::sync`] explicitly before reopening it.
//!
//! ## Compose and consume without result storage
//!
//! This example is compile-checked. The caller supplies filesystem-backed inputs
//! with compatible matrix shapes and a callback for each coordinate/value pair.
//! No `FE: Clone` bound or concrete storage codec is required.
//!
//! ```no_run
//! use fensor::{
//!     Result, Tensor, TensorFileEntry, TensorMatMul, TensorMathScalar, TensorRead,
//!     TensorUnary,
//! };
//! use futures::TryStreamExt;
//!
//! async fn consume<FE: TensorFileEntry<f32>>(
//!     left: &Tensor<FE, f32>,
//!     right: &Tensor<FE, f32>,
//!     mut visit: impl FnMut(&[u64], f32),
//! ) -> Result<()> {
//!     let result = left.view()
//!         .matmul(&right.view()).await?
//!         .add_scalar(1.0).await?
//!         .exp().await?;
//!
//!     let mut batches = result.read_coordinate_blocks()?;
//!
//!     while let Some((coordinates, values)) = batches.try_next().await? {
//!         // Batch delivery order is unspecified; coordinates stay paired.
//!         for (coordinate, value) in coordinates.iter().zip(values) {
//!             visit(coordinate, value);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod binary;

mod error;

mod expression;

mod mapping;

mod matmul;

mod matrix;

mod metadata;

#[cfg(test)]
mod read_metrics;

#[cfg(test)]
extern crate self as fensor;

#[cfg(all(test, feature = "benchmarks"))]
mod profiling;

pub mod reduce;

mod request;

pub mod scalar;

mod schema;

mod selection;

mod slice;

mod storage_read;

mod tensor;

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_support;

mod traits;

pub mod unary;

mod validate;

mod view;

pub use binary::BinaryView;
pub use error::{Error, Result};
pub use ha_ndarray::Axes;
pub use matmul::MatMulView;
pub use matrix::DiagView;
pub use metadata::TensorMetadata;
pub use number_general::NumberType;
pub use reduce::ReduceView;
pub use schema::{
    AxisRange, Layout, Range, Shape, SparseIndexSchema, SparseTableSchema, Strides, TensorSchema,
    contiguous_strides,
};
pub use selection::WhereView;
pub use traits::{
    BoxFuture, CoordinateBlockStream, SparseElementStream, TensorAbs, TensorArray,
    TensorBlockStore, TensorBoolean, TensorBooleanScalar, TensorCast, TensorCompare,
    TensorCompareScalar, TensorGeometry, TensorMatMul, TensorMath, TensorMathScalar,
    TensorMatrixUnary, TensorNumeric, TensorRead, TensorReduce, TensorReduceAll,
    TensorReduceBoolean, TensorSparseIndex, TensorTransform, TensorTrig, TensorUnary,
    TensorUnaryBoolean, TensorViewSemantics, TensorWhere, TensorWrite, TensorWriteBulk,
    ValueBlockStream,
};

pub use tensor::{Tensor, TensorElement, TensorFileEntry};
pub use unary::UnaryView;
pub use view::TensorView;

pub(crate) const PORTABLE_INLINE_RANK: usize = 8;
