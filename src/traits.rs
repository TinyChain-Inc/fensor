use std::future::Future;
use std::pin::Pin;

use futures::{Stream, StreamExt, TryStreamExt};
use number_general::NumberType;

use crate::schema::Layout;
use crate::{Axes, Error, Range, Result, Shape, TensorSchema, validate};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Minimal shape/dtype surface shared by base tensors AND their views.
pub trait TensorGeometry: Send + Sync {
    type DType: Copy + Send + Sync + 'static;

    fn dtype(&self) -> NumberType;

    fn layout(&self) -> Layout;

    fn shape(&self) -> &[u64];

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Checked logical cardinality, independent of the machine-sized batch buffers.
    fn size(&self) -> Result<u64> {
        crate::schema::checked_product(self.shape())
    }
}

/// Adds the persistent, storage-backed schema -- implemented ONLY by base tensors.
pub trait TensorArray: TensorGeometry {
    fn schema(&self) -> &TensorSchema;

    fn strides(&self) -> &[u64];

    fn schema_dtype(&self) -> NumberType {
        self.schema().dtype()
    }
}

/// A lazily-produced, row-major-ordered stream of populated sparse elements.
pub type SparseElementStream<'a, ET> =
    Pin<Box<dyn Stream<Item = Result<(Vec<u64>, ET)>> + Send + 'a>>;

/// Bounded batches of values in logical row-major order, independent of storage tiling.
pub type ValueBlockStream<'a, T> = Pin<Box<dyn Stream<Item = Result<Vec<T>>> + Send + 'a>>;

/// Coordinate-bearing bounded blocks, in implementation-selected order.
/// Successful complete consumption visits every logical coordinate exactly once,
/// including zero-valued coordinates. Built-in readers deliver completed batches
/// without input-order error precedence; dropping a stream cancels pending work.
/// Coordinates and values have equal length within the
/// [execution limit](https://github.com/TinyChain-Inc/fensor/blob/main/DESIGN.md#bound-and-policy-constants).
/// Each call is independent.
pub type CoordinateBlockStream<'a, T> =
    Pin<Box<dyn Stream<Item = Result<(Vec<Vec<u64>>, Vec<T>)>> + Send + 'a>>;

/// Async value reads aligned with ndarray coordinate semantics.
pub trait TensorRead: TensorGeometry {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>>;

    /// Each call constructs an independent, demand-driven stream; no tensor-sized buffer.
    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        let batches = coordinate_batches(crate::schema::row_major_coords(self.shape())?);
        Ok(futures::stream::iter(batches)
            .map(move |coords| async move {
                let mut values = Vec::with_capacity(coords.len());

                for coord in coords {
                    values.push(self.read_value(&coord).await?);
                }

                Ok(values)
            })
            .buffered(num_cpus::get().max(1))
            .boxed())
    }

    /// Read all logical coordinates in bounded blocks, without promising row-major order.
    /// Exactly-once coverage is an implementer's contract, not globally tracked state.
    fn read_coordinate_blocks(&self) -> Result<CoordinateBlockStream<'_, Self::DType>> {
        let coords = crate::schema::row_major_coords(self.shape())?;
        let blocks = self.read_blocks()?;
        Ok(
            futures::stream::try_unfold((coords, blocks), |(mut coords, mut blocks)| async move {
                let Some(values) = blocks.try_next().await? else {
                    if coords.next().is_some() {
                        return Err(Error::InvalidLayout(
                            "reader returned fewer values than its shape".into(),
                        ));
                    }
                    return Ok(None);
                };
                if values.len() > crate::expression::MAX_BATCH_ELEMENTS {
                    return Err(Error::InvalidLayout(format!(
                        "coordinate block: expected at most {} values, got {}",
                        crate::expression::MAX_BATCH_ELEMENTS,
                        values.len()
                    )));
                }

                let selected: Vec<_> = coords.by_ref().take(values.len()).collect();
                if selected.len() != values.len() {
                    return Err(Error::InvalidLayout(
                        "reader returned more values than its shape".into(),
                    ));
                }

                Ok(Some(((selected, values), (coords, blocks))))
            })
            .boxed(),
        )
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>>
    where
        Self::DType: Default + PartialEq,
    {
        Box::pin(async move {
            let coords = sparse_coords(self, range, requested_order)?;
            let elements = futures::stream::iter(coordinate_batches(coords))
                .map(move |coords| async move {
                    let mut elements = Vec::new();

                    for coord in coords {
                        let value = self.read_value(&coord).await?;
                        if value != Self::DType::default() {
                            elements.push((coord, value));
                        }
                    }
                    Ok::<_, Error>(futures::stream::iter(elements.into_iter().map(Ok)))
                })
                .buffered(num_cpus::get().max(1))
                .try_flatten();
            Ok(elements.boxed())
        })
    }
}

/// Async value writes aligned with ndarray coordinate semantics.
pub trait TensorWrite: TensorGeometry {
    fn write_value<'a>(&'a self, coord: &'a [u64], value: Self::DType)
    -> BoxFuture<'a, Result<()>>;
}

/// Bulk writes consume caller-owned values without collecting their coordinates.
/// The caller budgets the supplied buffer; tensor-to-tensor writes and fill iterate lazily.
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

/// Async storage-block access, bounded by validated storage block capacity.
/// A block is not a whole-tensor collection.
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
pub trait TensorUnary: TensorGeometry + Sized {
    type ExpOutput: TensorRead<DType = Self::DType>;

    type LnOutput: TensorRead<DType = Self::DType>;

    type RoundOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Real;

    fn exp(&self) -> BoxFuture<'_, Result<Self::ExpOutput>>;

    fn ln(&self) -> BoxFuture<'_, Result<Self::LnOutput>>;

    fn round(&self) -> BoxFuture<'_, Result<Self::RoundOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Logical negation evaluated only on original support for sparse tensors.
pub trait TensorUnaryBoolean: TensorGeometry + Sized {
    type Output: TensorRead<DType = u8>;

    fn not(&self) -> BoxFuture<'_, Result<Self::Output>>;
}

/// Floating-point predicates returning u8 masks.
///
/// Predicate outputs cannot themselves be used as floating-point inputs:
///
/// ```compile_fail,E0277
/// use fensor::{Tensor, TensorFileEntry, TensorNumeric};
/// async fn unsupported<FE: TensorFileEntry<u8>>(tensor: &Tensor<FE, u8>) {
///     let _ = TensorNumeric::is_nan(&tensor.view()).await;
/// }
/// ```
///
/// Source and destination adapters need only support their respective dtypes:
///
/// ```
/// use fensor::{Result, Tensor, TensorFileEntry, TensorNumeric, TensorUnaryBoolean};
/// use freqfs::DirLock;
/// async fn mask<S, D>(
///     tensor: &Tensor<S, f32>,
///     dir: DirLock<D>,
///     max_capacity: usize,
/// ) -> Result<Tensor<D, u8>>
/// where
///     S: TensorFileEntry<f32>,
///     D: TensorFileEntry<u8>,
/// {
///     let mask = tensor.view().is_nan().await?.not().await?.clone();
///     Tensor::copy_from(dir, &mask, max_capacity).await
/// }
/// ```
pub trait TensorNumeric: TensorGeometry + Sized {
    type IsNanOutput: TensorRead<DType = u8>;

    type IsInfOutput: TensorRead<DType = u8>;

    fn is_nan(&self) -> BoxFuture<'_, Result<Self::IsNanOutput>>;

    fn is_inf(&self) -> BoxFuture<'_, Result<Self::IsInfOutput>>;
}

/// Lazy conversion between supported concrete element types, including narrowing.
/// Uses ha-ndarray's number-general conversion pipeline, not Rust `as` casts.
/// Select the destination with result typing or `TensorCast::<To>::cast(&source)`.
///
/// The destination storage adapter need only support the output dtype:
///
/// ```
/// use fensor::{Result, Tensor, TensorCast, TensorFileEntry};
/// use freqfs::DirLock;
///
/// async fn widen<Source, Destination>(
///     source: &Tensor<Source, f32>,
///     dir: DirLock<Destination>,
///     max_capacity: usize,
/// ) -> Result<Tensor<Destination, f64>>
/// where
///     Source: TensorFileEntry<f32>,
///     Destination: TensorFileEntry<f64>,
/// {
///     let view = source.view();
///     let cast = TensorCast::<f64>::cast(&view).await?;
///     Tensor::copy_from(dir, &cast, max_capacity).await
/// }
/// ```
///
/// Narrowing follows the backend conversion contract:
///
/// ```
/// use fensor::{Tensor, TensorCast, TensorFileEntry};
/// async fn narrow<FE: TensorFileEntry<f64>>(source: &Tensor<FE, f64>) {
///     let _ = TensorCast::<f32>::cast(&source.view()).await;
/// }
/// ```
pub trait TensorCast<To: crate::TensorElement>: TensorGeometry + Sized {
    type Output: TensorRead<DType = To>;

    fn cast(&self) -> BoxFuture<'_, Result<Self::Output>>;
}

/// Elementwise absolute value; complex magnitudes have the component's real dtype.
pub trait TensorAbs: TensorGeometry + Sized
where
    Self::DType: ha_ndarray::Number,
{
    type Output: TensorRead<DType = <Self::DType as ha_ndarray::Number>::Abs>;

    fn abs(&self) -> BoxFuture<'_, Result<Self::Output>>;
}

/// Elementwise trigonometry, evaluated lazily over source support.
pub trait TensorTrig: TensorGeometry + Sized {
    type SinOutput: TensorRead<DType = Self::DType>;

    type AsinOutput: TensorRead<DType = Self::DType>;

    type SinhOutput: TensorRead<DType = Self::DType>;

    type CosOutput: TensorRead<DType = Self::DType>;

    type AcosOutput: TensorRead<DType = Self::DType>;

    type CoshOutput: TensorRead<DType = Self::DType>;

    type TanOutput: TensorRead<DType = Self::DType>;

    type AtanOutput: TensorRead<DType = Self::DType>;

    type TanhOutput: TensorRead<DType = Self::DType>;

    fn sin(&self) -> BoxFuture<'_, Result<Self::SinOutput>>;

    fn asin(&self) -> BoxFuture<'_, Result<Self::AsinOutput>>;

    fn sinh(&self) -> BoxFuture<'_, Result<Self::SinhOutput>>;

    fn cos(&self) -> BoxFuture<'_, Result<Self::CosOutput>>;

    fn acos(&self) -> BoxFuture<'_, Result<Self::AcosOutput>>;

    fn cosh(&self) -> BoxFuture<'_, Result<Self::CoshOutput>>;

    fn tan(&self) -> BoxFuture<'_, Result<Self::TanOutput>>;

    fn atan(&self) -> BoxFuture<'_, Result<Self::AtanOutput>>;

    fn tanh(&self) -> BoxFuture<'_, Result<Self::TanhOutput>>;
}

/// Elementwise tensor math operations.
pub trait TensorMath<Rhs = Self>: TensorGeometry + Sized
where
    Rhs: TensorGeometry<DType = Self::DType>,
{
    type AddOutput: TensorRead<DType = Self::DType>;

    type SubOutput: TensorRead<DType = Self::DType>;

    type MulOutput: TensorRead<DType = Self::DType>;

    type DivOutput: TensorRead<DType = Self::DType>;

    type PowOutput: TensorRead<DType = Self::DType>;

    type LogOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Float;
    type RemOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Real;

    fn add<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::AddOutput>>;

    fn sub<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::SubOutput>>;

    fn mul<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::MulOutput>>;

    fn div<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::DivOutput>>;

    fn pow<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::PowOutput>>;

    fn log<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::LogOutput>>
    where
        Self::DType: ha_ndarray::Float;

    fn rem<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::RemOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Lazy elementwise arithmetic with scalar arguments.
///
/// Scalars preserve the source's support: implicit sparse zeros stay absent even
/// when the operation would map zero to a nonzero value. Arithmetic follows
/// ha-ndarray's numerical contract, including wrapping u8 operations.
pub trait TensorMathScalar: TensorGeometry {
    type AddOutput: TensorRead<DType = Self::DType>;

    type SubOutput: TensorRead<DType = Self::DType>;

    type MulOutput: TensorRead<DType = Self::DType>;

    type DivOutput: TensorRead<DType = Self::DType>;

    type PowOutput: TensorRead<DType = Self::DType>;

    type LogOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Float;

    type RemOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Real;

    fn add_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::AddOutput>>;

    fn sub_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::SubOutput>>;

    fn mul_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::MulOutput>>;

    fn div_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::DivOutput>>;

    fn pow_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::PowOutput>>;

    fn log_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::LogOutput>>
    where
        Self::DType: ha_ndarray::Float;

    fn rem_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::RemOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Lazy elementwise comparisons returning exactly zero or one.
///
/// Operands must have matching shapes and dtypes. Sparse expressions retain the
/// union of original source support; comparisons do not populate absent values.
/// Floating comparisons follow IEEE unordered-NaN and signed-zero rules.
pub trait TensorCompare<Rhs = Self>: TensorGeometry
where
    Rhs: TensorGeometry<DType = Self::DType>,
{
    type EqOutput: TensorRead<DType = u8>;

    type NeOutput: TensorRead<DType = u8>;

    type GtOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type GeOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type LtOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type LeOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    fn eq<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::EqOutput>>;

    fn ne<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::NeOutput>>;

    fn gt<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::GtOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn ge<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::GeOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn lt<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::LtOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn le<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::LeOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Lazy elementwise comparisons with scalar arguments, returning zero or one.
///
/// Source support is preserved: comparing implicit sparse zeros to zero does
/// not populate them.
pub trait TensorCompareScalar: TensorGeometry {
    type EqOutput: TensorRead<DType = u8>;

    type NeOutput: TensorRead<DType = u8>;

    type GtOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type GeOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type LtOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    type LeOutput: TensorRead<DType = u8>
    where
        Self::DType: ha_ndarray::Real;

    fn eq_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::EqOutput>>;

    fn ne_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::NeOutput>>;

    fn gt_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::GtOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn ge_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::GeOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn lt_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::LtOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn le_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::LeOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Lazy elementwise logical operations returning exactly zero or one.
///
/// Zero is false; nonzero values, including NaN, are true. These are not bitwise
/// operations. Shapes and dtypes must match; sparse support is their union.
pub trait TensorBoolean<Rhs = Self>: TensorGeometry
where
    Rhs: TensorGeometry<DType = Self::DType>,
{
    type AndOutput: TensorRead<DType = u8>;

    type OrOutput: TensorRead<DType = u8>;

    type XorOutput: TensorRead<DType = u8>;

    fn and<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::AndOutput>>;

    fn or<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::OrOutput>>;

    fn xor<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::XorOutput>>;
}

/// Lazy elementwise logical operations with scalar arguments.
///
/// Zero is false and nonzero is true, including NaN. Results are zero or one;
/// scalars do not add support to implicit sparse coordinates.
pub trait TensorBooleanScalar: TensorGeometry {
    type AndOutput: TensorRead<DType = u8>;

    type OrOutput: TensorRead<DType = u8>;

    type XorOutput: TensorRead<DType = u8>;

    fn and_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::AndOutput>>;

    fn or_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::OrOutput>>;

    fn xor_scalar<'a>(&'a self, rhs: Self::DType) -> BoxFuture<'a, Result<Self::XorOutput>>;
}

/// Lazy selection with union-of-source-support semantics.
///
/// A zero condition selects `or_else`; any nonzero condition selects `then`.
/// All shapes must match and branch dtypes must match. Both branches are read,
/// including an unselected branch, and their errors propagate. Sparse support
/// is the union of the condition and both branches, independent of selection.
pub trait TensorWhere<Then, Else>: TensorGeometry<DType = u8>
where
    Then: TensorGeometry,
    Else: TensorGeometry<DType = Then::DType>,
{
    type Output: TensorRead<DType = Then::DType>;

    fn cond<'a>(&'a self, then: &'a Then, or_else: &'a Else)
    -> BoxFuture<'a, Result<Self::Output>>;
}

/// Lazy axis reductions over retained source support, including intermediate zeros.
/// Empty sparse groups remain absent. Axes are sorted and deduplicated.
pub trait TensorReduce: TensorGeometry {
    type SumOutput: TensorRead<DType = Self::DType>;

    type ProductOutput: TensorRead<DType = Self::DType>;

    type MinOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Real;

    type MaxOutput: TensorRead<DType = Self::DType>
    where
        Self::DType: ha_ndarray::Real;

    fn sum(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::SumOutput>>;

    fn product(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::ProductOutput>>;

    fn min(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::MinOutput>>
    where
        Self::DType: ha_ndarray::Real;

    fn max(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::MaxOutput>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Terminal reductions over retained source support, not implicit sparse zeros.
/// Empty support gives sum=0 and product=1; extrema return an error.
/// Built-in expressions accumulate batches in completion order. Floating results
/// may depend on scheduling within ha-ndarray's aggregate contract, including
/// permitted extreme-range differences. The first observed error cancels pending
/// evaluation; errors have no input-order precedence. Extrema do not short-circuit.
pub trait TensorReduceAll: TensorRead {
    fn sum_all(&self) -> BoxFuture<'_, Result<Self::DType>>;

    fn product_all(&self) -> BoxFuture<'_, Result<Self::DType>>;

    fn min_all(&self) -> BoxFuture<'_, Result<Self::DType>>
    where
        Self::DType: ha_ndarray::Real;

    fn max_all(&self) -> BoxFuture<'_, Result<Self::DType>>
    where
        Self::DType: ha_ndarray::Real;
}

/// Short-circuit boolean reductions over retained source support.
/// Empty support gives all=true and any=false. Errors after a decisive batch
/// may remain unobserved; dropping the remaining stream cancels pending work.
pub trait TensorReduceBoolean: TensorRead {
    fn all(&self) -> BoxFuture<'_, Result<bool>>;

    fn any(&self) -> BoxFuture<'_, Result<bool>>;
}

/// Matrix geometry for the final two axes; stored tensors use `.view()`.
/// Construction is lazy, including when awaited.
pub trait TensorMatrixUnary: TensorGeometry {
    type TransposeOutput: TensorRead<DType = Self::DType>;

    type DiagOutput: TensorRead<DType = Self::DType>;

    /// Swap the final two axes, preserving batch axes and existing write constraints.
    fn mt(&self) -> BoxFuture<'_, Result<Self::TransposeOutput>>;

    /// Extract square matrix diagonals: `[..., N, N]` becomes `[..., N]`.
    /// Only selected source coordinates contribute support.
    fn diag(&self) -> BoxFuture<'_, Result<Self::DiagOutput>>;
}

/// Lazy conjugate transpose of the final two axes, preserving batch axes.
#[cfg(feature = "complex")]
pub trait TensorMatrixUnaryComplex: TensorGeometry {
    type HermitianOutput: TensorRead<DType = Self::DType>;

    fn mh(&self) -> BoxFuture<'_, Result<Self::HermitianOutput>>;
}

/// Bounded, unnormalized last-axis Fourier transforms of complex expressions.
///
/// Construction rejects axes exceeding the execution batch limit. Point reads
/// evaluate the complete corresponding axis group. Any supported input makes
/// every frequency in that group supported; an empty sparse group stays absent.
#[cfg(feature = "complex")]
pub trait TensorFourier: TensorGeometry {
    type FftOutput: TensorRead<DType = Self::DType>;
    type IfftOutput: TensorRead<DType = Self::DType>;

    fn fft(&self) -> BoxFuture<'_, Result<Self::FftOutput>>;

    fn ifft(&self) -> BoxFuture<'_, Result<Self::IfftOutput>>;
}

/// Lazy matrix multiplication with matching dtypes and explicit batch broadcasting.
///
/// Shapes must be `[..., M, K]` and `[..., K, N]`, with identical batch dimensions
/// and rank at least two. The output is `[..., M, N]`; shape/size errors are
/// rejected before construction. Use transforms to broadcast explicitly.
/// Sparse output support unions both operands across contraction positions,
/// retaining supported zero results. See [`crate::MatMulView`] for composition.
pub trait TensorMatMul<Rhs = Self>: TensorGeometry
where
    Rhs: TensorGeometry<DType = Self::DType>,
{
    type Output: TensorRead<DType = Self::DType>;

    fn matmul_output_shape(&self, rhs: &Rhs) -> Result<Shape> {
        validate::matmul_output_shape(self.shape(), rhs.shape())
    }

    fn matmul<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::Output>>;
}

/// Batch coordinates without expanding the remaining logical range.
pub(crate) fn coordinate_batches(
    mut coords: impl Iterator<Item = Vec<u64>>,
) -> impl Iterator<Item = Vec<Vec<u64>>> {
    std::iter::from_fn(move || {
        let batch: Vec<_> = coords
            .by_ref()
            .take(crate::expression::MAX_BATCH_ELEMENTS)
            .collect();
        (!batch.is_empty()).then_some(batch)
    })
}

pub(crate) fn sparse_coords<V: TensorGeometry + ?Sized>(
    tensor: &V,
    range: Range,
    requested_order: Axes,
) -> Result<validate::RangeCoords> {
    let base_order: Vec<_> = (0..tensor.ndim()).collect();

    if requested_order.as_slice() != base_order.as_slice() {
        return Err(Error::UnsupportedSparseIterationOrder {
            requested_order: requested_order.to_vec(),
            base_order,
            hint: "materialize or perform external sort for incompatible order".into(),
        });
    }

    if !matches!(tensor.layout(), Layout::Sparse { .. }) {
        return Err(Error::Unsupported(
            "sparse iteration requires sparse layout".into(),
        ));
    }

    crate::schema::validate_shape_dims(tensor.shape())?;
    // Sparse ranges select a set of coordinates, independent of selection order.
    let mut range = range;

    for axis in &mut range {
        if let crate::AxisRange::Of(indices) = axis {
            indices.sort_unstable();
            indices.dedup();
        }
    }

    validate::iter_range_coords(tensor.shape(), &range)
}

/// Complex elementwise projections, retaining source support.
///
/// Descriptions clone without requiring adapters to clone. Only the destination
/// adapter must support the real output of a component projection:
///
/// ```
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorComplex, TensorCast, TensorRead};
/// async fn project<A: TensorFileEntry<Complex32>, B: TensorFileEntry<f64>>(
///     source: &Tensor<A, Complex32>, dir: freqfs::DirLock<B>,
/// ) -> fensor::Result<Tensor<B, f64>> {
///     let real = source.view().conj().await?.re().await?.clone();
///     let widened = TensorCast::<f64>::cast(&real).await?;
///     Tensor::copy_from(dir, &widened, 16).await
/// }
/// ```
///
/// Complex projections remain read-only:
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorComplex, TensorWrite};
/// fn writable<V: TensorWrite>(_: V) {}
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     writable(t.view().re().await.unwrap());
/// }
/// ```
///
/// Complex ordering and extrema are unavailable; equality and sum/product remain supported:
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorCompare};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     let _ = t.view().gt(&t.view()).await;
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorReduceAll};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     let _ = t.min_all().await;
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorReduce, Axes};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     let _ = t.view().max(Axes::new(), false).await;
/// }
/// ```
///
/// Remainder and rounding also require real data:
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorMathScalar};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     let _ = t.view().rem_scalar(Complex32::new(1., 0.)).await;
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorUnary};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     let _ = t.view().round().await;
/// }
/// ```
#[cfg(feature = "complex")]
pub trait TensorComplex: TensorGeometry + Sized
where
    Self::DType: ha_ndarray::Complex,
{
    type ConjOutput: TensorRead<DType = Self::DType>;

    type ReOutput: TensorRead<DType = <Self::DType as ha_ndarray::Complex>::Real>;

    type ImOutput: TensorRead<DType = <Self::DType as ha_ndarray::Complex>::Real>;

    type AngleOutput: TensorRead<DType = <Self::DType as ha_ndarray::Complex>::Real>;

    fn conj(&self) -> BoxFuture<'_, Result<Self::ConjOutput>>;

    fn re(&self) -> BoxFuture<'_, Result<Self::ReOutput>>;

    fn im(&self) -> BoxFuture<'_, Result<Self::ImOutput>>;

    fn angle(&self) -> BoxFuture<'_, Result<Self::AngleOutput>>;
}
