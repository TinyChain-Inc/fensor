use std::future::Future;
use std::pin::Pin;

use futures::{Stream, StreamExt, TryStreamExt};
use ha_ndarray::{Axes, Range, Shape};
use number_general::NumberType;

use crate::schema::Layout;
use crate::validate;
use crate::{Error, Result, TensorSchema};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Minimal shape/dtype surface shared by base tensors AND their views.
pub trait TensorGeometry: Send + Sync {
    type DType: Copy + Send + Sync + 'static;

    fn dtype(&self) -> NumberType;

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

    fn schema_dtype(&self) -> NumberType {
        self.schema().dtype()
    }
}

/// A lazily-produced, row-major-ordered stream of populated sparse elements.
pub type SparseElementStream<'a, ET> =
    Pin<Box<dyn Stream<Item = Result<(Vec<u64>, ET)>> + Send + 'a>>;

/// Bounded batches of values in logical row-major order, independent of storage tiling.
pub type ValueBlockStream<'a, T> = Pin<Box<dyn Stream<Item = Result<Vec<T>>> + Send + 'a>>;

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
pub trait TensorUnary: TensorGeometry + Sized {
    type ExpOutput: TensorRead<DType = Self::DType>;
    type LnOutput: TensorRead<DType = Self::DType>;
    type RoundOutput: TensorRead<DType = Self::DType>;

    fn exp(&self) -> BoxFuture<'_, Result<Self::ExpOutput>>;
    fn ln(&self) -> BoxFuture<'_, Result<Self::LnOutput>>;
    fn round(&self) -> BoxFuture<'_, Result<Self::RoundOutput>>;
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
/// async fn mask<S, D>(tensor: &Tensor<S, f32>, dir: DirLock<D>) -> Result<Tensor<D, u8>>
/// where
///     S: TensorFileEntry<f32>,
///     D: TensorFileEntry<u8>,
/// {
///     let mask = tensor.view().is_nan().await?.not().await?.clone();
///     Tensor::copy_from(dir, &mask, 4096).await
/// }
/// ```
pub trait TensorNumeric: TensorGeometry + Sized {
    type IsNanOutput: TensorRead<DType = u8>;
    type IsInfOutput: TensorRead<DType = u8>;

    fn is_nan(&self) -> BoxFuture<'_, Result<Self::IsNanOutput>>;
    fn is_inf(&self) -> BoxFuture<'_, Result<Self::IsInfOutput>>;
}

/// Lazy element-type conversion. Currently supports f32 to f64.
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
/// ) -> Result<Tensor<Destination, f64>>
/// where
///     Source: TensorFileEntry<f32>,
///     Destination: TensorFileEntry<f64>,
/// {
///     let view = source.view();
///     let cast = TensorCast::<f64>::cast(&view).await?;
///     Tensor::copy_from(dir, &cast, 4096).await
/// }
/// ```
///
/// Narrowing is not supported:
///
/// ```compile_fail,E0277
/// use fensor::{Tensor, TensorCast, TensorFileEntry};
/// async fn narrow<FE: TensorFileEntry<f64>>(source: &Tensor<FE, f64>) {
///     let _ = TensorCast::<f32>::cast(&source.view()).await;
/// }
/// ```
pub trait TensorCast<To: crate::TensorElement>: TensorGeometry + Sized {
    type Output: TensorRead<DType = To>;

    fn cast(&self) -> BoxFuture<'_, Result<Self::Output>>;
}

/// Elementwise absolute value preserving the stored element type.
pub trait TensorAbs: TensorGeometry + Sized {
    type Output: TensorRead<DType = Self::DType>;

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
    type RemOutput: TensorRead<DType = Self::DType>;

    fn add<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::AddOutput>>;

    fn sub<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::SubOutput>>;

    fn mul<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::MulOutput>>;

    fn div<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::DivOutput>>;

    fn pow<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::PowOutput>>;

    fn log<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::LogOutput>>
    where
        Self::DType: ha_ndarray::Float;

    fn rem<'a>(&'a self, rhs: &'a Rhs) -> BoxFuture<'a, Result<Self::RemOutput>>;
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

/// Batch coordinates without expanding the remaining logical range.
pub(crate) fn coordinate_batches(
    mut coords: impl Iterator<Item = Vec<u64>>,
) -> impl Iterator<Item = Vec<Vec<u64>>> {
    std::iter::from_fn(move || {
        let batch: Vec<_> = coords
            .by_ref()
            .take(crate::schema::MAX_BLOCK_CAPACITY)
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
        if let ha_ndarray::AxisRange::Of(indices) = axis {
            indices.sort_unstable();
            indices.dedup();
        }
    }

    validate::iter_range_coords(tensor.shape(), &range)
}
