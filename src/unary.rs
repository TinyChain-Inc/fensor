//! Typed, read-only unary views over tensor expressions.

use std::marker::PhantomData;

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{
    ArrayAccess, Axes, NDArrayAbs, NDArrayCast, NDArrayNumeric, NDArrayTrig, NDArrayUnary,
    NDArrayUnaryBoolean, Range, Shape,
};

use crate::Result;
use crate::expression::{self, Batch, Expression};
use crate::schema::Layout;
use crate::tensor::TensorElement;
use crate::traits::{
    BoxFuture, SparseElementStream, TensorAbs, TensorCast, TensorGeometry, TensorNumeric,
    TensorRead, TensorTransform, TensorTrig, TensorUnary, TensorUnaryBoolean, TensorViewSemantics,
    ValueBlockStream,
};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// An operation which builds an ndarray expression without evaluating it.
///
/// This trait is sealed: only the supported unary operations can be used
/// in a `UnaryView`.
pub trait UnaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type Output: TensorElement;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>>;
}

/// Elementwise exponentiation.
#[derive(Clone, Copy, Debug)]
pub struct Exp;

/// Elementwise natural logarithm.
#[derive(Clone, Copy, Debug)]
pub struct Ln;

/// Elementwise rounding.
#[derive(Clone, Copy, Debug)]
pub struct Round;

impl sealed::Sealed for Exp {}
impl sealed::Sealed for Ln {}
impl sealed::Sealed for Round {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Exp {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.exp()?))
    }
}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Ln {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.ln()?))
    }
}

impl<T: TensorElement + ha_ndarray::Float + ha_ndarray::Real> UnaryOp<T> for Round {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.round()?))
    }
}

/// Elementwise absolute value.
#[derive(Clone, Copy, Debug)]
pub struct Abs;

impl sealed::Sealed for Abs {}

impl<T: TensorElement + ha_ndarray::Number<Abs = T>> UnaryOp<T> for Abs {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.abs()?))
    }
}

/// Elementwise sine.
#[derive(Clone, Copy, Debug)]
pub struct Sin;

impl sealed::Sealed for Sin {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Sin {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.sin()?))
    }
}

/// Elementwise inverse sine.
#[derive(Clone, Copy, Debug)]
pub struct Asin;

impl sealed::Sealed for Asin {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Asin {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.asin()?))
    }
}

/// Elementwise hyperbolic sine.
#[derive(Clone, Copy, Debug)]
pub struct Sinh;

impl sealed::Sealed for Sinh {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Sinh {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.sinh()?))
    }
}

/// Elementwise cosine.
#[derive(Clone, Copy, Debug)]
pub struct Cos;

impl sealed::Sealed for Cos {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Cos {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.cos()?))
    }
}

/// Elementwise inverse cosine.
#[derive(Clone, Copy, Debug)]
pub struct Acos;

impl sealed::Sealed for Acos {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Acos {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.acos()?))
    }
}

/// Elementwise hyperbolic cosine.
#[derive(Clone, Copy, Debug)]
pub struct Cosh;

impl sealed::Sealed for Cosh {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Cosh {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.cosh()?))
    }
}

/// Elementwise tangent.
#[derive(Clone, Copy, Debug)]
pub struct Tan;

impl sealed::Sealed for Tan {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Tan {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.tan()?))
    }
}

/// Elementwise inverse tangent.
#[derive(Clone, Copy, Debug)]
pub struct Atan;

impl sealed::Sealed for Atan {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Atan {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.atan()?))
    }
}

/// Elementwise hyperbolic tangent.
#[derive(Clone, Copy, Debug)]
pub struct Tanh;

impl sealed::Sealed for Tanh {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Tanh {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.tanh()?))
    }
}

/// Elementwise logical negation, returning 0 or 1.
#[derive(Clone, Copy, Debug)]
pub struct Not;

impl sealed::Sealed for Not {}

impl<T: TensorElement> UnaryOp<T> for Not {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, u8>> {
        Ok(ArrayAccess::from(array.not()?))
    }
}

/// Elementwise NaN detection, returning 0 or 1.
#[derive(Clone, Copy, Debug)]
pub struct IsNan;

impl sealed::Sealed for IsNan {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for IsNan {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, u8>> {
        Ok(ArrayAccess::from(array.is_nan()?))
    }
}

/// Elementwise infinity detection, returning 0 or 1.
#[derive(Clone, Copy, Debug)]
pub struct IsInf;

impl sealed::Sealed for IsInf {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for IsInf {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, u8>> {
        Ok(ArrayAccess::from(array.is_inf()?))
    }
}

/// Element-type conversion; currently only `Cast<f64>` on f32 is supported.
#[derive(Clone, Copy, Debug)]
pub struct Cast<To>(PhantomData<To>);

impl sealed::Sealed for Cast<f64> {}

impl UnaryOp<f32> for Cast<f64> {
    type Output = f64;

    fn apply(&self, array: ArrayAccess<'static, f32>) -> Result<ArrayAccess<'static, f64>> {
        Ok(ArrayAccess::from(array.cast()?))
    }
}

/// A reusable unary expression with an immediate source and a typed operation.
///
/// Unary views implement read and composition traits, but not `TensorWrite`.
/// Unlike a geometric view, a computed result cannot be passed to a writer:
///
/// ```compile_fail,E0277
/// use fensor::{Tensor, TensorAbs, TensorCast, TensorFileEntry, TensorNumeric, TensorTransform, TensorTrig, TensorUnary, TensorWrite};
/// fn requires_write<V: TensorWrite>(_: &V) {}
/// async fn computed_is_read_only<FE: TensorFileEntry<f32>>(tensor: &Tensor<FE, f32>) {
///     let computed = tensor.view().abs().await.unwrap().cast().await.unwrap().cos().await.unwrap().exp().await.unwrap().is_nan().await.unwrap()
///         .transpose(None).unwrap();
///     requires_write(&computed);
/// }
/// ```
///
/// The geometric source remains writable without requiring `FE: Clone`:
///
/// ```
/// use fensor::{Tensor, TensorAbs, TensorCast, TensorFileEntry, TensorNumeric, TensorTransform, TensorTrig, TensorUnary, TensorWrite};
/// fn requires_write<V: TensorWrite>(_: &V) {}
/// async fn geometric_is_writable<FE: TensorFileEntry<f32>>(tensor: &Tensor<FE, f32>) {
///     let source = tensor.view().clone();
///     requires_write(&source);
///     let computed = source.abs().await.unwrap().cast().await.unwrap().sin().await.unwrap().round().await.unwrap()
///         .transpose(None).unwrap().clone();
///     let _cloned = computed.clone();
/// }
/// ```
#[derive(Clone)]
pub struct UnaryView<Source, Op> {
    source: Source,
    op: Op,
}

impl<S, O> UnaryView<S, O> {
    pub(crate) fn new(source: S, op: O) -> Self {
        Self { source, op }
    }
}

impl<E> TensorUnary for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    type ExpOutput = UnaryView<Self, Exp>;
    type LnOutput = UnaryView<Self, Ln>;
    type RoundOutput = UnaryView<Self, Round>;

    fn exp(&self) -> BoxFuture<'_, Result<Self::ExpOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Exp,
            })
        })
    }

    fn ln(&self) -> BoxFuture<'_, Result<Self::LnOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Ln,
            })
        })
    }

    fn round(&self) -> BoxFuture<'_, Result<Self::RoundOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Round,
            })
        })
    }
}

impl<E> TensorUnaryBoolean for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
{
    type Output = UnaryView<Self, Not>;

    fn not(&self) -> BoxFuture<'_, Result<Self::Output>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Not,
            })
        })
    }
}

impl<E> TensorNumeric for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float,
{
    type IsNanOutput = UnaryView<Self, IsNan>;
    type IsInfOutput = UnaryView<Self, IsInf>;

    fn is_nan(&self) -> BoxFuture<'_, Result<Self::IsNanOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: IsNan,
            })
        })
    }

    fn is_inf(&self) -> BoxFuture<'_, Result<Self::IsInfOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: IsInf,
            })
        })
    }
}

impl<E> TensorCast<f64> for E
where
    E: Expression<DType = f32> + Clone,
{
    type Output = UnaryView<Self, Cast<f64>>;

    fn cast(&self) -> BoxFuture<'_, Result<Self::Output>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Cast(PhantomData),
            })
        })
    }
}

impl<E> TensorAbs for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Number<Abs = E::DType>,
{
    type Output = UnaryView<Self, Abs>;

    fn abs(&self) -> BoxFuture<'_, Result<Self::Output>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Abs,
            })
        })
    }
}

impl<E> TensorTrig for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float,
{
    type SinOutput = UnaryView<Self, Sin>;
    type AsinOutput = UnaryView<Self, Asin>;
    type SinhOutput = UnaryView<Self, Sinh>;
    type CosOutput = UnaryView<Self, Cos>;
    type AcosOutput = UnaryView<Self, Acos>;
    type CoshOutput = UnaryView<Self, Cosh>;
    type TanOutput = UnaryView<Self, Tan>;
    type AtanOutput = UnaryView<Self, Atan>;
    type TanhOutput = UnaryView<Self, Tanh>;

    fn sin(&self) -> BoxFuture<'_, Result<Self::SinOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Sin,
            })
        })
    }

    fn asin(&self) -> BoxFuture<'_, Result<Self::AsinOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Asin,
            })
        })
    }

    fn sinh(&self) -> BoxFuture<'_, Result<Self::SinhOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Sinh,
            })
        })
    }

    fn cos(&self) -> BoxFuture<'_, Result<Self::CosOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Cos,
            })
        })
    }

    fn acos(&self) -> BoxFuture<'_, Result<Self::AcosOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Acos,
            })
        })
    }

    fn cosh(&self) -> BoxFuture<'_, Result<Self::CoshOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Cosh,
            })
        })
    }

    fn tan(&self) -> BoxFuture<'_, Result<Self::TanOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Tan,
            })
        })
    }

    fn atan(&self) -> BoxFuture<'_, Result<Self::AtanOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Atan,
            })
        })
    }

    fn tanh(&self) -> BoxFuture<'_, Result<Self::TanhOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Tanh,
            })
        })
    }
}

impl<S, O> TensorGeometry for UnaryView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    type DType = O::Output;

    fn dtype(&self) -> number_general::NumberType {
        <Self::DType as number_general::DType>::dtype()
    }
    fn layout(&self) -> Layout {
        self.source.layout()
    }
    fn shape(&self) -> &[usize] {
        self.source.shape()
    }
}

impl<S, O> TensorViewSemantics for UnaryView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn is_base_tensor(&self) -> bool {
        false
    }
    fn supports_write_through(&self) -> bool {
        false
    }
}

impl<S, O> TensorRead for UnaryView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move { Ok(expression::evaluate(self, &[coord.to_vec()]).await?[0]) })
    }

    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        let coords = crate::schema::row_major_coords(self.shape())?;

        Ok(expression::batches(self, coords)
            .map_ok(|(_, values)| values)
            .boxed())
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let coords = crate::traits::sparse_coords(self, range, requested_order)?;

            Ok(expression::batches(self, coords)
                .map_ok(|(coords, values)| {
                    futures::stream::iter(
                        coords
                            .into_iter()
                            .zip(values)
                            .filter(|(_, value)| *value != Self::DType::default())
                            .map(Ok),
                    )
                })
                .try_flatten()
                .boxed())
        })
    }
}

impl<S, O> TensorTransform for UnaryView<S, O>
where
    S: TensorTransform,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            source: self.source.reshape(shape)?,
            op: self.op,
        })
    }
    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            source: self.source.broadcast(shape)?,
            op: self.op,
        })
    }
    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            source: self.source.flip(axis)?,
            op: self.op,
        })
    }
    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            source: self.source.slice(range)?,
            op: self.op,
        })
    }
    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            source: self.source.squeeze(axes)?,
            op: self.op,
        })
    }
    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            source: self.source.transpose(permutation)?,
            op: self.op,
        })
    }
    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            source: self.source.unsqueeze(axes)?,
            op: self.op,
        })
    }
}

impl<S, O> Expression for UnaryView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn build<'a>(&'a self, coords: &'a [Vec<u64>]) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let source = self.source.build(coords).await?;
            Batch {
                array: self.op.apply(source.array)?,
                support: source.support,
            }
            .masked()
        })
    }
}
