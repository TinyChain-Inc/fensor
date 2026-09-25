//! Typed, read-only unary views over tensor expressions.

use std::marker::PhantomData;

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{
    ArrayAccess, NDArrayAbs, NDArrayCast, NDArrayNumeric, NDArrayTrig, NDArrayUnary,
    NDArrayUnaryBoolean,
};

use crate::expression::{self, Batch, Expression};
use crate::request::{self, BatchRequest};
use crate::schema::Layout;
use crate::tensor::TensorElement;
use crate::traits::{
    BoxFuture, SparseElementStream, TensorAbs, TensorCast, TensorGeometry, TensorNumeric,
    TensorRead, TensorTransform, TensorTrig, TensorUnary, TensorUnaryBoolean, TensorViewSemantics,
    ValueBlockStream,
};
use crate::{Axes, Range, Result, Shape};

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

// Each declaration preserves its explicit dtype bounds and backend operation.
macro_rules! unary_op {
    ($(#[$doc:meta])* $name:ident, $method:ident, $ty:ident: [$($bounds:tt)+] => $output:ty) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl<$ty: $($bounds)+> UnaryOp<$ty> for $name {
            type Output = $output;

            fn apply(
                &self,
                array: ArrayAccess<'static, $ty>,
            ) -> Result<ArrayAccess<'static, Self::Output>> {
                Ok(ArrayAccess::from(array.$method()?))
            }
        }
    };
}

unary_op!(
    /// Elementwise exponentiation.
    Exp, exp, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise natural logarithm.
    Ln, ln, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise rounding.
    Round, round, T: [TensorElement + ha_ndarray::Float + ha_ndarray::Real] => T
);

unary_op!(
    /// Elementwise absolute value.
    Abs, abs, T: [TensorElement + ha_ndarray::Number<Abs = T>] => T
);

unary_op!(
    /// Elementwise sine.
    Sin, sin, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise inverse sine.
    Asin, asin, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise hyperbolic sine.
    Sinh, sinh, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise cosine.
    Cos, cos, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise inverse cosine.
    Acos, acos, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise hyperbolic cosine.
    Cosh, cosh, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise tangent.
    Tan, tan, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise inverse tangent.
    Atan, atan, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise hyperbolic tangent.
    Tanh, tanh, T: [TensorElement + ha_ndarray::Float] => T
);

unary_op!(
    /// Elementwise logical negation, returning 0 or 1.
    Not, not, T: [TensorElement] => u8
);

unary_op!(
    /// Elementwise NaN detection, returning 0 or 1.
    IsNan, is_nan, T: [TensorElement + ha_ndarray::Float] => u8
);

unary_op!(
    /// Elementwise infinity detection, returning 0 or 1.
    IsInf, is_inf, T: [TensorElement + ha_ndarray::Float] => u8
);

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

// Expand only members of the explicit public-trait implementation below.
macro_rules! unary_constructor {
    ($output:ident, $method:ident, $op:ident) => {
        type $output = UnaryView<Self, $op>;

        fn $method(&self) -> BoxFuture<'_, Result<Self::$output>> {
            Box::pin(async move { Ok(UnaryView::new(self.clone(), $op)) })
        }
    };
}

impl<E> TensorUnary for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    unary_constructor!(ExpOutput, exp, Exp);

    unary_constructor!(LnOutput, ln, Ln);

    unary_constructor!(RoundOutput, round, Round);
}

impl<E> TensorUnaryBoolean for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
{
    unary_constructor!(Output, not, Not);
}

impl<E> TensorNumeric for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float,
{
    unary_constructor!(IsNanOutput, is_nan, IsNan);

    unary_constructor!(IsInfOutput, is_inf, IsInf);
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
    unary_constructor!(Output, abs, Abs);
}

impl<E> TensorTrig for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float,
{
    unary_constructor!(SinOutput, sin, Sin);

    unary_constructor!(AsinOutput, asin, Asin);

    unary_constructor!(SinhOutput, sinh, Sinh);

    unary_constructor!(CosOutput, cos, Cos);

    unary_constructor!(AcosOutput, acos, Acos);

    unary_constructor!(CoshOutput, cosh, Cosh);

    unary_constructor!(TanOutput, tan, Tan);

    unary_constructor!(AtanOutput, atan, Atan);

    unary_constructor!(TanhOutput, tanh, Tanh);
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

    fn shape(&self) -> &[u64] {
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
    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, Self::DType>> {
        expression::coordinate_blocks(self)
    }

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Ok(
                expression::evaluate_batch(self, &BatchRequest::point(coord))
                    .await?
                    .values[0],
            )
        })
    }

    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        let coords = request::linear_requests(self.shape())?;

        Ok(expression::ordered_batches(self, coords)
            .map_ok(|(_, batch)| batch.values)
            .boxed())
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let coords = crate::traits::sparse_coords(self, range, requested_order)?;

            Ok(
                expression::ordered_batches(self, request::explicit_requests(coords))
                    .and_then(move |(coords, values)| async move {
                        Ok(futures::stream::iter(expression::sparse_elements(
                            coords,
                            values,
                            self.shape(),
                        )?))
                    })
                    .try_flatten()
                    .boxed(),
            )
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
    fn slice_requests(&self, slice: crate::slice::Slice) -> Result<crate::slice::Requests<'_>> {
        self.source.slice_requests(slice)
    }

    fn preferred_requests(&self, shape: &[u64]) -> Result<Option<expression::RequestIterator>> {
        self.source.preferred_requests(shape)
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
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
