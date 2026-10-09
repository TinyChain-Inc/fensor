//! Typed, read-only unary views over tensor expressions.

use std::marker::PhantomData;

#[cfg(feature = "complex")]
use ha_ndarray::NDArrayComplex;
use ha_ndarray::{
    ArrayAccess, NDArrayAbs, NDArrayCast, NDArrayNumeric, NDArrayRead, NDArrayTrig, NDArrayUnary,
    NDArrayUnaryBoolean, Number,
};

use crate::expression::{self, Batch, Expression};
use crate::request::BatchRequest;
use crate::schema::Layout;
use crate::tensor::TensorElement;
use crate::traits::{
    BoxFuture, TensorAbs, TensorCast, TensorGeometry, TensorNumeric, TensorRead, TensorTransform,
    TensorTrig, TensorUnary, TensorUnaryBoolean, TensorViewSemantics,
};
use crate::{Error, Result};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// An operation which builds an ndarray expression without evaluating it.
///
/// This trait is sealed: only the supported unary operations can be used
/// in a `UnaryView`.
pub trait UnaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type Output: TensorElement;

    const NAME: &'static str;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>>;
}

// Each declaration preserves its explicit dtype bounds and backend operation.
macro_rules! unary_op {
    ($(#[$doc:meta])* $name:ident, $method:ident, $ty:ident: [$($bounds:tt)+] => $output:ty) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl<$ty: $($bounds)+> UnaryOp<$ty> for $name
        where $output: TensorElement,
        {
            type Output = $output;

            const NAME: &'static str = stringify!($method);

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
    /// Elementwise absolute value, returning real magnitudes for complex inputs.
    Abs, abs, T: [TensorElement] => T::Abs
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

/// Element-type conversion using the backend's number-general cast pipeline.
#[derive(Clone, Copy, Debug)]
pub struct Cast<To>(PhantomData<To>);

impl<To: TensorElement> sealed::Sealed for Cast<To> {}

impl<From: TensorElement, To: TensorElement> UnaryOp<From> for Cast<To> {
    type Output = To;

    const NAME: &'static str = "cast";

    fn apply(&self, array: ArrayAccess<'static, From>) -> Result<ArrayAccess<'static, To>> {
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
///     let cast = TensorCast::<f64>::cast(&tensor.view().abs().await.unwrap()).await.unwrap();
///     let computed = cast.cos().await.unwrap().exp().await.unwrap().is_nan().await.unwrap()
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
///     let cast = TensorCast::<f64>::cast(&source.abs().await.unwrap()).await.unwrap();
///     let computed = cast.sin().await.unwrap().round().await.unwrap()
///         .transpose(None).unwrap().clone();
///     let _cloned = computed.clone();
/// }
/// ```
#[derive(Clone)]
pub struct UnaryView<Source, Op>
where
    Source: TensorGeometry,
    Source::DType: TensorElement,
    Op: UnaryOp<Source::DType>,
{
    source: Source,
    op: Op,
    zero: Op::Output,
}

impl<S, O> UnaryView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    pub(crate) fn new(source: S, op: O) -> Result<Self> {
        let zero = if matches!(source.layout(), Layout::Sparse { .. }) {
            let input = expression::batch_array(vec![source.implicit_zero()])?;
            let zero = op.apply(input)?.read_value(&[0])?;
            if zero != O::Output::ZERO {
                return Err(Error::WouldDensify { operation: O::NAME });
            }
            zero
        } else {
            O::Output::ZERO
        };
        Ok(Self { source, op, zero })
    }
}

// Expand only members of the explicit public-trait implementation below.
macro_rules! unary_constructor {
    ($output:ident, $method:ident, $op:ident $(; where [$($bounds:tt)+])?) => {
        type $output = UnaryView<Self, $op> $(where $($bounds)+)?;

        fn $method(&self) -> BoxFuture<'_, Result<Self::$output>>
        $(where $($bounds)+)?
        {
            Box::pin(async move { UnaryView::new(self.clone(), $op) })
        }
    };
}

impl<E> TensorUnary for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Float,
{
    unary_constructor!(ExpOutput, exp, Exp);

    unary_constructor!(LnOutput, ln, Ln);

    unary_constructor!(RoundOutput, round, Round; where [E::DType: ha_ndarray::Real]);
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

impl<E, To> TensorCast<To> for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
    To: TensorElement,
{
    type Output = UnaryView<Self, Cast<To>>;

    fn cast(&self) -> BoxFuture<'_, Result<Self::Output>> {
        Box::pin(async move { UnaryView::new(self.clone(), Cast(PhantomData)) })
    }
}

impl<E> TensorAbs for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
    <E::DType as ha_ndarray::Number>::Abs: TensorElement,
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
}

impl<S, O> TensorRead for UnaryView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<S, O> TensorTransform for UnaryView<S, O>
where
    S: TensorTransform,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    crate::mapping::transform_methods!(operands: source; preserve_rest);
}

impl<S, O> Expression for UnaryView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn implicit_zero(&self) -> Self::DType {
        self.zero
    }

    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([self.source.expression_nodes()?])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.source.detach_sources(pending);
    }

    fn selection_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Selection<'_>> {
        Ok(expression::traversal::Selection::Source(Box::new(
            move || self.source.selection_step(slice),
        )))
    }

    fn ordered_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Ordered<'_>> {
        Ok(expression::traversal::Ordered::Sources(vec![Box::new(
            move || self.source.ordered_step(slice),
        )]))
    }

    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        Ok(expression::traversal::Preferred::Sources(vec![Box::new(
            move || self.source.preferred_step(shape),
        )]))
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let source = context.batch(&self.source, coords).await?;
            Batch {
                _allocation: None,
                array: self.op.apply(source.array)?,
            }
            .realize()
        })
    }
}

#[cfg(feature = "complex")]
unary_op!(
    /// Complex conjugation.
    Conj, conj, T: [TensorElement + ha_ndarray::Complex] => T
);

#[cfg(feature = "complex")]
unary_op!(
    /// Real component of a complex value.
    Re, re, T: [TensorElement + ha_ndarray::Complex] => T::Real
);

#[cfg(feature = "complex")]
unary_op!(
    /// Imaginary component of a complex value.
    Im, im, T: [TensorElement + ha_ndarray::Complex] => T::Real
);

#[cfg(feature = "complex")]
unary_op!(
    /// Principal complex argument in radians.
    Angle, angle, T: [TensorElement + ha_ndarray::Complex] => T::Real
);

#[cfg(feature = "complex")]
impl<E> crate::TensorComplex for E
where
    E: Expression + Clone,
    E::DType: TensorElement + ha_ndarray::Complex,
    <E::DType as ha_ndarray::Complex>::Real: TensorElement,
{
    unary_constructor!(ConjOutput, conj, Conj);

    unary_constructor!(ReOutput, re, Re);

    unary_constructor!(ImOutput, im, Im);

    unary_constructor!(AngleOutput, angle, Angle);
}
