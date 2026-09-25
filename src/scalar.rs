//! Scalar parameters attached to lazy unary expressions.
//!
//! Scalar operands preserve the source's support; they never populate implicit
//! sparse zeros. The operation parameter has the source dtype.
//!
//! Computed scalar views remain read-only after transforms.
//!
//! ```compile_fail
//! use fensor::{Tensor, TensorFileEntry, TensorMathScalar, TensorTransform, TensorWrite};
//! fn writable<T: TensorWrite>(_: &T) {}
//! async fn example<F: TensorFileEntry<f32>>(tensor: &Tensor<F, f32>) {
//!     writable(&tensor.view().add_scalar(1.).await.unwrap().flip(0).unwrap());
//! }
//! ```
//!
//! Scalar logarithms require floating-point data.
//!
//! ```compile_fail
//! use fensor::{Tensor, TensorFileEntry, TensorMathScalar};
//! async fn example<F: TensorFileEntry<u8>>(tensor: &Tensor<F, u8>) {
//!     let _ = tensor.view().log_scalar(2).await;
//! }
//! ```

use ha_ndarray::{
    ArrayAccess, Float, NDArrayBooleanScalar, NDArrayCompareScalar, NDArrayMathScalar, Real,
};

use crate::expression::Expression;
use crate::unary::{UnaryOp, sealed};
use crate::{
    BoxFuture, Result, TensorBooleanScalar, TensorCompareScalar, TensorElement, TensorMathScalar,
    UnaryView,
};

// Each declaration preserves its explicit dtype bounds and backend operation.
macro_rules! scalar_op {
    ($(#[$doc:meta])* $name:ident, $method:ident, $ty:ident: [$($bounds:tt)+] => $output:ty) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name<$ty>($ty);

        impl<$ty> sealed::Sealed for $name<$ty> {}

        impl<$ty: $($bounds)+> UnaryOp<$ty> for $name<$ty> {
            type Output = $output;

            fn apply(
                &self,
                array: ArrayAccess<'static, $ty>,
            ) -> Result<ArrayAccess<'static, Self::Output>> {
                Ok(ArrayAccess::from(array.$method(self.0)?))
            }
        }
    };
}

scalar_op!(
    /// Lazy scalar add operation.
    AddScalar, add_scalar, T: [TensorElement + Real] => T
);

scalar_op!(
    /// Lazy scalar sub operation.
    SubScalar, sub_scalar, T: [TensorElement + Real] => T
);

scalar_op!(
    /// Lazy scalar mul operation.
    MulScalar, mul_scalar, T: [TensorElement + Real] => T
);

scalar_op!(
    /// Lazy scalar div operation.
    DivScalar, div_scalar, T: [TensorElement + Real] => T
);

scalar_op!(
    /// Lazy scalar pow operation.
    PowScalar, pow_scalar, T: [TensorElement + Real] => T
);

scalar_op!(
    /// Lazy scalar log operation.
    LogScalar, log_scalar, T: [TensorElement + Float] => T
);

scalar_op!(
    /// Lazy scalar rem operation.
    RemScalar, rem_scalar, T: [TensorElement + Real] => T
);

// Expand only members of the explicit public-trait implementation below.
macro_rules! scalar_constructor {
    ($output:ident, $method:ident, $op:ident) => {
        type $output = UnaryView<Self, $op<Self::DType>>;

        fn $method(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::$output>> {
            Box::pin(async move { Ok(UnaryView::new(self.clone(), $op(rhs))) })
        }
    };
}

impl<E> TensorMathScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    scalar_constructor!(AddOutput, add_scalar, AddScalar);

    scalar_constructor!(SubOutput, sub_scalar, SubScalar);

    scalar_constructor!(MulOutput, mul_scalar, MulScalar);

    scalar_constructor!(DivOutput, div_scalar, DivScalar);

    scalar_constructor!(PowOutput, pow_scalar, PowScalar);

    type LogOutput
        = UnaryView<Self, LogScalar<E::DType>>
    where
        E::DType: Float;

    scalar_constructor!(RemOutput, rem_scalar, RemScalar);

    fn log_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::LogOutput>>
    where
        Self::DType: Float,
    {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), LogScalar(rhs))) })
    }
}

scalar_op!(
    /// Lazy scalar eq operation.
    EqScalar, eq_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar ne operation.
    NeScalar, ne_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar gt operation.
    GtScalar, gt_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar ge operation.
    GeScalar, ge_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar lt operation.
    LtScalar, lt_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar le operation.
    LeScalar, le_scalar, T: [TensorElement + Real] => u8
);

impl<E> TensorCompareScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    scalar_constructor!(EqOutput, eq_scalar, EqScalar);

    scalar_constructor!(NeOutput, ne_scalar, NeScalar);

    scalar_constructor!(GtOutput, gt_scalar, GtScalar);

    scalar_constructor!(GeOutput, ge_scalar, GeScalar);

    scalar_constructor!(LtOutput, lt_scalar, LtScalar);

    scalar_constructor!(LeOutput, le_scalar, LeScalar);
}

scalar_op!(
    /// Lazy scalar and operation.
    AndScalar, and_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar or operation.
    OrScalar, or_scalar, T: [TensorElement + Real] => u8
);

scalar_op!(
    /// Lazy scalar xor operation.
    XorScalar, xor_scalar, T: [TensorElement + Real] => u8
);

impl<E> TensorBooleanScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    scalar_constructor!(AndOutput, and_scalar, AndScalar);

    scalar_constructor!(OrOutput, or_scalar, OrScalar);

    scalar_constructor!(XorOutput, xor_scalar, XorScalar);
}
