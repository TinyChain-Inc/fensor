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

/// Lazy scalar add operation.
#[derive(Clone, Copy, Debug)]
pub struct AddScalar<T>(T);

impl<T> sealed::Sealed for AddScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for AddScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.add_scalar(self.0)?))
    }
}

/// Lazy scalar sub operation.
#[derive(Clone, Copy, Debug)]
pub struct SubScalar<T>(T);

impl<T> sealed::Sealed for SubScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for SubScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.sub_scalar(self.0)?))
    }
}

/// Lazy scalar mul operation.
#[derive(Clone, Copy, Debug)]
pub struct MulScalar<T>(T);

impl<T> sealed::Sealed for MulScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for MulScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.mul_scalar(self.0)?))
    }
}

/// Lazy scalar div operation.
#[derive(Clone, Copy, Debug)]
pub struct DivScalar<T>(T);

impl<T> sealed::Sealed for DivScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for DivScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.div_scalar(self.0)?))
    }
}

/// Lazy scalar pow operation.
#[derive(Clone, Copy, Debug)]
pub struct PowScalar<T>(T);

impl<T> sealed::Sealed for PowScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for PowScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.pow_scalar(self.0)?))
    }
}

/// Lazy scalar log operation.
#[derive(Clone, Copy, Debug)]
pub struct LogScalar<T>(T);

impl<T> sealed::Sealed for LogScalar<T> {}

impl<T: TensorElement + Float> UnaryOp<T> for LogScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.log_scalar(self.0)?))
    }
}

/// Lazy scalar rem operation.
#[derive(Clone, Copy, Debug)]
pub struct RemScalar<T>(T);

impl<T> sealed::Sealed for RemScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for RemScalar<T> {
    type Output = T;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.rem_scalar(self.0)?))
    }
}

impl<E> TensorMathScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    type AddOutput = UnaryView<Self, AddScalar<E::DType>>;

    type SubOutput = UnaryView<Self, SubScalar<E::DType>>;

    type MulOutput = UnaryView<Self, MulScalar<E::DType>>;

    type DivOutput = UnaryView<Self, DivScalar<E::DType>>;

    type PowOutput = UnaryView<Self, PowScalar<E::DType>>;

    type LogOutput
        = UnaryView<Self, LogScalar<E::DType>>
    where
        E::DType: Float;

    type RemOutput = UnaryView<Self, RemScalar<E::DType>>;

    fn add_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::AddOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), AddScalar(rhs))) })
    }

    fn sub_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::SubOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), SubScalar(rhs))) })
    }

    fn mul_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::MulOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), MulScalar(rhs))) })
    }

    fn div_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::DivOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), DivScalar(rhs))) })
    }

    fn pow_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::PowOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), PowScalar(rhs))) })
    }

    fn log_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::LogOutput>>
    where
        Self::DType: Float,
    {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), LogScalar(rhs))) })
    }

    fn rem_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::RemOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), RemScalar(rhs))) })
    }
}

/// Lazy scalar eq operation.
#[derive(Clone, Copy, Debug)]
pub struct EqScalar<T>(T);

impl<T> sealed::Sealed for EqScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for EqScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.eq_scalar(self.0)?))
    }
}

/// Lazy scalar ne operation.
#[derive(Clone, Copy, Debug)]
pub struct NeScalar<T>(T);

impl<T> sealed::Sealed for NeScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for NeScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.ne_scalar(self.0)?))
    }
}

/// Lazy scalar gt operation.
#[derive(Clone, Copy, Debug)]
pub struct GtScalar<T>(T);

impl<T> sealed::Sealed for GtScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for GtScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.gt_scalar(self.0)?))
    }
}

/// Lazy scalar ge operation.
#[derive(Clone, Copy, Debug)]
pub struct GeScalar<T>(T);

impl<T> sealed::Sealed for GeScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for GeScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.ge_scalar(self.0)?))
    }
}

/// Lazy scalar lt operation.
#[derive(Clone, Copy, Debug)]
pub struct LtScalar<T>(T);

impl<T> sealed::Sealed for LtScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for LtScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.lt_scalar(self.0)?))
    }
}

/// Lazy scalar le operation.
#[derive(Clone, Copy, Debug)]
pub struct LeScalar<T>(T);

impl<T> sealed::Sealed for LeScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for LeScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.le_scalar(self.0)?))
    }
}

impl<E> TensorCompareScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    type EqOutput = UnaryView<Self, EqScalar<E::DType>>;

    type NeOutput = UnaryView<Self, NeScalar<E::DType>>;

    type GtOutput = UnaryView<Self, GtScalar<E::DType>>;

    type GeOutput = UnaryView<Self, GeScalar<E::DType>>;

    type LtOutput = UnaryView<Self, LtScalar<E::DType>>;

    type LeOutput = UnaryView<Self, LeScalar<E::DType>>;

    fn eq_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::EqOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), EqScalar(rhs))) })
    }

    fn ne_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::NeOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), NeScalar(rhs))) })
    }

    fn gt_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::GtOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), GtScalar(rhs))) })
    }

    fn ge_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::GeOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), GeScalar(rhs))) })
    }

    fn lt_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::LtOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), LtScalar(rhs))) })
    }

    fn le_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::LeOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), LeScalar(rhs))) })
    }
}

/// Lazy scalar and operation.
#[derive(Clone, Copy, Debug)]
pub struct AndScalar<T>(T);

impl<T> sealed::Sealed for AndScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for AndScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.and_scalar(self.0)?))
    }
}

/// Lazy scalar or operation.
#[derive(Clone, Copy, Debug)]
pub struct OrScalar<T>(T);

impl<T> sealed::Sealed for OrScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for OrScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.or_scalar(self.0)?))
    }
}

/// Lazy scalar xor operation.
#[derive(Clone, Copy, Debug)]
pub struct XorScalar<T>(T);

impl<T> sealed::Sealed for XorScalar<T> {}

impl<T: TensorElement + Real> UnaryOp<T> for XorScalar<T> {
    type Output = u8;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, Self::Output>> {
        Ok(ArrayAccess::from(array.xor_scalar(self.0)?))
    }
}

impl<E> TensorBooleanScalar for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    type AndOutput = UnaryView<Self, AndScalar<E::DType>>;

    type OrOutput = UnaryView<Self, OrScalar<E::DType>>;

    type XorOutput = UnaryView<Self, XorScalar<E::DType>>;

    fn and_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::AndOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), AndScalar(rhs))) })
    }

    fn or_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::OrOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), OrScalar(rhs))) })
    }

    fn xor_scalar(&self, rhs: Self::DType) -> BoxFuture<'_, Result<Self::XorOutput>> {
        Box::pin(async move { Ok(UnaryView::new(self.clone(), XorScalar(rhs))) })
    }
}
