//! Typed, read-only elementwise expressions over two sources.

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{ArrayAccess, Float, NDArrayBoolean, NDArrayCompare, NDArrayMath, Real};

use crate::expression::{self, Batch, Expression};
use crate::request::{self, BatchRequest};
use crate::traits::{BoxFuture, SparseElementStream, ValueBlockStream};
use crate::{
    Axes, Error, Layout, Range, Result, Shape, TensorBoolean, TensorCompare, TensorElement,
    TensorGeometry, TensorMath, TensorRead, TensorTransform, TensorViewSemantics,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed operation building an ndarray expression without evaluating it.
pub trait BinaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type Output: TensorElement;

    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, Self::Output>>;
}

/// A read-only expression retaining both source descriptions.
///
/// Operands can use distinct file-entry adapters; cloning descriptions does not
/// require either adapter to implement `Clone`.
///
/// ```
/// use fensor::{Tensor, TensorFileEntry, TensorMath, TensorTransform, TensorUnary};
/// async fn clone_description<A: TensorFileEntry<f32>, B: TensorFileEntry<f32>>(
///     left: &Tensor<A, f32>, right: &Tensor<B, f32>,
/// ) -> fensor::Result<()> {
///     let view = left.view().add(&right.view()).await?.exp().await?
///         .transpose(None)?.clone();
///     let _ = view.clone();
///     Ok(())
/// }
/// ```
///
/// Binary expressions cannot be written through or used as persistent arrays.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMath, TensorTransform, TensorWrite};
/// fn writable<T: TensorWrite>(_: &T) {}
/// async fn test<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     writable(&a.view().add(&a.view()).await.unwrap().transpose(None).unwrap());
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorArray, TensorFileEntry, TensorMath};
/// fn persistent<T: TensorArray>(_: &T) {}
/// async fn test<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     persistent(&a.view().add(&a.view()).await.unwrap());
/// }
/// ```
///
/// Operands require matching dtypes; casts must be explicit.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMath};
/// async fn test<F: TensorFileEntry<f32> + TensorFileEntry<f64>>(
///     a: &Tensor<F, f32>, b: &Tensor<F, f64>,
/// ) { let _ = a.view().add(&b.view()).await; }
/// ```
///
/// Logarithms require floating-point operands.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMath};
/// async fn test<F: TensorFileEntry<u8>>(a: &Tensor<F, u8>) {
///     let _ = a.view().log(&a.view()).await;
/// }
/// ```
///
/// Comparison and logical results are also read-only.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorCompare, TensorBooleanScalar, TensorWrite};
/// fn writable<T: TensorWrite>(_: &T) {}
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     writable(&a.view().eq(&a.view()).await.unwrap().or_scalar(1).await.unwrap());
/// }
/// ```
///
/// Comparison and logical operands require identical dtypes.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorCompare};
/// async fn example<F: TensorFileEntry<f32> + TensorFileEntry<f64>>(
///     a: &Tensor<F, f32>, b: &Tensor<F, f64>,
/// ) { let _ = a.view().eq(&b.view()).await; }
/// ```
#[derive(Clone)]
pub struct BinaryView<Left, Right, Op> {
    left: Left,
    right: Right,
    op: Op,
}

impl<L: crate::TensorGeometry, R: crate::TensorGeometry, O> BinaryView<L, R, O> {
    fn new(left: L, right: R, op: O) -> Result<Self> {
        if left.shape() != right.shape() {
            return Err(Error::InvalidSchema(format!(
                "binary operand shapes differ: {:?} and {:?}",
                left.shape(),
                right.shape()
            )));
        }

        Ok(Self { left, right, op })
    }
}

// Each declaration preserves its explicit dtype bounds and backend operation.
macro_rules! binary_op {
    ($(#[$doc:meta])* $name:ident, $method:ident, $ty:ident: [$($bounds:tt)+] => $output:ty) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl<$ty: $($bounds)+> BinaryOp<$ty> for $name {
            type Output = $output;

            fn apply(
                &self,
                left: ArrayAccess<'static, $ty>,
                right: ArrayAccess<'static, $ty>,
            ) -> Result<ArrayAccess<'static, Self::Output>> {
                Ok(ArrayAccess::from(left.$method(right)?))
            }
        }
    };
}

binary_op!(
    /// Elementwise add.
    Add, add, T: [TensorElement] => T
);

binary_op!(
    /// Elementwise sub.
    Sub, sub, T: [TensorElement] => T
);

binary_op!(
    /// Elementwise mul.
    Mul, mul, T: [TensorElement] => T
);

binary_op!(
    /// Elementwise div.
    Div, div, T: [TensorElement] => T
);

binary_op!(
    /// Elementwise pow.
    Pow, pow, T: [TensorElement] => T
);

binary_op!(
    /// Elementwise log.
    Log, log, T: [TensorElement + Float] => T
);

binary_op!(
    /// Elementwise rem.
    Rem, rem, T: [TensorElement + Real] => T
);

// Expand only members of the explicit public-trait implementation below.
macro_rules! binary_constructor {
    ($output:ident, $method:ident, $op:ident, $rhs:ty) => {
        type $output = BinaryView<Self, $rhs, $op>;

        fn $method<'a>(&'a self, rhs: &'a $rhs) -> BoxFuture<'a, Result<Self::$output>> {
            Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), $op) })
        }
    };
}

impl<L, R> TensorMath<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement + Real,
{
    binary_constructor!(AddOutput, add, Add, R);

    binary_constructor!(SubOutput, sub, Sub, R);

    binary_constructor!(MulOutput, mul, Mul, R);

    binary_constructor!(DivOutput, div, Div, R);

    binary_constructor!(PowOutput, pow, Pow, R);

    type LogOutput
        = BinaryView<Self, R, Log>
    where
        L::DType: Float;

    binary_constructor!(RemOutput, rem, Rem, R);

    fn log<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::LogOutput>>
    where
        L::DType: Float,
    {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Log) })
    }
}

impl<L, R, O> TensorGeometry for BinaryView<L, R, O>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    type DType = O::Output;

    fn dtype(&self) -> crate::NumberType {
        <Self::DType as number_general::DType>::dtype()
    }

    fn shape(&self) -> &[u64] {
        self.left.shape()
    }

    fn layout(&self) -> Layout {
        match (self.left.layout(), self.right.layout()) {
            (Layout::Sparse { .. }, Layout::Sparse { .. }) => Layout::Sparse { axis: None },
            _ => Layout::Dense,
        }
    }
}

impl<L, R, O> TensorViewSemantics for BinaryView<L, R, O>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    fn is_base_tensor(&self) -> bool {
        false
    }

    fn supports_write_through(&self) -> bool {
        false
    }
}

impl<L, R, O> Expression for BinaryView<L, R, O>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    fn preferred_requests(&self, shape: &[u64]) -> Result<Option<expression::RequestIterator>> {
        match self.left.preferred_requests(shape)? {
            Some(requests) => Ok(Some(requests)),
            None => self.right.preferred_requests(shape),
        }
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let left = self.left.build(coords).await?;
            let right = self.right.build(coords).await?;
            let support = expression::union_support(left.support, right.support)?;

            Batch {
                array: self.op.apply(left.array, right.array)?,
                support,
            }
            .masked()
        })
    }
}

impl<L, R, O> TensorRead for BinaryView<L, R, O>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
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

impl<L, R, O> TensorTransform for BinaryView<L, R, O>
where
    L: TensorTransform,
    R: TensorTransform<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            left: self.left.reshape(shape.clone())?,
            right: self.right.reshape(shape)?,
            op: self.op,
        })
    }

    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            left: self.left.broadcast(shape.clone())?,
            right: self.right.broadcast(shape)?,
            op: self.op,
        })
    }

    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            left: self.left.flip(axis)?,
            right: self.right.flip(axis)?,
            op: self.op,
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            left: self.left.slice(range.clone())?,
            right: self.right.slice(range)?,
            op: self.op,
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            left: self.left.squeeze(axes.clone())?,
            right: self.right.squeeze(axes)?,
            op: self.op,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            left: self.left.transpose(permutation.clone())?,
            right: self.right.transpose(permutation)?,
            op: self.op,
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            left: self.left.unsqueeze(axes.clone())?,
            right: self.right.unsqueeze(axes)?,
            op: self.op,
        })
    }
}

binary_op!(
    /// Elementwise eq operation returning zero or one.
    Eq, eq, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise ne operation returning zero or one.
    Ne, ne, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise gt operation returning zero or one.
    Gt, gt, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise ge operation returning zero or one.
    Ge, ge, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise lt operation returning zero or one.
    Lt, lt, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise le operation returning zero or one.
    Le, le, T: [TensorElement + Real] => u8
);

impl<L, R> TensorCompare<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement + Real,
{
    binary_constructor!(EqOutput, eq, Eq, R);

    binary_constructor!(NeOutput, ne, Ne, R);

    binary_constructor!(GtOutput, gt, Gt, R);

    binary_constructor!(GeOutput, ge, Ge, R);

    binary_constructor!(LtOutput, lt, Lt, R);

    binary_constructor!(LeOutput, le, Le, R);
}

binary_op!(
    /// Elementwise and operation returning zero or one.
    And, and, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise or operation returning zero or one.
    Or, or, T: [TensorElement + Real] => u8
);

binary_op!(
    /// Elementwise xor operation returning zero or one.
    Xor, xor, T: [TensorElement + Real] => u8
);

impl<L, R> TensorBoolean<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement + Real,
{
    binary_constructor!(AndOutput, and, And, R);

    binary_constructor!(OrOutput, or, Or, R);

    binary_constructor!(XorOutput, xor, Xor, R);
}
