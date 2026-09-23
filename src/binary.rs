//! Typed, read-only elementwise expressions over two sources.
use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{ArrayAccess, Axes, Float, NDArrayMath, Range, Real, Shape};

use crate::expression::{self, Batch, Expression};
use crate::traits::{BoxFuture, SparseElementStream, ValueBlockStream};
use crate::{
    Error, Layout, Result, TensorElement, TensorGeometry, TensorMath, TensorRead, TensorTransform,
    TensorViewSemantics,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed operation building an ndarray expression without evaluating it.
pub trait BinaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>>;
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

/// Elementwise add.
#[derive(Clone, Copy, Debug)]
pub struct Add;

impl sealed::Sealed for Add {}

impl<T: TensorElement> BinaryOp<T> for Add {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.add(right)?))
    }
}

/// Elementwise sub.
#[derive(Clone, Copy, Debug)]
pub struct Sub;

impl sealed::Sealed for Sub {}

impl<T: TensorElement> BinaryOp<T> for Sub {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.sub(right)?))
    }
}

/// Elementwise mul.
#[derive(Clone, Copy, Debug)]
pub struct Mul;

impl sealed::Sealed for Mul {}

impl<T: TensorElement> BinaryOp<T> for Mul {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.mul(right)?))
    }
}

/// Elementwise div.
#[derive(Clone, Copy, Debug)]
pub struct Div;

impl sealed::Sealed for Div {}

impl<T: TensorElement> BinaryOp<T> for Div {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.div(right)?))
    }
}

/// Elementwise pow.
#[derive(Clone, Copy, Debug)]
pub struct Pow;

impl sealed::Sealed for Pow {}

impl<T: TensorElement> BinaryOp<T> for Pow {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.pow(right)?))
    }
}

/// Elementwise log.
#[derive(Clone, Copy, Debug)]
pub struct Log;

impl sealed::Sealed for Log {}

impl<T: TensorElement + Float> BinaryOp<T> for Log {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.log(right)?))
    }
}

/// Elementwise rem.
#[derive(Clone, Copy, Debug)]
pub struct Rem;

impl sealed::Sealed for Rem {}

impl<T: TensorElement + Real> BinaryOp<T> for Rem {
    fn apply(
        &self,
        left: ArrayAccess<'static, T>,
        right: ArrayAccess<'static, T>,
    ) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(left.rem(right)?))
    }
}

impl<L, R> TensorMath<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement + Real,
{
    type AddOutput = BinaryView<Self, R, Add>;
    type SubOutput = BinaryView<Self, R, Sub>;
    type MulOutput = BinaryView<Self, R, Mul>;
    type DivOutput = BinaryView<Self, R, Div>;
    type PowOutput = BinaryView<Self, R, Pow>;
    type LogOutput
        = BinaryView<Self, R, Log>
    where
        L::DType: Float;
    type RemOutput = BinaryView<Self, R, Rem>;

    fn add<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::AddOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Add) })
    }

    fn sub<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::SubOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Sub) })
    }

    fn mul<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::MulOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Mul) })
    }

    fn div<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::DivOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Div) })
    }

    fn pow<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::PowOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Pow) })
    }

    fn log<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::LogOutput>>
    where
        L::DType: Float,
    {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Log) })
    }

    fn rem<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::RemOutput>> {
        Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), Rem) })
    }
}

impl<L, R, O> TensorGeometry for BinaryView<L, R, O>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    type DType = L::DType;

    fn dtype(&self) -> crate::NumberType {
        self.left.dtype()
    }

    fn shape(&self) -> &[usize] {
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
    fn build<'a>(&'a self, coords: &'a [Vec<u64>]) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let left = self.left.build(coords).await?;
            let right = self.right.build(coords).await?;
            let support = match (left.support, right.support) {
                (Some(left), Some(right)) => {
                    Some(left.into_iter().zip(right).map(|(l, r)| l | r).collect())
                }
                _ => None,
            };

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
