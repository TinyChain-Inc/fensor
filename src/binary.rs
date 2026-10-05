//! Typed, read-only elementwise expressions over two sources.

use ha_ndarray::{
    ArrayAccess, Float, NDArrayBoolean, NDArrayCompare, NDArrayMath, NDArrayRead, Number, Real,
};

use crate::expression::{self, Batch, Expression};
use crate::request::BatchRequest;
use crate::traits::BoxFuture;
use crate::{
    Error, Layout, Result, TensorBoolean, TensorCompare, TensorElement, TensorGeometry, TensorMath,
    TensorRead, TensorTransform, TensorViewSemantics,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed operation building an ndarray expression without evaluating it.
pub trait BinaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type Output: TensorElement;

    const NAME: &'static str;

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
pub struct BinaryView<Left, Right, Op>
where
    Left: TensorGeometry,
    Left::DType: TensorElement,
    Op: BinaryOp<Left::DType>,
{
    left: Left,
    right: Right,
    op: Op,
    zero: Op::Output,
}

impl<L, R, O> BinaryView<L, R, O>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    fn new(left: L, right: R, op: O) -> Result<Self> {
        if left.shape() != right.shape() {
            return Err(Error::InvalidSchema(format!(
                "binary operand shapes differ: {:?} and {:?}",
                left.shape(),
                right.shape()
            )));
        }

        let zero = if matches!(left.layout(), Layout::Sparse { .. })
            && matches!(right.layout(), Layout::Sparse { .. })
        {
            let left_zero = expression::batch_array(vec![left.implicit_zero()])?;
            let right_zero = expression::batch_array(vec![right.implicit_zero()])?;
            let zero = op.apply(left_zero, right_zero)?.read_value(&[0])?;
            if zero != O::Output::ZERO {
                return Err(Error::WouldDensify { operation: O::NAME });
            }
            zero
        } else {
            O::Output::ZERO
        };
        Ok(Self {
            left,
            right,
            op,
            zero,
        })
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

            const NAME: &'static str = stringify!($method);

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
    ($output:ident, $method:ident, $op:ident, $rhs:ty $(; where [$($bounds:tt)+])?) => {
        type $output = BinaryView<Self, $rhs, $op> $(where $($bounds)+)?;

        fn $method<'a>(&'a self, rhs: &'a $rhs) -> BoxFuture<'a, Result<Self::$output>>
        $(where $($bounds)+)?
        {
            Box::pin(async move { BinaryView::new(self.clone(), rhs.clone(), $op) })
        }
    };
}

impl<L, R> TensorMath<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement,
{
    binary_constructor!(AddOutput, add, Add, R);

    binary_constructor!(SubOutput, sub, Sub, R);

    binary_constructor!(MulOutput, mul, Mul, R);

    binary_constructor!(DivOutput, div, Div, R);

    binary_constructor!(PowOutput, pow, Pow, R);

    binary_constructor!(LogOutput, log, Log, R; where [L::DType: Float]);

    binary_constructor!(RemOutput, rem, Rem, R; where [L::DType: Real]);
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
}

impl<L, R, O> Expression for BinaryView<L, R, O>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    fn implicit_zero(&self) -> Self::DType {
        self.zero
    }

    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([
            self.left.expression_nodes()?,
            self.right.expression_nodes()?,
        ])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.left.detach_sources(pending);
        self.right.detach_sources(pending);
    }

    fn selection_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Selection<'_>> {
        expression::traversal::ordered(self, slice).map(expression::traversal::Selection::Ready)
    }

    fn ordered_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Ordered<'_>> {
        if matches!(self.layout(), Layout::Dense) {
            return Ok(expression::traversal::Ordered::Ready(slice.stream()));
        }

        let left_slice = slice.clone();
        Ok(expression::traversal::Ordered::Sources(vec![
            (0, Box::new(move || self.left.ordered_step(left_slice))),
            (1, Box::new(move || self.right.ordered_step(slice))),
        ]))
    }

    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        Ok(expression::traversal::Preferred::Sources(vec![
            Box::new(move || self.left.preferred_step(shape)),
            Box::new(move || self.right.preferred_step(shape)),
        ]))
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let left = context.batch(&self.left, coords.clone()).await?;
            let right = context.batch(&self.right, coords.clone()).await?;

            Batch {
                _allocation: None,
                array: self.op.apply(left.array, right.array)?,
            }
            .realize()
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
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<L, R, O> TensorTransform for BinaryView<L, R, O>
where
    L: TensorTransform,
    R: TensorTransform<DType = L::DType>,
    L::DType: TensorElement,
    O: BinaryOp<L::DType>,
{
    crate::mapping::transform_methods!(operands: left, right; preserve_rest);
}

binary_op!(
    /// Elementwise eq operation returning zero or one.
    Eq, eq, T: [TensorElement] => u8
);

binary_op!(
    /// Elementwise ne operation returning zero or one.
    Ne, ne, T: [TensorElement] => u8
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
    L::DType: TensorElement,
{
    binary_constructor!(EqOutput, eq, Eq, R);

    binary_constructor!(NeOutput, ne, Ne, R);

    binary_constructor!(GtOutput, gt, Gt, R; where [L::DType: Real]);

    binary_constructor!(GeOutput, ge, Ge, R; where [L::DType: Real]);

    binary_constructor!(LtOutput, lt, Lt, R; where [L::DType: Real]);

    binary_constructor!(LeOutput, le, Le, R; where [L::DType: Real]);
}

binary_op!(
    /// Elementwise and operation returning zero or one.
    And, and, T: [TensorElement] => u8
);

binary_op!(
    /// Elementwise or operation returning zero or one.
    Or, or, T: [TensorElement] => u8
);

binary_op!(
    /// Elementwise xor operation returning zero or one.
    Xor, xor, T: [TensorElement] => u8
);

impl<L, R> TensorBoolean<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement,
{
    binary_constructor!(AndOutput, and, And, R);

    binary_constructor!(OrOutput, or, Or, R);

    binary_constructor!(XorOutput, xor, Xor, R);
}
