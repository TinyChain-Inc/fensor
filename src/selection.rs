//! Lazy conditional selection over three expression sources.

use ha_ndarray::{ArrayAccess, NDArrayWhere};

use crate::expression::{self, Batch, Expression};
use crate::request::BatchRequest;
use crate::{
    BoxFuture, Error, Layout, Result, TensorElement, TensorGeometry, TensorRead, TensorTransform,
    TensorViewSemantics, TensorWhere,
};

/// Read-only selection over the condition and both branches.
///
/// A zero condition selects `or_else`; every nonzero condition selects `then`.
/// Both branches are read, so errors in an unselected branch still propagate.
///
/// Sources may use distinct adapters without requiring adapter clones.
///
/// ```
/// use fensor::{Tensor, TensorFileEntry, TensorMathScalar, TensorCompareScalar, TensorBoolean, TensorWhere, TensorTransform};
/// async fn example<A: TensorFileEntry<f32>, B: TensorFileEntry<f32>, C: TensorFileEntry<u8>>(
///     a: &Tensor<A, f32>, b: &Tensor<B, f32>, c: &Tensor<C, u8>,
/// ) -> fensor::Result<()> {
///     let predicate = a.view().add_scalar(1.).await?.gt_scalar(0.).await?;
///     let condition = predicate.and(&c.view()).await?;
///     let view = condition.cond(&a.view(), &b.view()).await?.flip(0)?.clone();
///     let _cloned = view.clone();
///     Ok(())
/// }
/// ```
///
/// Selection cannot be written through, including after transformation.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorCompareScalar, TensorWhere, TensorTransform, TensorWrite};
/// fn writable<T: TensorWrite>(_: &T) {}
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     let c = a.view().eq_scalar(0.).await.unwrap();
///     writable(&c.cond(&a.view(), &a.view()).await.unwrap().flip(0).unwrap());
/// }
/// ```
///
/// Conditions must have dtype u8.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorWhere};
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     let _ = a.view().cond(&a.view(), &a.view()).await;
/// }
/// ```
///
/// Branch dtypes must match.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorWhere};
/// async fn example<F: TensorFileEntry<f32> + TensorFileEntry<f64> + TensorFileEntry<u8>>(
///     c: &Tensor<F, u8>, a: &Tensor<F, f32>, b: &Tensor<F, f64>,
/// ) { let _ = c.view().cond(&a.view(), &b.view()).await; }
/// ```
#[derive(Clone)]
pub struct WhereView<Condition, Then, Else>
where
    Then: TensorGeometry,
    Then::DType: TensorElement,
{
    condition: Condition,
    then: Then,
    or_else: Else,
    zero: Then::DType,
}

impl<C, L, R> TensorWhere<L, R> for C
where
    C: Expression<DType = u8> + Clone,
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement,
{
    type Output = WhereView<Self, L, R>;

    fn cond<'a>(&'a self, then: &'a L, or_else: &'a R) -> BoxFuture<'a, Result<Self::Output>> {
        Box::pin(async move {
            if self.shape() != then.shape() || self.shape() != or_else.shape() {
                return Err(Error::InvalidSchema(format!(
                    "conditional operand shapes differ: {:?}, {:?}, {:?}",
                    self.shape(),
                    then.shape(),
                    or_else.shape()
                )));
            }

            Ok(WhereView {
                zero: or_else.implicit_zero(),
                condition: self.clone(),
                then: then.clone(),
                or_else: or_else.clone(),
            })
        })
    }
}

impl<C, L, R> TensorGeometry for WhereView<C, L, R>
where
    C: TensorGeometry<DType = u8>,
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    type DType = L::DType;

    fn dtype(&self) -> crate::NumberType {
        self.then.dtype()
    }

    fn shape(&self) -> &[u64] {
        self.condition.shape()
    }

    fn layout(&self) -> Layout {
        match (
            self.condition.layout(),
            self.then.layout(),
            self.or_else.layout(),
        ) {
            (Layout::Sparse { .. }, Layout::Sparse { .. }, Layout::Sparse { .. }) => {
                Layout::Sparse { axis: None }
            }
            _ => Layout::Dense,
        }
    }
}

impl<C, L, R> TensorViewSemantics for WhereView<C, L, R>
where
    C: TensorGeometry<DType = u8>,
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    fn is_base_tensor(&self) -> bool {
        false
    }
}

impl<C, L, R> Expression for WhereView<C, L, R>
where
    C: Expression<DType = u8>,
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
{
    fn implicit_zero(&self) -> Self::DType {
        self.zero
    }

    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([
            self.condition.expression_nodes()?,
            self.then.expression_nodes()?,
            self.or_else.expression_nodes()?,
        ])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.condition.detach_sources(pending);
        self.then.detach_sources(pending);
        self.or_else.detach_sources(pending);
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

        let condition_slice = slice.clone();
        let then_slice = slice.clone();
        Ok(expression::traversal::Ordered::Sources(vec![
            Box::new(move || self.condition.ordered_step(condition_slice)),
            Box::new(move || self.then.ordered_step(then_slice)),
            Box::new(move || self.or_else.ordered_step(slice)),
        ]))
    }

    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        Ok(expression::traversal::Preferred::Sources(vec![
            Box::new(move || self.condition.preferred_step(shape)),
            Box::new(move || self.then.preferred_step(shape)),
            Box::new(move || self.or_else.preferred_step(shape)),
        ]))
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let condition = context.batch(&self.condition, coords.clone()).await?;
            let then = context.batch(&self.then, coords.clone()).await?;
            let or_else = context.batch(&self.or_else, coords).await?;

            Batch::from_array(ArrayAccess::from(
                condition.array.cond(then.array, or_else.array)?,
            ))
        })
    }
}

impl<C, L, R> TensorRead for WhereView<C, L, R>
where
    C: Expression<DType = u8>,
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
{
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<C, L, R> TensorTransform for WhereView<C, L, R>
where
    C: TensorTransform<DType = u8>,
    L: TensorTransform,
    R: TensorTransform<DType = L::DType>,
    L::DType: TensorElement,
{
    crate::mapping::transform_methods!(operands: condition, then, or_else; preserve_rest);
}
