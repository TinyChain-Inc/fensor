//! Lazy conditional selection over three expression sources.

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{ArrayAccess, Axes, NDArrayWhere, Range, Shape};

use crate::expression::{self, Batch, Expression};
use crate::request::{self, BatchRequest};
use crate::{
    BoxFuture, Error, Layout, Result, SparseElementStream, TensorElement, TensorGeometry,
    TensorRead, TensorTransform, TensorViewSemantics, TensorWhere, ValueBlockStream,
};

/// Read-only selection retaining support from the condition and both branches.
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
pub struct WhereView<Condition, Then, Else> {
    condition: Condition,
    then: Then,
    or_else: Else,
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

    fn shape(&self) -> &[usize] {
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

    fn supports_write_through(&self) -> bool {
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
    fn preferred_requests(&self, shape: &[usize]) -> Result<Option<expression::RequestIterator>> {
        if let Some(requests) = self.condition.preferred_requests(shape)? {
            return Ok(Some(requests));
        }
        match self.then.preferred_requests(shape)? {
            Some(requests) => Ok(Some(requests)),
            None => self.or_else.preferred_requests(shape),
        }
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let condition = self.condition.build(coords).await?;
            let then = self.then.build(coords).await?;
            let or_else = self.or_else.build(coords).await?;
            let support = expression::union_support(
                condition.support,
                expression::union_support(then.support, or_else.support)?,
            )?;

            Batch {
                array: ArrayAccess::from(condition.array.cond(then.array, or_else.array)?),
                support,
            }
            .masked()
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

impl<C, L, R> TensorTransform for WhereView<C, L, R>
where
    C: TensorTransform<DType = u8>,
    L: TensorTransform,
    R: TensorTransform<DType = L::DType>,
    L::DType: TensorElement,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            condition: self.condition.reshape(shape.clone())?,
            then: self.then.reshape(shape.clone())?,
            or_else: self.or_else.reshape(shape)?,
        })
    }

    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            condition: self.condition.broadcast(shape.clone())?,
            then: self.then.broadcast(shape.clone())?,
            or_else: self.or_else.broadcast(shape)?,
        })
    }

    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            condition: self.condition.flip(axis)?,
            then: self.then.flip(axis)?,
            or_else: self.or_else.flip(axis)?,
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            condition: self.condition.slice(range.clone())?,
            then: self.then.slice(range.clone())?,
            or_else: self.or_else.slice(range)?,
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            condition: self.condition.squeeze(axes.clone())?,
            then: self.then.squeeze(axes.clone())?,
            or_else: self.or_else.squeeze(axes)?,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            condition: self.condition.transpose(permutation.clone())?,
            then: self.then.transpose(permutation.clone())?,
            or_else: self.or_else.transpose(permutation)?,
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            condition: self.condition.unsqueeze(axes.clone())?,
            then: self.then.unsqueeze(axes.clone())?,
            or_else: self.or_else.unsqueeze(axes)?,
        })
    }
}
