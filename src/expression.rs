//! Shared bounded batch consumption; concrete operations own the mathematics.
//! The driver evaluates continuations without recursive polling, traversal plans
//! requests iteratively, and owned expressions detach their operands on destruction.

use std::ops::Deref;

use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArray, NDArrayRead, Number};
use tokio::sync::OwnedSemaphorePermit;

use crate::request::{self, BatchRequest};
use crate::traits::BoxFuture;
use crate::{Error, Result, Tensor, TensorElement, TensorFileEntry, TensorGeometry, TensorView};

mod driver;
pub use driver::Context;
pub(crate) use driver::evaluate_batch;

pub(crate) mod traversal;

/// Execution limit independent of filesystem storage block capacity.
pub(crate) const MAX_BATCH_ELEMENTS: usize = 4096;

fn validate_len(context: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidLayout(format!(
            "{context}: expected {expected} elements, got {actual}"
        )));
    }

    Ok(())
}

fn validate_bound(context: &str, actual: usize) -> Result<()> {
    if actual > MAX_BATCH_ELEMENTS {
        return Err(Error::InvalidLayout(format!(
            "{context}: expected at most {MAX_BATCH_ELEMENTS} elements, got {actual}"
        )));
    }

    Ok(())
}

// This module is private: callers cannot introduce arbitrary expression sources.
pub trait Expression: TensorGeometry
where
    Self::DType: TensorElement,
{
    /// Value of an omitted coordinate, including the sign of numerical zero.
    /// Computed owners cache this scalar; querying it never traverses operands.
    fn implicit_zero(&self) -> Self::DType {
        Self::DType::ZERO
    }

    /// Count retained descriptions for admission at the dynamic value boundary.
    fn expression_nodes(&self) -> Result<usize> {
        Ok(1)
    }

    /// Detach runtime-owned operands before releasing this expression description.
    fn detach_sources(&mut self, _pending: &mut Vec<Box<dyn crate::owned::Drain>>) {}

    /// Describe one preferred-request step for the consumer's logical shape.
    /// None leaves request selection to another source or the consumer.
    fn preferred_step<'a>(&'a self, _shape: &'a [u64]) -> Result<traversal::Preferred<'a>> {
        Ok(traversal::Preferred::Ready(None))
    }

    /// Describe ordered occupied-request candidates, independent of numerical values.
    fn ordered_step(&self, slice: crate::slice::Slice) -> Result<traversal::Ordered<'_>> {
        Ok(traversal::Ordered::Ready(slice.stream()))
    }

    /// Describe one selection step; native readers may retain storage order.
    fn selection_step(&self, slice: crate::slice::Slice) -> Result<traversal::Selection<'_>> {
        Ok(traversal::Selection::Ready(slice.stream()))
    }

    fn build<'a>(
        &'a self,
        context: Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>>;
}

/// Owned bounded requests; expression types themselves use static dispatch.
pub(crate) type RequestIterator = Box<dyn Iterator<Item = BatchRequest> + Send>;

pub(crate) async fn read_value<E: Expression + ?Sized>(
    source: &E,
    coord: &[u64],
) -> Result<E::DType>
where
    E::DType: TensorElement,
{
    Ok(evaluate_batch(source, &BatchRequest::point(coord))
        .await?
        .values[0])
}

pub(crate) fn read_blocks<'a, H>(
    source: H,
) -> Result<crate::ValueBlockStream<'a, <H::Target as TensorGeometry>::DType>>
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
{
    let requests = request::linear_requests(source.shape())?;
    Ok(
        ordered_requests(source, futures::stream::iter(requests.map(Ok)))
            .map_ok(|(_, batch)| batch.values)
            .boxed(),
    )
}

macro_rules! reader_members {
    ($($method:ident),+ $(,)?) => { $($crate::expression::reader_members!(@member $method);)+ };
    (@member read_value) => {
        fn read_value<'a>(&'a self, coord: &'a [u64]) -> $crate::BoxFuture<'a, $crate::Result<Self::DType>> {
            Box::pin($crate::expression::read_value(self, coord))
        }
    };
    (@member read_blocks) => {
        fn read_blocks(&self) -> $crate::Result<$crate::ValueBlockStream<'_, Self::DType>> {
            $crate::expression::read_blocks(self)
        }
    };
    (@member read_coordinate_blocks) => {
        fn read_coordinate_blocks(&self) -> $crate::Result<$crate::CoordinateBlockStream<'_, Self::DType>> {
            $crate::expression::coordinate_blocks(self)
        }
    };
    (@member read_sparse_elements_in_order) => {
        fn read_sparse_elements_in_order<'a>(&'a self, range: $crate::Range, order: $crate::Axes)
            -> $crate::BoxFuture<'a, $crate::Result<$crate::SparseElementStream<'a, Self::DType>>>
        {
            Box::pin(async move { $crate::expression::ordered_sparse(self, range, order) })
        }
    };
}
pub(crate) use reader_members;

pub fn coordinate_blocks<E: Expression + ?Sized>(
    expression: &E,
) -> Result<crate::CoordinateBlockStream<'_, E::DType>>
where
    E::DType: TensorElement,
{
    Ok(completion_batches(expression)?
        .map(move |batch| {
            let (request, batch) = batch?;
            Ok((request.into_coordinates(expression.shape())?, batch.values))
        })
        .boxed())
}

/// Shared completion-order consumption before a caller chooses its output representation.
pub(crate) fn completion_batches<'a, H>(
    expression: H,
) -> Result<
    impl futures::Stream<Item = Result<CoordinateBatch<<H::Target as TensorGeometry>::DType>>>
    + Send
    + 'a,
>
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
{
    let requests = match traversal::preferred(&*expression, expression.shape())? {
        Some(requests) => requests,
        None => Box::new(request::linear_requests(expression.shape())?),
    };

    Ok(
        evaluation_futures(expression, futures::stream::iter(requests.map(Ok)))
            .buffer_unordered(num_cpus::get().max(1)),
    )
}

pub struct Batch<T: TensorElement> {
    _reservation: Option<OwnedSemaphorePermit>,
    pub array: ArrayAccess<'static, T>,
}

pub fn batch_array<T: TensorElement>(values: Vec<T>) -> Result<ArrayAccess<'static, T>> {
    validate_bound("batch array", values.len())?;
    let shape = std::iter::once(values.len()).collect();

    Ok(ArrayAccess::from(Array::new(Buffer::from(values), shape)?))
}

impl<T: TensorElement> Batch<T> {
    /// Complete a numerical operation before another expression layer can use it.
    pub(crate) fn from_array(array: ArrayAccess<'static, T>) -> Result<Self> {
        validate_bound("batch expression", array.size())?;
        Ok(Self {
            _reservation: None,
            array: ArrayAccess::from(array.into_read()?),
        })
    }

    /// Host values already own their completed buffer; do not realize them again.
    pub(crate) fn from_values(values: Vec<T>) -> Result<Self> {
        Ok(Self {
            _reservation: None,
            array: batch_array(values)?,
        })
    }

    fn validate(&self, expected: usize) -> Result<()> {
        validate_bound("batch expression", self.array.size())?;
        validate_len("batch expression", self.array.size(), expected)
    }

    fn into_evaluated(self) -> Result<EvaluatedBatch<T>> {
        let expected = self.array.size();
        self.validate(expected)?;
        let batch = EvaluatedBatch {
            _reservation: self._reservation,
            values: self.array.buffer()?.to_slice()?.into_vec(),
        };
        batch.validate(expected)?;
        Ok(batch)
    }
}

pub struct EvaluatedBatch<T> {
    _reservation: Option<OwnedSemaphorePermit>,
    pub values: Vec<T>,
}

impl<T> EvaluatedBatch<T> {
    fn validate(&self, expected: usize) -> Result<()> {
        validate_bound("evaluated batch", self.values.len())?;
        validate_len("evaluated batch", self.values.len(), expected)
    }
}

/// Allocate compact request coordinates only for nonzero sparse output. Explicit
/// coordinate payloads transfer their existing allocations, retaining order and duplicates.
pub(crate) fn sparse_elements<T: TensorElement>(
    request: BatchRequest,
    batch: EvaluatedBatch<T>,
    shape: &[u64],
) -> Result<impl Iterator<Item = Result<(Vec<u64>, T)>> + Send + use<T>> {
    enum Coordinates {
        Explicit(std::vec::IntoIter<Vec<u64>>),
        Compact(request::Cursor<BatchRequest, std::sync::Arc<[u64]>>),
    }

    batch.validate(request.len())?;
    let mut coordinates = match request.kind() {
        request::RequestKind::Explicit(_) => {
            Coordinates::Explicit(request.into_coordinates(shape)?.into_iter())
        }
        _ => Coordinates::Compact(request.into_cursor(shape.into())?),
    };

    let mut scratch = crate::schema::Coord::new();
    Ok(batch
        .values
        .into_iter()
        .filter_map(move |value| match &mut coordinates {
            Coordinates::Explicit(coords) => {
                let coord = coords.next().expect("validated sparse output cardinality");
                (value != T::default()).then_some(Ok((coord, value)))
            }
            Coordinates::Compact(cursor) => {
                assert!(
                    cursor.next_into(&mut scratch),
                    "validated sparse output cardinality"
                );
                if value == T::default() {
                    return None;
                }

                #[cfg(test)]
                crate::read_metrics::record(|m| m.expanded_coordinates += 1);
                Some(Ok((scratch.to_vec(), value)))
            }
        }))
}

type CoordinateBatch<T> = (BatchRequest, EvaluatedBatch<T>);

/// Evaluate bounded coordinate batches with at most `num_cpus::get().max(1)`
/// batches in flight. Expression temporaries scale with batch size, expression
/// size, and concurrency, not total tensor size. Collecting the returned stream
/// can still allocate whole-tensor output in the caller.
pub fn ordered_batches<'a, E, I>(
    expression: &'a E,
    coords: I,
) -> BoxStream<'a, Result<CoordinateBatch<E::DType>>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
    I: Iterator<Item = BatchRequest> + Send + 'a,
{
    ordered_requests(expression, futures::stream::iter(coords.map(Ok)))
}

/// Ordered delivery for consumers whose value or error boundaries require it.
/// Also accepts demand-driven sparse-index requests without another buffer.
pub(crate) fn ordered_requests<'a, H, R>(
    expression: H,
    requests: R,
) -> BoxStream<'a, Result<CoordinateBatch<<H::Target as TensorGeometry>::DType>>>
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
    R: futures::Stream<Item = Result<BatchRequest>> + Send + 'a,
{
    evaluation_futures(expression, requests)
        .buffered(num_cpus::get().max(1))
        .boxed()
}

/// Build lazy evaluation futures from borrowed or owned source handles. Only
/// the outer consumer chooses buffering and delivery order.
pub(crate) fn evaluation_futures<'a, H, R>(
    expression: H,
    requests: R,
) -> impl futures::Stream<
    Item = impl std::future::Future<
        Output = Result<CoordinateBatch<<H::Target as TensorGeometry>::DType>>,
    > + Send
           + 'a,
> + Send
+ 'a
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
    R: futures::Stream<Item = Result<BatchRequest>> + Send + 'a,
{
    requests.map(move |coords| {
        let expression = expression.clone();
        async move {
            let coords = coords?;
            let values = evaluate_batch(&*expression, &coords).await?;
            Ok((coords, values))
        }
    })
}

impl<S: crate::TensorSource> Expression for TensorView<S>
where
    S::DType: TensorElement,
{
    fn selection_step(&self, slice: crate::slice::Slice) -> Result<traversal::Selection<'_>> {
        self.slice_requests_for_storage(slice)
            .map(traversal::Selection::Ready)
    }

    fn ordered_step(&self, slice: crate::slice::Slice) -> Result<traversal::Ordered<'_>> {
        self.ordered_storage_requests(slice)
            .map(traversal::Ordered::Ready)
    }

    fn build<'a>(
        &'a self,
        _context: Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<S::DType>>> {
        Box::pin(async move {
            let values = self.read_batch(&coords).await?;

            Batch::from_values(values)
        })
    }
}

impl<FE, T> Expression for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn selection_step(&self, slice: crate::slice::Slice) -> Result<traversal::Selection<'_>> {
        crate::storage::slice_requests(self, slice, None).map(traversal::Selection::Ready)
    }

    fn ordered_step(&self, slice: crate::slice::Slice) -> Result<traversal::Ordered<'_>> {
        crate::storage::ordered_requests(
            self,
            slice,
            crate::mapping::StorageSlice::identity(self.shape()),
        )
        .map(traversal::Ordered::Ready)
    }

    fn build<'a>(
        &'a self,
        _context: Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let values = self.read_batch(&coords, None).await?;
            Batch::from_values(values)
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/expression/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/expression/concurrency_tests.rs"]
mod concurrency_tests;

pub(crate) fn ordered_sparse<'a, H>(
    source: H,
    range: crate::Range,
    order: crate::Axes,
) -> Result<crate::SparseElementStream<'a, <H::Target as TensorGeometry>::DType>>
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
{
    let slice = crate::traits::sparse_slice(&*source, range, order)?;
    let requests = traversal::ordered(&*source, slice)?;
    Ok(sparse_stream(source, requests))
}

/// Consume ordered candidates with one ordered buffer and no intermediate output collection.
pub(crate) fn sparse_stream<'a, H, R>(
    source: H,
    requests: R,
) -> crate::SparseElementStream<'a, <H::Target as TensorGeometry>::DType>
where
    H: Deref + Clone + Send + Sync + 'a,
    H::Target: Expression,
    <H::Target as TensorGeometry>::DType: TensorElement,
    R: futures::Stream<Item = Result<BatchRequest>> + Send + 'a,
{
    let shape = source.shape().to_vec();
    ordered_requests(source, requests)
        .map(move |result| {
            let (request, batch) = result?;
            Ok::<_, Error>(futures::stream::iter(sparse_elements(
                request, batch, &shape,
            )?))
        })
        .try_flatten()
        .boxed()
}
