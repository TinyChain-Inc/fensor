//! Private batch construction shared by typed elementwise expressions.

use std::ops::Deref;

use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArray, NDArrayRead, Number};

use crate::request::{self, BatchRequest};
use crate::traits::BoxFuture;
use crate::{Error, Result, Tensor, TensorElement, TensorFileEntry, TensorGeometry, TensorView};

mod driver;
pub use driver::Context;

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

pub(crate) fn read_blocks<E: Expression + ?Sized>(
    source: &E,
) -> Result<crate::ValueBlockStream<'_, E::DType>>
where
    E::DType: TensorElement,
{
    let requests = request::linear_requests(source.shape())?;
    Ok(ordered_batches(source, requests)
        .map_ok(|(_, batch)| batch.values)
        .boxed())
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
    pub(crate) _allocation: Option<driver::Allocation>,
    pub array: ArrayAccess<'static, T>,
}

pub fn batch_array<T: TensorElement>(values: Vec<T>) -> Result<ArrayAccess<'static, T>> {
    validate_bound("batch array", values.len())?;
    let shape = std::iter::once(values.len()).collect();

    Ok(ArrayAccess::from(Array::new(Buffer::from(values), shape)?))
}

impl<T: TensorElement> Batch<T> {
    fn validate(&self, expected: usize) -> Result<()> {
        validate_bound("batch expression", self.array.size())?;
        validate_len("batch expression", self.array.size(), expected)
    }

    fn into_evaluated(self) -> Result<EvaluatedBatch<T>> {
        let expected = self.array.size();
        self.validate(expected)?;
        let batch = EvaluatedBatch {
            _allocation: self._allocation,
            values: self.array.buffer()?.to_slice()?.into_vec(),
        };
        batch.validate(expected)?;
        Ok(batch)
    }

    pub(crate) fn realize(self) -> Result<Self> {
        self.validate(self.array.size())?;
        Ok(Self {
            _allocation: self._allocation,
            array: ArrayAccess::from(self.array.into_read()?),
        })
    }
}

pub async fn evaluate_batch<E>(
    expression: &E,
    coords: &BatchRequest,
) -> Result<EvaluatedBatch<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    driver::evaluate(expression, std::sync::Arc::new(coords.clone())).await
}

pub struct EvaluatedBatch<T> {
    pub(crate) _allocation: Option<driver::Allocation>,
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

            Ok(Batch {
                _allocation: None,
                array: batch_array(values)?,
            })
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
            Ok(Batch {
                _allocation: None,
                array: batch_array(values)?,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layout;
    use crate::test_support::counters::Counter;

    // Wrap real storage only to observe dispatch; numerical reads still delegate.
    #[derive(Clone)]
    struct Provider<'a> {
        source: &'a Tensor<crate::test_support::FsEntry, u8>,
        calls: &'a Counter,
        provide: fn(&[u64]) -> Result<Option<RequestIterator>>,
    }

    impl TensorGeometry for Provider<'_> {
        type DType = u8;

        fn dtype(&self) -> crate::NumberType {
            self.source.dtype()
        }

        fn shape(&self) -> &[u64] {
            self.source.shape()
        }

        fn layout(&self) -> Layout {
            self.source.layout()
        }
    }

    impl Expression for Provider<'_> {
        fn preferred_step<'a>(&'a self, shape: &'a [u64]) -> Result<traversal::Preferred<'a>> {
            self.calls.increment();
            (self.provide)(shape).map(traversal::Preferred::Ready)
        }

        fn ordered_step(&self, slice: crate::slice::Slice) -> Result<traversal::Ordered<'_>> {
            self.calls.increment();
            Ok(traversal::Ordered::Ready(
                match (self.provide)(self.shape())? {
                    Some(requests) => futures::stream::iter(requests.map(Ok)).boxed(),
                    None => slice.stream(),
                },
            ))
        }

        fn build<'a>(
            &'a self,
            context: Context<'a>,
            request: std::sync::Arc<BatchRequest>,
        ) -> BoxFuture<'a, Result<Batch<u8>>> {
            self.source.build(context, request)
        }
    }

    #[tokio::test]
    async fn request_providers_delegate_once_in_order_and_propagate_errors() {
        use crate::{TensorMath, TensorUnaryBoolean, TensorWhere};

        type Provide = fn(&[u64]) -> Result<Option<RequestIterator>>;
        let none: Provide = |_| Ok(None);
        let some: Provide = |shape| Ok(Some(Box::new(request::linear_requests(shape)?)));
        let error: Provide = |_| Err(Error::Unsupported("provider failure".into()));
        let (root, dir) = crate::test_support::new_dir("request_providers").await;
        let tensor = Tensor::create(
            dir,
            crate::TensorSchema::new(
                <u8 as number_general::DType>::dtype(),
                ha_ndarray::shape![2, 3],
            )
            .unwrap(),
            Layout::Dense,
            2,
        )
        .await
        .unwrap();

        for (providers, expected, fails, supplies) in [
            ([some, error, error], [1, 0, 0], false, true),
            ([some, some, some], [1, 0, 0], false, true),
            ([none, some, error], [1, 1, 0], false, true),
            ([none, none, some], [1, 1, 1], false, true),
            ([error, some, some], [1, 0, 0], true, false),
            ([none, error, some], [1, 1, 0], true, false),
            ([none, none, error], [1, 1, 1], true, false),
            ([none, none, none], [1, 1, 1], false, false),
        ] {
            let calls = [Counter::new(), Counter::new(), Counter::new()];
            let [a, b, c] = std::array::from_fn(|i| Provider {
                source: &tensor,
                calls: &calls[i],
                provide: providers[i],
            });
            // Consumer shape is forwarded unchanged, rather than replaced by a leaf shape.
            let selected = a.cond(&b, &c).await.unwrap().not().await.unwrap();
            let result = traversal::preferred(&selected, &[7, 3]);
            assert_eq!(result.is_err(), fails);
            assert_eq!(matches!(&result, Ok(Some(_))), supplies);
            if let Ok(Some(mut requests)) = result {
                assert_eq!(requests.next().unwrap().len(), 21);
                assert!(requests.next().is_none());
            }
            assert_eq!(calls.each_ref().map(|n| n.read()), expected);

            for call in &calls {
                call.reset();
            }

            let binary = a.add(&b).await.unwrap();
            let result = traversal::preferred(&binary, &[7, 3]);
            assert_eq!(result.is_err(), fails && expected[2] == 0);
            assert_eq!(matches!(&result, Ok(Some(_))), supplies && expected[2] == 0);
            assert_eq!(
                calls.each_ref().map(|n| n.read()),
                [expected[0], expected[1], 0]
            );
        }
        crate::test_support::cleanup(&root).await;
    }

    #[tokio::test]
    async fn ordered_providers_visit_operands_left_to_right() {
        use crate::{TensorAbs, TensorMath, TensorWhere};

        type Provide = fn(&[u64]) -> Result<Option<RequestIterator>>;
        let none: Provide = |_| Ok(None);
        let some: Provide = |_| {
            Ok(Some(Box::new(
                [
                    BatchRequest::explicit(vec![vec![0], vec![0], vec![2]])?,
                    BatchRequest::explicit(vec![vec![4]])?,
                ]
                .into_iter(),
            )))
        };

        let error: Provide = |_| Err(Error::Unsupported("ordered provider failure".into()));
        let (root, tensor) = crate::test_support::fixture::source(
            "ordered_providers",
            vec![6].into(),
            Layout::Sparse { axis: None },
            2,
            4096,
            [0u8; 6],
        )
        .await;

        for (providers, expected, fails) in [
            ([none, error, none], [1, 1, 0], true),
            ([none, none, error], [1, 1, 1], true),
            ([error, none, none], [1, 0, 0], true),
            ([none, some, some], [1, 1, 1], false),
            ([some, none, some], [1, 1, 1], false),
            ([some, some, none], [1, 1, 1], false),
        ] {
            let calls = [Counter::new(), Counter::new(), Counter::new()];
            let [condition, left, right] = std::array::from_fn(|i| Provider {
                source: &tensor,
                calls: &calls[i],
                provide: providers[i],
            });
            let selected = condition
                .cond(&left, &right)
                .await
                .unwrap()
                .abs()
                .await
                .unwrap();
            let requests = traversal::ordered(&selected, crate::slice::Slice::full(&[6]).unwrap());
            assert_eq!(requests.is_err(), fails);
            assert_eq!(calls.each_ref().map(|n| n.read()), expected);
            if let Ok(mut requests) = requests {
                let mut coordinates = Vec::new();

                while let Some(request) = requests.try_next().await.unwrap() {
                    assert!(request.len() <= MAX_BATCH_ELEMENTS);
                    coordinates.extend(request.into_coordinates(&[6]).unwrap());
                }
                assert_eq!(coordinates, (0..6).map(|i| vec![i]).collect::<Vec<_>>());
            }
        }

        for (providers, expected) in [([error, none], [1, 0]), ([none, error], [1, 1])] {
            let calls = [Counter::new(), Counter::new()];
            let [left, right] = std::array::from_fn(|i| Provider {
                source: &tensor,
                calls: &calls[i],
                provide: providers[i],
            });
            assert!(
                traversal::ordered(
                    &left.add(&right).await.unwrap(),
                    crate::slice::Slice::full(&[6]).unwrap(),
                )
                .is_err()
            );
            assert_eq!(calls.each_ref().map(|n| n.read()), expected);
        }
        crate::test_support::cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_output_preserves_order_duplicates_and_nan() {
        use crate::mapping::CoordinateMap;
        use crate::request::{Axis, Cartesian, RequestKind};

        let mapping = CoordinateMap::identity(ha_ndarray::shape![2, 3], &[3, 1])
            .transpose(None)
            .unwrap()
            .flip(0)
            .unwrap();
        let mut high_rank = vec![1; crate::PORTABLE_INLINE_RANK + 2];
        *high_rank.last_mut().unwrap() = 7;
        let pattern = [
            3f32,
            -0.,
            4.,
            f32::from_bits(0x7fc01234),
            0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let cases = [
            (
                "explicit",
                BatchRequest::explicit(vec![vec![2], vec![0], vec![2], vec![1], vec![0]]).unwrap(),
                vec![3],
                &pattern[..],
            ),
            (
                "linear batch boundary",
                BatchRequest::linear(MAX_BATCH_ELEMENTS as u64 - 1, MAX_BATCH_ELEMENTS).unwrap(),
                vec![2 * MAX_BATCH_ELEMENTS as u64],
                &pattern[..],
            ),
            (
                "rectangular duplicates",
                BatchRequest::rectangles(vec![
                    Cartesian::new(vec![
                        Axis::Selected(vec![2, 0, 2]),
                        Axis::Selected(vec![1, 3]),
                    ])
                    .unwrap(),
                ])
                .unwrap(),
                vec![3, 4],
                &pattern[..],
            ),
            (
                "transformed flat runs",
                BatchRequest::linear(0, 6)
                    .unwrap()
                    .mapped(&mapping, &[2, 3], &[3, 1])
                    .unwrap(),
                vec![2, 3],
                &pattern[..],
            ),
            (
                "high rank",
                BatchRequest::linear(0, 7).unwrap(),
                high_rank,
                &pattern[..],
            ),
            (
                "empty",
                BatchRequest::linear(0, 0).unwrap(),
                vec![1],
                &pattern[..],
            ),
            (
                "all zeros",
                BatchRequest::linear(0, 7).unwrap(),
                vec![7],
                &[0., -0.][..],
            ),
        ];

        for (name, request, shape, pattern) in cases {
            let explicit = matches!(request.kind(), RequestKind::Explicit(_));
            let expected_coords = request.coordinates(&shape).unwrap();
            let pointers = match request.kind() {
                RequestKind::Explicit(coords) => coords
                    .iter()
                    .map(|coord| coord.as_ptr())
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            };

            let values: Vec<_> = (0..request.len())
                .map(|i| pattern[i % pattern.len()])
                .collect();
            let expected: Vec<_> = expected_coords
                .iter()
                .zip(&values)
                .enumerate()
                .filter(|(_, (_, value))| **value != 0.)
                .map(|(i, (coord, value))| (i, coord.clone(), value.to_bits()))
                .collect();
            crate::read_metrics::CURRENT
                .scope(Default::default(), async {
                    let output = sparse_elements(
                        request,
                        EvaluatedBatch {
                            _allocation: None,
                            values,
                        },
                        &shape,
                    )
                    .unwrap();
                    crate::read_metrics::CURRENT.with(|m| {
                        assert_eq!(m.borrow().expanded_coordinates, 0, "{name}");
                    });
                    let output = output.collect::<Result<Vec<_>>>().unwrap();
                    assert_eq!(output.len(), expected.len(), "{name}");

                    for ((coord, value), (i, expected_coord, bits)) in output.iter().zip(&expected)
                    {
                        assert_eq!(coord, expected_coord, "{name}");
                        assert_eq!(value.to_bits(), *bits, "{name}");
                        if explicit {
                            assert_eq!(coord.as_ptr(), pointers[*i], "{name}");
                        }
                    }
                    crate::read_metrics::CURRENT.with(|m| {
                        assert_eq!(
                            m.borrow().expanded_coordinates,
                            if explicit { 0 } else { expected.len() },
                            "{name}"
                        );
                    });
                })
                .await;
        }
    }

    #[test]
    fn sparse_output_rejects_mismatched_values_and_invalid_requests() {
        let batch = EvaluatedBatch {
            _allocation: None,
            values: vec![1u8, 2],
        };
        assert!(matches!(
            sparse_elements(BatchRequest::point(&[0]), batch, &[1]),
            Err(Error::InvalidLayout(_))
        ));

        for request in [
            BatchRequest::linear(2, 1).unwrap(),
            BatchRequest::linear(2, 0).unwrap(),
            BatchRequest::rectangles(vec![
                request::Cartesian::new(vec![request::Axis::range(1, 1)]).unwrap(),
            ])
            .unwrap(),
        ] {
            let batch = EvaluatedBatch {
                _allocation: None,
                values: vec![0u8; request.len()],
            };
            assert!(matches!(
                sparse_elements(request, batch, &[1]),
                Err(Error::InvalidCoord(_))
            ));
        }
    }

    #[test]
    fn tiled_requests_retain_only_a_bounded_prefix() {
        let shape = [2, 1_000_000_000, 1_000_000_000];
        let request = request::tiled_requests(&shape).unwrap().next().unwrap();
        let coords = request.coordinates(&shape).unwrap();
        assert_eq!(coords.len(), MAX_BATCH_ELEMENTS);
        assert_eq!(coords[32], vec![0, 1, 0]);

        for shape in [vec![2, 33, 35], vec![1], vec![2, 1, 9]] {
            let mut tiled: Vec<_> = request::tiled_requests(&shape)
                .unwrap()
                .flat_map(|request| request.coordinates(&shape).unwrap())
                .collect();
            tiled.sort();
            assert_eq!(
                tiled,
                crate::schema::row_major_coords(&shape)
                    .unwrap()
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn malformed_batches_fail_closed() {
        assert!(batch_array(vec![0u8; MAX_BATCH_ELEMENTS + 1]).is_err());
        let batch = Batch {
            _allocation: None,
            array: batch_array(vec![1u8, 2]).unwrap(),
        };
        assert!(batch.validate(1).is_err());
        let evaluated = EvaluatedBatch {
            _allocation: None,
            values: vec![1u8],
        };
        assert!(evaluated.validate(2).is_err());
        let evaluated = EvaluatedBatch {
            _allocation: None,
            values: vec![0u8; MAX_BATCH_ELEMENTS + 1],
        };
        assert!(evaluated.validate(MAX_BATCH_ELEMENTS + 1).is_err());
    }

    #[test]
    fn batching_only_consumes_its_bounded_prefix() {
        use std::cell::Cell;
        let consumed = Cell::new(0);
        let coords = (0..1_000_000_000u64).map(|i| {
            consumed.set(consumed.get() + 1);
            vec![i]
        });
        let mut batches = crate::traits::coordinate_batches(coords);
        assert_eq!(consumed.get(), 0);
        assert_eq!(batches.next().unwrap().len(), MAX_BATCH_ELEMENTS);
        assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
        drop(batches);
        assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
    }
}

#[cfg(test)]
#[path = "expression/concurrency_tests.rs"]
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
