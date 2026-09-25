//! Private batch construction shared by typed elementwise expressions.

use futures::StreamExt;
use futures::stream::BoxStream;
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArray, NDArrayRead, NDArrayWhere};

use crate::request::{self, BatchRequest};
use crate::traits::BoxFuture;
use crate::{
    Error, Layout, Result, Tensor, TensorElement, TensorFileEntry, TensorGeometry, TensorView,
};

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

fn validate_support(support: &Option<Vec<u8>>, expected: usize) -> Result<()> {
    if let Some(support) = support {
        validate_len("batch support", support.len(), expected)?;
    }

    Ok(())
}

// This module is private: callers cannot introduce arbitrary expression sources.
pub trait Expression: TensorGeometry
where
    Self::DType: TensorElement,
{
    /// Supply requests for the consumer's current logical shape, if preferred.
    /// None delegates to another source or the consumer's default linear order.
    fn preferred_requests(&self, _shape: &[usize]) -> Result<Option<RequestIterator>> {
        Ok(None)
    }

    fn slice_requests(&self, slice: crate::slice::Slice) -> Result<crate::slice::Requests<'_>> {
        Ok(slice.stream())
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>>;
}

/// Owned bounded requests; expression types themselves use static dispatch.
pub(crate) type RequestIterator = Box<dyn Iterator<Item = BatchRequest> + Send>;

pub fn coordinate_blocks<E: Expression + ?Sized>(
    expression: &E,
) -> Result<crate::CoordinateBlockStream<'_, E::DType>>
where
    E::DType: TensorElement,
{
    let requests = match expression.preferred_requests(expression.shape())? {
        Some(requests) => requests,
        None => Box::new(request::linear_requests(expression.shape())?),
    };
    Ok(
        evaluation_futures(expression, futures::stream::iter(requests.map(Ok)))
            .buffer_unordered(num_cpus::get().max(1))
            .map(move |batch| {
                let (request, batch) = batch?;
                Ok((request.into_coordinates(expression.shape())?, batch.values))
            })
            .boxed(),
    )
}

pub struct Batch<T: TensorElement> {
    pub array: ArrayAccess<'static, T>,
    // One byte per coordinate in this evaluation batch, not the whole tensor.
    // Streaming batches contain at most 4096 coordinates; point reads use one.
    // None means every coordinate in this batch is supported, independently of
    // intermediate numerical zeros. Batches need not align with storage blocks.
    pub support: Option<Vec<u8>>,
}

pub fn batch_array<T: TensorElement>(values: Vec<T>) -> Result<ArrayAccess<'static, T>> {
    validate_bound("batch array", values.len())?;
    let shape = std::iter::once(values.len()).collect();

    Ok(ArrayAccess::from(Array::new(Buffer::from(values), shape)?))
}

impl<T: TensorElement> Batch<T> {
    fn validate(&self, expected: usize) -> Result<()> {
        validate_bound("batch expression", self.array.size())?;
        validate_len("batch expression", self.array.size(), expected)?;
        validate_support(&self.support, expected)
    }

    // Selection is part of the lazy expression, not a materialized intermediate.
    pub fn masked(self) -> Result<Self> {
        validate_bound("batch expression", self.array.size())?;
        validate_support(&self.support, self.array.size())?;
        let values = match &self.support {
            Some(support) if support.contains(&0) => ArrayAccess::from(
                batch_array(support.clone())?
                    .cond(self.array, batch_array(vec![T::ZERO; support.len()])?)?,
            ),
            _ => self.array,
        };

        Ok(Self {
            array: values,
            support: self.support,
        })
    }
}

/// Union original support without inspecting intermediate numerical values.
///
/// Both masks describe the same bounded coordinate batch, including when called
/// recursively by nested expressions. The result has at most 4096 bytes; this
/// helper neither reads tensor data nor materializes whole-tensor support.
pub fn union_support(left: Option<Vec<u8>>, right: Option<Vec<u8>>) -> Result<Option<Vec<u8>>> {
    for mask in [&left, &right].into_iter().flatten() {
        validate_bound("support union", mask.len())?;
    }
    match (left, right) {
        (Some(mut left), Some(right)) => {
            validate_len("support union", right.len(), left.len())?;

            for (l, r) in left.iter_mut().zip(right) {
                *l |= r;
            }

            Ok(Some(left))
        }
        _ => Ok(None),
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
    coords.validate(expression.shape())?;

    #[cfg(test)]
    crate::read_metrics::record(|m| m.slice_requests += 1);
    let batch = expression.build(coords).await?;
    batch.validate(coords.len())?;

    // Only this bounded ndarray batch is materialized, never the full tensor.
    let evaluated = EvaluatedBatch {
        values: batch.array.buffer()?.to_slice()?.into_vec(),
        support: batch.support,
    };
    evaluated.validate(coords.len())?;
    Ok(evaluated)
}

pub struct EvaluatedBatch<T> {
    pub values: Vec<T>,
    pub support: Option<Vec<u8>>,
}

impl<T> EvaluatedBatch<T> {
    fn validate(&self, expected: usize) -> Result<()> {
        validate_bound("evaluated batch", self.values.len())?;
        validate_len("evaluated batch", self.values.len(), expected)?;
        validate_support(&self.support, expected)
    }

    pub fn populated(mut self) -> Result<Vec<T>> {
        self.validate(self.values.len())?;
        if let Some(support) = self.support {
            let mut index = 0;
            self.values.retain(|_| {
                let keep = support[index] != 0;
                index += 1;
                keep
            });
        }

        Ok(self.values)
    }
}

/// Expand coordinates only at the sparse output boundary, preserving order and duplicates.
/// Both collections are bounded by the validated execution request.
pub(crate) fn sparse_elements<T: TensorElement>(
    request: BatchRequest,
    batch: EvaluatedBatch<T>,
    shape: &[usize],
) -> Result<impl Iterator<Item = Result<(Vec<u64>, T)>>> {
    batch.validate(request.len())?;
    let coords = request.into_coordinates(shape)?;
    validate_len(
        "sparse output coordinates",
        coords.len(),
        batch.values.len(),
    )?;
    Ok(coords
        .into_iter()
        .zip(batch.values)
        .filter(|(_, value)| *value != T::default())
        .map(Ok))
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
pub(crate) fn ordered_requests<'a, E, R>(
    expression: &'a E,
    requests: R,
) -> BoxStream<'a, Result<CoordinateBatch<E::DType>>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
    R: futures::Stream<Item = Result<BatchRequest>> + Send + 'a,
{
    evaluation_futures(expression, requests)
        .buffered(num_cpus::get().max(1))
        .boxed()
}

/// Build lazy evaluation futures, without buffering, spawning, or expanding
/// coordinates. Each consumer chooses delivery order at its sole buffer boundary.
pub(crate) fn evaluation_futures<'a, E, R>(
    expression: &'a E,
    requests: R,
) -> impl futures::Stream<
    Item = impl std::future::Future<Output = Result<CoordinateBatch<E::DType>>> + Send + 'a,
> + Send
+ 'a
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
    R: futures::Stream<Item = Result<BatchRequest>> + Send + 'a,
{
    requests.map(move |coords| async move {
        let coords = coords?;
        let values = evaluate_batch(expression, &coords).await?;

        Ok((coords, values))
    })
}

impl<FE, T> Expression for TensorView<'_, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn slice_requests(&self, slice: crate::slice::Slice) -> Result<crate::slice::Requests<'_>> {
        self.slice_requests_for_storage(slice)
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let values = self.read_batch(coords).await?;

            let support = if matches!(self.layout(), Layout::Sparse { .. }) {
                Some(values.iter().map(|v| u8::from(*v != T::ZERO)).collect())
            } else {
                None
            };

            Ok(Batch {
                array: batch_array(values)?,
                support,
            })
        })
    }
}

impl<FE, T> Expression for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn slice_requests(&self, slice: crate::slice::Slice) -> Result<crate::slice::Requests<'_>> {
        self.storage_slice_requests(slice, None)
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move { self.view().build(coords).await })
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::counters::Counter;

    use super::*;

    // Wrap real storage only to observe dispatch; numerical reads still delegate.
    #[derive(Clone)]
    struct Provider<'a> {
        source: &'a Tensor<crate::test_support::FsEntry, u8>,
        calls: &'a Counter,
        provide: fn(&[usize]) -> Result<Option<RequestIterator>>,
    }

    impl TensorGeometry for Provider<'_> {
        type DType = u8;

        fn dtype(&self) -> crate::NumberType {
            self.source.dtype()
        }

        fn shape(&self) -> &[usize] {
            self.source.shape()
        }

        fn layout(&self) -> Layout {
            self.source.layout()
        }
    }

    impl Expression for Provider<'_> {
        fn preferred_requests(&self, shape: &[usize]) -> Result<Option<RequestIterator>> {
            self.calls.increment();
            (self.provide)(shape)
        }

        fn build<'a>(&'a self, request: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<u8>>> {
            self.source.build(request)
        }
    }

    #[tokio::test]
    async fn request_providers_delegate_once_in_order_and_propagate_errors() {
        use crate::{TensorMath, TensorUnaryBoolean, TensorWhere};

        type Provide = fn(&[usize]) -> Result<Option<RequestIterator>>;
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
            let result = selected.preferred_requests(&[7, 3]);
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
            let result = binary.preferred_requests(&[7, 3]);
            assert_eq!(result.is_err(), fails && expected[2] == 0);
            assert_eq!(matches!(&result, Ok(Some(_))), supplies && expected[2] == 0);
            assert_eq!(
                calls.each_ref().map(|n| n.read()),
                [expected[0], expected[1], 0]
            );
        }
        crate::test_support::cleanup(&root).await;
    }

    #[test]
    fn sparse_output_preserves_order_duplicates_and_nan() {
        let request =
            BatchRequest::explicit(vec![vec![2], vec![0], vec![2], vec![1], vec![0]]).unwrap();
        let batch = EvaluatedBatch {
            values: vec![3f32, -0., 4., f32::NAN, 0.],
            support: Some(vec![1; 5]),
        };
        let output = sparse_elements(request, batch, &[3])
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(&output[..2], &[(vec![2], 3.), (vec![2], 4.)]);
        assert_eq!(output[2].0, vec![1]);
        assert!(output[2].1.is_nan());
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn sparse_output_rejects_mismatched_values_and_support() {
        for batch in [
            EvaluatedBatch {
                values: vec![1u8, 2],
                support: None,
            },
            EvaluatedBatch {
                values: vec![1u8],
                support: Some(vec![]),
            },
        ] {
            assert!(matches!(
                sparse_elements(BatchRequest::point(&[0]), batch, &[1]),
                Err(Error::InvalidLayout(_))
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
            array: batch_array(vec![1u8, 2]).unwrap(),
            support: None,
        };
        assert!(batch.validate(1).is_err());
        let batch = Batch {
            array: batch_array(vec![1u8, 2]).unwrap(),
            support: Some(vec![1]),
        };
        assert!(batch.validate(2).is_err());
        assert!(batch.masked().is_err());
        let evaluated = EvaluatedBatch {
            values: vec![1u8],
            support: None,
        };
        assert!(evaluated.validate(2).is_err());
        let evaluated = EvaluatedBatch {
            values: vec![1u8],
            support: Some(vec![]),
        };
        assert!(evaluated.populated().is_err());
        assert!(union_support(Some(vec![1]), Some(vec![1, 1])).is_err());
        assert!(union_support(None, Some(vec![1; MAX_BATCH_ELEMENTS + 1])).is_err());
    }

    #[test]
    fn filtering_reuses_the_owned_allocation() {
        let values = vec![2u8, 3, 4, 5];
        let pointer = values.as_ptr();
        let values = EvaluatedBatch {
            values,
            support: Some(vec![0, 1, 0, 1]),
        }
        .populated()
        .unwrap();
        assert_eq!(values, [3, 5]);
        assert_eq!(values.as_ptr(), pointer);
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
