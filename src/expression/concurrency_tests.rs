//! Exercise the real outer buffer with controlled readiness over stored values.

use std::sync::Mutex;

use futures::{TryStreamExt, channel::oneshot};
use number_general::DType;

use crate::test_support::{self, FsEntry, counters::Counter};
use crate::{TensorFileEntry, TensorRead, TensorReduceAll, TensorSchema, TensorWrite};

use super::*;

struct Source<'a, T: TensorElement>
where
    FsEntry: TensorFileEntry<T>,
{
    tensor: &'a Tensor<FsEntry, T>,
    first: Mutex<Option<oneshot::Receiver<Result<()>>>>,
    fail_at: Option<u64>,
    started: Counter,
    finished: Counter,
    completed: Counter,
}

struct Finish<'a>(&'a Counter);

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.increment();
    }
}

impl<T: TensorElement> TensorGeometry for Source<'_, T>
where
    FsEntry: TensorFileEntry<T>,
{
    type DType = T;

    fn dtype(&self) -> crate::NumberType {
        self.tensor.dtype()
    }

    fn shape(&self) -> &[u64] {
        self.tensor.shape()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }
}

impl<T: TensorElement> Expression for Source<'_, T>
where
    FsEntry: TensorFileEntry<T>,
{
    fn preferred_requests(&self, shape: &[u64]) -> Result<Option<RequestIterator>> {
        Ok(Some(Box::new(crate::schema::row_major_coords(shape)?.map(
            |coord| BatchRequest::explicit(vec![coord]).unwrap(),
        ))))
    }

    fn slice_requests(&self, slice: crate::slice::Slice) -> Result<crate::slice::Requests<'_>> {
        // Test-only singleton batches preserve the validated slice selection.
        let requests = slice.requests().flat_map(|request| {
            request
                .into_coordinates(self.shape())
                .unwrap()
                .into_iter()
                .map(|coord| BatchRequest::explicit(vec![coord]))
        });
        Ok(futures::stream::iter(requests).boxed())
    }

    fn build<'a>(&'a self, request: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let index = self.started.increment();
            let _finished = Finish(&self.finished);
            if index == 0 {
                let wait = self.first.lock().unwrap().take().unwrap();
                wait.await.unwrap()?;
            } else if self.fail_at == Some(index) {
                return Err(Error::Unsupported("injected later batch failure".into()));
            }
            let batch = self.tensor.build(request).await?;
            self.completed.increment();
            Ok(batch)
        })
    }
}

impl<T: TensorElement> TensorRead for Source<'_, T>
where
    FsEntry: TensorFileEntry<T>,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<T>> {
        self.tensor.read_value(coord)
    }

    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, T>> {
        coordinate_blocks(self)
    }
}

fn delayed<T: TensorElement>(
    tensor: &Tensor<FsEntry, T>,
) -> (Source<'_, T>, oneshot::Sender<Result<()>>)
where
    FsEntry: TensorFileEntry<T>,
{
    let (release, wait) = oneshot::channel();
    (
        Source {
            tensor,
            first: Mutex::new(Some(wait)),
            fail_at: None,
            started: Counter::new(),
            finished: Counter::new(),
            completed: Counter::new(),
        },
        release,
    )
}

#[tokio::test]
async fn buffered_batches_are_bounded_ordered_and_cancelled_by_drop() {
    let limit = num_cpus::get().max(1);
    let count = limit + 2;
    let (root, dir) = test_support::new_dir("ordered_batches").await;
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(u8::dtype(), vec![count as u64].into()).unwrap(),
        Layout::Dense,
        MAX_BATCH_ELEMENTS,
    )
    .await
    .unwrap();

    for i in 0..count {
        tensor
            .write_value(&[i as u64], (i % 255) as u8)
            .await
            .unwrap();
    }

    for cancel in [false, true] {
        let (source, release) = delayed(&tensor);
        let requests = (0..count).map(|i| BatchRequest::explicit(vec![vec![i as u64]]).unwrap());
        let mut stream = ordered_batches(&source, requests);
        // Tokio's cooperative I/O budget may yield before the window finishes.
        futures::future::poll_fn(|cx| {
            assert!(stream.as_mut().poll_next(cx).is_pending());
            assert!(source.started.read() <= limit as u64);
            if source.completed.read() == (limit - 1) as u64 {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        })
        .await;
        assert_eq!(source.started.read(), limit as u64);
        assert_eq!(source.completed.read(), (limit - 1) as u64);

        if cancel {
            drop(stream);
            assert_eq!(source.finished.read(), source.started.read());
            assert!(
                release.send(Ok(())).is_err(),
                "pending evaluation was dropped"
            );
        } else {
            release.send(Ok(())).unwrap();
            for i in 0..count {
                let (_, batch) = stream.try_next().await.unwrap().unwrap();
                assert_eq!(batch.values, vec![(i % 255) as u8]);
            }
            assert!(stream.try_next().await.unwrap().is_none());
            assert_eq!(source.started.read(), count as u64);
            assert_eq!(source.finished.read(), count as u64);
        }
    }

    test_support::cleanup(&root).await;
}

async fn stored<T: TensorElement>(
    name: &str,
    values: &[T],
    sparse: bool,
) -> (std::path::PathBuf, Tensor<FsEntry, T>)
where
    FsEntry: TensorFileEntry<T>,
{
    let (root, dir) = test_support::new_dir(name).await;
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(T::dtype(), vec![values.len() as u64].into()).unwrap(),
        if sparse {
            Layout::Sparse { axis: None }
        } else {
            Layout::Dense
        },
        7,
    )
    .await
    .unwrap();

    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.unwrap();
    }

    (root, tensor)
}

#[tokio::test]
async fn completion_order_replenishes_slots_and_preserves_pairs() {
    use futures::FutureExt;

    let count = num_cpus::get().max(2) + 3;
    let values: Vec<_> = (0..count).map(|i| (i % 255) as u8).collect();
    let (root, tensor) = stored("unordered_slots", &values, false).await;
    for window in [1, 2, num_cpus::get().max(1)] {
        let (source, release) = delayed(&tensor);
        let requests = (0..count).map(|i| BatchRequest::explicit(vec![vec![i as u64]]));
        let mut stream = evaluation_futures(&source, futures::stream::iter(requests))
            .buffer_unordered(window)
            .boxed();
        let mut seen = Vec::new();
        if window == 1 {
            assert!(stream.next().now_or_never().is_none());
            assert_eq!(source.started.read(), 1);
        } else {
            for _ in 1..count {
                let (request, batch) = stream.try_next().await.unwrap().unwrap();
                let coords = request.into_coordinates(source.shape()).unwrap();
                let index = coords[0][0] as usize;
                assert_ne!(index, 0);
                assert_eq!(batch.values, vec![values[index]]);
                assert!(source.started.read() - source.finished.read() <= window as u64);
                seen.push(index);
            }
            assert_eq!(
                source.started.read(),
                count as u64,
                "completed slots admit later reads"
            );
            assert!(stream.next().now_or_never().is_none());
        }
        release.send(Ok(())).unwrap();
        while let Some((request, batch)) = stream.try_next().await.unwrap() {
            let index = request.into_coordinates(source.shape()).unwrap()[0][0] as usize;
            assert_eq!(batch.values, vec![values[index]]);
            seen.push(index);
        }
        assert!(stream.try_next().await.unwrap().is_none());
        assert_eq!(source.started.read(), count as u64);
        assert_eq!(source.finished.read(), count as u64);
        seen.sort();
        assert_eq!(seen, (0..count).collect::<Vec<_>>());
    }

    // Exercise the public coordinate consumer and its cancellation, independently
    // of the fixed-window helper checks above.
    let (source, release) = delayed(&tensor);
    let mut stream = source.read_coordinate_blocks().unwrap();
    if num_cpus::get() > 1 {
        let (coords, batch) = stream.try_next().await.unwrap().unwrap();
        assert_ne!(coords[0], vec![0]);
        assert_eq!(batch[0], values[coords[0][0] as usize]);
    } else {
        assert!(stream.next().now_or_never().is_none());
    }
    drop(stream);
    assert_eq!(source.finished.read(), source.started.read());
    assert!(release.send(Ok(())).is_err());

    // Separate descriptions can consume the same stored source concurrently.
    let (left, left_release) = delayed(&tensor);
    let (right, right_release) = delayed(&tensor);
    left_release.send(Ok(())).unwrap();
    right_release.send(Ok(())).unwrap();
    let (left, right) = futures::try_join!(
        left.read_coordinate_blocks()
            .unwrap()
            .try_collect::<Vec<_>>(),
        right
            .read_coordinate_blocks()
            .unwrap()
            .try_collect::<Vec<_>>()
    )
    .unwrap();

    for blocks in [left, right] {
        let mut pairs: Vec<_> = blocks
            .into_iter()
            .flat_map(|(coords, values)| coords.into_iter().zip(values))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(pairs.len(), count);
        for (i, (coord, value)) in pairs.into_iter().enumerate() {
            assert_eq!(coord, vec![i as u64]);
            assert_eq!(value, values[i]);
        }
    }

    test_support::cleanup(&root).await;
}

#[tokio::test]
async fn unordered_errors_cancel_pending_evaluation() {
    let (root, tensor) = stored("unordered_errors", &[1u8, 2, 3], false).await;
    // Use an explicit two-slot window to prove a later error can overtake an
    // unresolved first read, even on a one-CPU test runner.
    let (mut source, release) = delayed(&tensor);
    source.fail_at = Some(1);
    let requests = (0..3).map(|i| BatchRequest::explicit(vec![vec![i]]));
    let result = evaluation_futures(&source, futures::stream::iter(requests))
        .buffer_unordered(2)
        .try_collect::<Vec<_>>()
        .await;
    assert!(matches!(result, Err(Error::Unsupported(_))));
    assert_eq!(source.finished.read(), source.started.read());
    assert!(release.send(Ok(())).is_err());

    let (mut source, release) = delayed(&tensor);
    source.fail_at = Some(1);
    if num_cpus::get() == 1 {
        release.send(Ok(())).unwrap();
    } else {
        let result = source.min_all().await;
        assert!(matches!(result, Err(Error::Unsupported(_))));
        assert!(release.send(Ok(())).is_err());
        assert_eq!(source.finished.read(), source.started.read());
        test_support::cleanup(&root).await;
        return;
    }
    assert!(matches!(source.min_all().await, Err(Error::Unsupported(_))));
    test_support::cleanup(&root).await;
}

async fn scheduled<T: TensorElement + ha_ndarray::Real>(
    source: &Source<'_, T>,
    release: oneshot::Sender<Result<()>>,
    operation: usize,
    release_first: bool,
) -> Result<T>
where
    FsEntry: TensorFileEntry<T>,
{
    let mut terminal = match operation {
        0 => source.sum_all(),
        1 => source.product_all(),
        2 => source.min_all(),
        _ => source.max_all(),
    };

    if !release_first && num_cpus::get() > 1 {
        futures::future::poll_fn(|cx| {
            assert!(terminal.as_mut().poll(cx).is_pending());
            if source.completed.read() == (source.shape()[0] - 1) {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        })
        .await;
    }
    release.send(Ok(())).unwrap();
    terminal.await
}

#[tokio::test]
async fn numeric_terminals_accept_completion_schedules() {
    use num_rational::BigRational;
    use num_traits::ToPrimitive;

    for sparse in [false, true] {
        let (root, tensor) = stored("unordered_integer", &[255u8, 2, 3], sparse).await;
        for release_first in [false, true] {
            for (operation, expected) in [4, 250, 2, 255].into_iter().enumerate() {
                let (source, release) = delayed(&tensor);
                assert_eq!(
                    scheduled(&source, release, operation, release_first)
                        .await
                        .unwrap(),
                    expected
                );
            }
        }
        test_support::cleanup(&root).await;
    }

    // Exact binary-rational inputs. Cancellation can change the sum's last bits;
    // compare both schedules with the established real aggregate gamma bound.
    let values = [1e16f64, -1e16, 1.];
    let exact: BigRational = values
        .into_iter()
        .map(|v| BigRational::from_float(v).unwrap())
        .sum();
    let scale = values.iter().map(|v| v.abs()).sum::<f64>();
    let ku = 8. * values.len() as f64 * (f64::EPSILON / 2.);
    let (root, tensor) = stored("unordered_float", &values, false).await;
    for release_first in [false, true] {
        let (source, release) = delayed(&tensor);
        let actual = scheduled(&source, release, 0, release_first).await.unwrap();
        assert!((actual - exact.to_f64().unwrap()).abs() <= ku / (1. - ku) * scale);
        for (operation, expected) in [(2, -1e16), (3, 1e16)] {
            let (source, release) = delayed(&tensor);
            assert_eq!(
                scheduled(&source, release, operation, release_first)
                    .await
                    .unwrap(),
                expected
            );
        }
    }

    test_support::cleanup(&root).await;

    for values in [
        [1.25f32, 0.75, 1.5],
        [-0., 0., 1.],
        [1., f32::NAN, f32::INFINITY],
    ] {
        let (root, tensor) = stored("unordered_float_edges", &values, false).await;
        for release_first in [false, true] {
            for operation in 0..4 {
                let (source, release) = delayed(&tensor);
                let actual = scheduled(&source, release, operation, release_first)
                    .await
                    .unwrap();
                if values[1].is_nan() {
                    assert!(actual.is_nan());
                } else if values[0] == 0. {
                    let expected = [1., -0., -0., 1.][operation];
                    assert_eq!(actual.to_bits(), f32::to_bits(expected));
                } else {
                    assert_eq!(actual, [3.5, 1.40625, 0.75, 1.5][operation]);
                }
            }
        }
        test_support::cleanup(&root).await;
    }
}
