//! Exercise sequential request consumption and cancellation over stored values.

use std::sync::Mutex;

use futures::{TryStreamExt, channel::oneshot};
use number_general::DType;

use super::*;
use crate::test_support::{self, FsEntry, counters::Counter};
use crate::{Layout, TensorFileEntry, TensorRead, TensorReduceAll, TensorSchema, TensorWrite};

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

impl<T: TensorElement> crate::expression::traversal::Plan for Source<'_, T>
where
    FsEntry: TensorFileEntry<T>,
{
    fn preferred_step<'a>(&'a self, shape: &'a [u64]) -> Result<traversal::Preferred<'a>> {
        Ok(traversal::Preferred::Ready(Some(Box::new(
            crate::schema::row_major_coords(shape)?
                .map(|coord| BatchRequest::explicit(vec![coord]).unwrap()),
        ))))
    }

    fn selection_step(&self, slice: crate::slice::Slice) -> Result<traversal::Selection<'_>> {
        // Test-only singleton batches preserve the validated slice selection.
        let requests = slice.requests().flat_map(|request| {
            request
                .into_coordinates(self.shape())
                .unwrap()
                .into_iter()
                .map(|coord| BatchRequest::explicit(vec![coord]))
        });
        Ok(traversal::Selection::Ready(
            futures::stream::iter(requests).boxed(),
        ))
    }
}

impl<T: TensorElement> Expression for Source<'_, T>
where
    FsEntry: TensorFileEntry<T>,
{
    fn build<'a>(
        &'a self,
        context: super::Context<'a>,
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let index = self.started.increment();
            let _finished = Finish(&self.finished);
            if index == 0 {
                let wait = self.first.lock().unwrap().take().unwrap();
                wait.await.unwrap()?;
            } else if self.fail_at == Some(index) {
                return Err(Error::Unsupported("injected later batch failure".into()));
            }

            let batch = self.tensor.build(context, request).await?;
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
async fn sequential_batches_are_ordered_and_cancelled_by_drop() {
    let limit = 1;
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

    for (owned, cancel) in [(false, false), (false, true), (true, false), (true, true)] {
        let (source, release) = delayed(&tensor);
        let source = std::sync::Arc::new(source);
        let weak = std::sync::Arc::downgrade(&source);
        let requests = (0..count).map(|i| BatchRequest::explicit(vec![vec![i as u64]]).unwrap());
        let mut stream = if owned {
            evaluate_requests(source.clone(), futures::stream::iter(requests.map(Ok)))
        } else {
            ordered_batches(&*source, requests)
        };
        // A blocked request prevents evaluation of the next request.
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
            drop(stream);
        }
        drop(source);
        assert!(
            weak.upgrade().is_none(),
            "stream retained its source after drop"
        );
    }

    test_support::cleanup(&root).await;
}

async fn stored<T: TensorElement>(
    name: &str,
    values: &[T],
    sparse: bool,
) -> (crate::test_support::Directory, Tensor<FsEntry, T>)
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
async fn sequential_consumption_preserves_pairs_and_backpressure() {
    use futures::FutureExt;

    let count = 8;
    let values: Vec<_> = (0..count).map(|i| i as u8).collect();
    let (root, tensor) = stored("sequential_pairs", &values, false).await;
    let (source, release) = delayed(&tensor);
    let mut stream = source.read_coordinate_blocks().unwrap();
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(source.started.read(), 1);
    assert_eq!(source.completed.read(), 0);
    release.send(Ok(())).unwrap();

    for (i, value) in values.iter().enumerate() {
        let (coords, batch) = stream.try_next().await.unwrap().unwrap();
        assert_eq!(coords, vec![vec![i as u64]]);
        assert_eq!(batch, vec![*value]);
        assert_eq!(source.started.read(), (i + 1) as u64);
    }

    assert!(stream.try_next().await.unwrap().is_none());
    assert_eq!(source.finished.read(), count as u64);
    drop(stream);

    let (source, release) = delayed(&tensor);
    let mut stream = source.read_coordinate_blocks().unwrap();
    assert!(stream.next().now_or_never().is_none());
    drop(stream);
    assert_eq!(source.finished.read(), 1);
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
async fn sequential_errors_stop_before_later_evaluation() {
    use futures::FutureExt;

    let (root, tensor) = stored("sequential_errors", &[1u8, 2, 3], false).await;
    let (mut source, release) = delayed(&tensor);
    source.fail_at = Some(1);
    let requests = (0..3).map(|i| BatchRequest::explicit(vec![vec![i]]));
    let mut stream = evaluate_requests(&source, futures::stream::iter(requests));
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(source.started.read(), 1);
    release.send(Ok(())).unwrap();
    assert_eq!(stream.try_next().await.unwrap().unwrap().1.values, vec![1]);
    assert!(matches!(
        stream.try_next().await,
        Err(Error::Unsupported(_))
    ));
    drop(stream);
    assert_eq!(source.started.read(), 2);
    assert_eq!(source.finished.read(), 2);

    let (mut source, release) = delayed(&tensor);
    source.fail_at = Some(1);
    release.send(Ok(())).unwrap();
    assert!(matches!(source.min_all().await, Err(Error::Unsupported(_))));
    assert_eq!(source.started.read(), 2);
    assert_eq!(source.finished.read(), 2);
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

    if !release_first {
        assert!(futures::poll!(terminal.as_mut()).is_pending());
        assert_eq!(source.started.read(), 1);
        assert_eq!(source.completed.read(), 0);
    }

    release.send(Ok(())).unwrap();
    terminal.await
}

#[tokio::test]
async fn numeric_terminals_preserve_values_across_input_delays() {
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

    // Compare delayed and ready consumption with the real aggregate gamma bound.
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

#[tokio::test]
async fn owned_stream_error_releases_pending_evaluation_on_drop() {
    let (root, tensor) = stored("owned_error", &[1u8, 2], false).await;
    let (source, release) = delayed(&tensor);
    let source = std::sync::Arc::new(source);
    let weak = std::sync::Arc::downgrade(&source);
    let requests = futures::stream::iter([Ok(BatchRequest::point(&[0]))]);
    let mut stream = evaluate_requests(source, requests);
    release
        .send(Err(Error::InvalidLayout("test read failure".into())))
        .unwrap();
    assert!(matches!(
        stream.try_next().await,
        Err(Error::InvalidLayout(_))
    ));
    drop(stream);
    assert!(weak.upgrade().is_none());
    test_support::cleanup(&root).await;
}
