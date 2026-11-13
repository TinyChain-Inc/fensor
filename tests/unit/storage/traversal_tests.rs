use number_general::DType;

use super::*;
use crate::Shape;

#[test]
fn block_consumers_share_order_padding_and_rank_scratch() {
    assert!(TensorSchema::new(f64::dtype(), Shape::new()).is_err());
    for shape in [
        vec![1],
        vec![5],
        vec![6],
        vec![2, 3],
        vec![3, 5],
        vec![3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ] {
        for capacity in [1, 3, 7] {
            for prefix in 0..=shape.len() {
                let layout = if prefix == 0 {
                    Layout::Dense
                } else {
                    Layout::Sparse {
                        axis: Some(prefix - 1),
                    }
                };

                let geometry = StorageGeometry::new(
                    TensorSchema::new(f64::dtype(), shape.clone().into()).unwrap(),
                    layout,
                    capacity,
                )
                .unwrap();

                let mut flat = 0;
                for id in 0..geometry.block_count() {
                    let range = geometry.block_range(id).unwrap();
                    assert_eq!(range.start, flat);
                    let len = geometry.block_len(id).unwrap();
                    for value in [-0., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                        geometry.validate_block(id, &vec![value; len]).unwrap();
                        assert!(geometry.validate_block(id, &vec![value; len + 1]).is_err());
                        assert!(geometry.validate_block(id, &vec![value; len - 1]).is_err());
                    }

                    flat = range.end;
                }

                assert_eq!(flat, shape.iter().product::<u64>());
                assert!(geometry.validate_id(geometry.block_count()).is_err());
            }
        }
    }
}

#[test]
fn block_validation_handles_u64_edge_without_overflow() {
    let geometry = StorageGeometry::new(
        TensorSchema::new(f64::dtype(), smallvec::smallvec![u64::MAX]).unwrap(),
        Layout::Dense,
        [2].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let last = geometry.block_count() - 1;

    assert_eq!(geometry.block_range(last).unwrap(), u64::MAX - 1..u64::MAX);
    geometry
        .validate_block(last - 1, &[f64::NAN, f64::INFINITY])
        .unwrap();

    for value in [1.0, f64::NAN, f64::INFINITY, -0.0] {
        geometry.validate_block(last, &[value]).unwrap();
        assert!(matches!(geometry.validate_block(last, &[f64::NAN, value]),
            Err(Error::InvalidLayout(message)) if message == "invalid logical block length"));
    }
}

#[tokio::test]
async fn preferred_regions_skip_gaps_and_include_sparse_zeros() {
    ready_candidates_yield_and_release().await;

    use futures::TryStreamExt;

    use crate::test_support::{FsEntry, new_dir};
    use crate::{
        Tensor, TensorExpression, TensorGeometry, TensorRead, TensorTransform, TensorWrite,
    };

    let (_root, dir) = new_dir("preferred_regions").await;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), smallvec::smallvec![1_000_000_001, 3]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        2,
    )
    .await
    .unwrap();

    let view = tensor
        .view()
        .slice(smallvec::smallvec![
            crate::AxisRange::In(0, 1_000_000_001, 1_000_000_000),
            crate::AxisRange::In(0, 3, 1),
        ])
        .unwrap()
        .transpose(None)
        .unwrap();

    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let requests = crate::expression::traversal::preferred(&view, view.shape())
                .unwrap()
                .unwrap();

            let mut actual = Vec::new();
            for request in requests {
                actual.extend(request.into_coordinates(view.shape()).unwrap());
            }

            let mut sorted = actual.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                crate::schema::row_major_coords(view.shape())
                    .unwrap()
                    .collect::<Vec<_>>()
            );

            assert_ne!(actual, sorted);
            assert_eq!(
                actual,
                vec![
                    vec![0, 0],
                    vec![1, 0],
                    vec![2, 0],
                    vec![0, 1],
                    vec![1, 1],
                    vec![2, 1]
                ]
            );

            // Linear block traversal needs no rectangular block-bounds allocation.
            crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
        })
        .await;

    let mut blocks = view.read_coordinate_blocks().unwrap();
    let mut count = 0;
    while let Some((coordinates, values)) = blocks.try_next().await.unwrap() {
        assert_eq!(coordinates.len(), values.len());
        assert!(values.iter().all(|&value| value == 0.));
        count += values.len();
    }

    assert_eq!(count, 6);

    let identity = TensorExpression::new(tensor.view().transpose(None).unwrap())
        .unwrap()
        .transpose(None)
        .unwrap();

    crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
        assert!(
            crate::expression::traversal::preferred(&identity, identity.shape())
                .unwrap()
                .is_none()
        );

        crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
    });

    for consume in [false, true] {
        let lease = std::sync::Arc::new(());
        let released = std::sync::Arc::downgrade(&lease);
        let source = PlanningSource {
            tensor: tensor.clone(),
            lease,
        };

        let slice = crate::slice::Slice::full(source.shape()).unwrap();
        let mut requests = slice_requests(&source, slice, None).unwrap();
        drop(source);
        assert!(released.upgrade().is_some());
        if consume {
            assert!(requests.try_next().await.unwrap().is_none());
            assert!(released.upgrade().is_none());
        }

        drop(requests);
        assert!(released.upgrade().is_none());
    }

    // Matching axis positions alone do not make a smaller selection an identity.
    let first_row = tensor
        .view()
        .slice(smallvec::smallvec![
            crate::AxisRange::In(0, 1, 1),
            crate::AxisRange::In(0, 3, 1),
        ])
        .unwrap();

    crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
        let requests = crate::expression::traversal::preferred(&first_row, first_row.shape())
            .unwrap()
            .unwrap();

        let actual: Vec<_> = requests
            .flat_map(|request| request.into_coordinates(first_row.shape()).unwrap())
            .collect();

        assert_eq!(actual, vec![vec![0, 0], vec![0, 1], vec![0, 2]]);
        crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
    });

    tensor.write_value(&[0, 0], 7.).await.unwrap();
    let repeated = crate::slice::Slice::new(
        tensor.shape(),
        vec![
            crate::request::Axis::selected(vec![0; 10_000]),
            crate::request::Axis::range(0, 1),
        ],
    )
    .unwrap();

    let mut requests = slice_requests(&tensor, repeated, None).unwrap();
    let mut count = 0;
    while let Some(request) = requests.try_next().await.unwrap() {
        assert!(request.len() <= crate::MAX_BATCH_ELEMENTS);
        let values = tensor.read_batch(&request, None).await.unwrap();
        assert!(values.iter().all(|&value| value == 7.));
        count += values.len();
    }

    assert_eq!(count, 10_000);
}

// A native setup borrow must not duplicate the source; the occupied stream owns
// its independent lease, including before its first poll.
struct PlanningSource {
    tensor: crate::Tensor<crate::test_support::FsEntry, f64>,
    lease: std::sync::Arc<()>,
}

impl Clone for PlanningSource {
    fn clone(&self) -> Self {
        panic!("sparse planning cloned its borrowed source")
    }
}

impl crate::TensorGeometry for PlanningSource {
    type DType = f64;

    fn dtype(&self) -> crate::NumberType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }

    fn shape(&self) -> &[u64] {
        self.tensor.shape()
    }
}

impl crate::TensorArray for PlanningSource {
    fn schema(&self) -> &TensorSchema {
        self.tensor.schema()
    }

    fn strides(&self) -> &[u64] {
        self.tensor.strides()
    }
}

impl TensorSource for PlanningSource {
    fn storage_geometry(&self) -> StorageGeometry {
        self.tensor.storage_geometry()
    }

    fn read_logical_block(&self, id: u64) -> crate::BoxFuture<'_, Result<Vec<f64>>> {
        self.tensor.read_logical_block(id)
    }

    fn occupied_blocks(&self, range: std::ops::Range<u64>) -> BoxStream<'static, Result<u64>> {
        let lease = self.lease.clone();
        let keys = self.tensor.occupied_blocks(range);
        Box::pin(keys.map(move |key| {
            let _ = &lease;
            key
        }))
    }
}

struct ReadyCandidates {
    geometry: StorageGeometry,
    inspected: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    lease: std::sync::Arc<()>,
    fail_at: Option<usize>,
}

impl Clone for ReadyCandidates {
    fn clone(&self) -> Self {
        panic!("candidate planning must borrow its source")
    }
}

impl crate::TensorGeometry for ReadyCandidates {
    type DType = f64;
    fn dtype(&self) -> crate::NumberType {
        f64::dtype()
    }

    fn layout(&self) -> Layout {
        self.geometry.layout()
    }

    fn shape(&self) -> &[u64] {
        self.geometry.schema().shape()
    }
}

impl crate::TensorArray for ReadyCandidates {
    fn schema(&self) -> &TensorSchema {
        self.geometry.schema()
    }

    fn strides(&self) -> &[u64] {
        self.geometry.schema().strides()
    }
}

impl TensorSource for ReadyCandidates {
    fn storage_geometry(&self) -> StorageGeometry {
        self.geometry.clone()
    }

    fn read_logical_block(&self, _: u64) -> crate::BoxFuture<'_, Result<Vec<f64>>> {
        Box::pin(async { panic!("candidate planning must not read payloads") })
    }

    fn occupied_blocks(&self, range: std::ops::Range<u64>) -> BoxStream<'static, Result<u64>> {
        let lease = self.lease.clone();
        let inspected = self.inspected.clone();
        let last = self.geometry.block_count() - 2;
        let fail_at = self.fail_at;
        futures::stream::iter(
            range
                .filter(move |&id| id == 0 || id % 2 == 1 || id == last)
                .enumerate()
                .map(move |(index, id)| {
                    let _lease = &lease;
                    inspected.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if Some(index) == fail_at {
                        Err(Error::InvalidCoord("ready candidate failure".into()))
                    } else {
                        Ok(id)
                    }
                }),
        )
        .boxed()
    }
}

#[derive(Default)]
struct CandidateWake(std::sync::atomic::AtomicUsize);

impl std::task::Wake for CandidateWake {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn ready_candidates_yield_and_release() {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    let rows = (crate::MAX_BATCH_ELEMENTS * 2) as u64;
    for mode in 0..4 {
        let inspected = Arc::new(AtomicUsize::new(0));
        let lease = Arc::new(());
        let released = Arc::downgrade(&lease);
        let source = ReadyCandidates {
            geometry: StorageGeometry::new(
                TensorSchema::new(f64::dtype(), vec![rows, 2].into()).unwrap(),
                Layout::Sparse { axis: None },
                1,
            )
            .unwrap(),
            inspected: inspected.clone(),
            lease,
            fail_at: (mode == 3).then_some(513),
        };

        let slice = crate::slice::Slice::new(
            source.geometry.schema().shape(),
            vec![
                crate::request::Axis::range(0, rows),
                crate::request::Axis::range(0, 1),
            ],
        )
        .unwrap();

        let mut requests = slice_requests(&source, slice, None).unwrap();
        drop(source);
        assert!(released.upgrade().is_some());
        if mode == 0 {
            drop(requests);
            assert_eq!(inspected.load(Ordering::Relaxed), 0);
            assert!(released.upgrade().is_none());
            continue;
        }

        // Start the manual poll in a fresh task budget, without timing assumptions.
        tokio::task::yield_now().await;
        let wakes = Arc::new(CandidateWake::default());
        let waker = Waker::from(wakes.clone());
        let mut cx = Context::from_waker(&waker);
        {
            let mut next = Box::pin(requests.try_next());
            assert!(matches!(next.as_mut().poll(&mut cx), Poll::Pending));
        }

        let count = inspected.load(Ordering::Relaxed);
        assert!(count > 0 && count < rows as usize + 1);
        // Tokio defers cooperative wakes until control returns to its scheduler.
        tokio::task::yield_now().await;
        assert!(wakes.0.load(Ordering::Relaxed) > 0);
        assert!(released.upgrade().is_some());

        if mode == 1 {
            drop(requests);
            assert!(released.upgrade().is_none());
        } else if mode == 3 {
            assert!(matches!(requests.try_next().await,
                Err(Error::InvalidCoord(message)) if message == "ready candidate failure"));

            assert_eq!(inspected.load(Ordering::Relaxed), 514);
            assert!(released.upgrade().is_none());
        } else {
            let mut coordinates = Vec::new();
            while let Some(request) = requests.try_next().await.unwrap() {
                assert!(request.len() <= crate::MAX_BATCH_ELEMENTS);
                coordinates.extend(request.into_coordinates(&[rows, 2]).unwrap());
            }

            assert_eq!(coordinates, [vec![0, 0], vec![rows - 1, 0]]);
            assert_eq!(inspected.load(Ordering::Relaxed), rows as usize + 1);
            assert!(released.upgrade().is_none());
        }
    }
}
