use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::read_metrics::{CURRENT, Metrics};

#[tokio::test]
async fn wide_sparse_candidates_retain_compact_requests_and_release_sources() {
    for (leaves, rank) in [(1, 1), (129, 9)] {
        let mut shape = crate::Shape::from_elem(1, rank);
        shape[rank - 1] = MAX_BATCH_ELEMENTS as u64 + 1;
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let reads = Arc::new(AtomicUsize::new(0));
        let streams = (0..leaves)
            .map(|_| {
                let lease = Arc::clone(&lease);
                let reads = Arc::clone(&reads);
                futures::stream::iter([
                    Ok(BatchRequest::linear(0, 0).unwrap()),
                    Ok(BatchRequest::linear(0, MAX_BATCH_ELEMENTS).unwrap()),
                    Ok(BatchRequest::linear(MAX_BATCH_ELEMENTS as u64, 1).unwrap()),
                ])
                .map(move |item| {
                    let _ = &lease;
                    reads.fetch_add(1, Ordering::Relaxed);
                    item
                })
                .boxed()
            })
            .collect();
        drop(lease);
        CURRENT
            .scope(RefCell::new(Metrics::default()), async {
                let mut stream = merge(streams, shape.clone());
                let mut count = 0;

                while let Some(request) = stream.try_next().await.unwrap() {
                    assert!(request.len() <= MAX_BATCH_ELEMENTS);
                    let mut cursor = request.cursor(&shape).unwrap();
                    let mut coord = Coord::new();

                    while cursor.next_into(&mut coord) {
                        assert!(coord[..rank - 1].iter().all(|&n| n == 0));
                        assert_eq!(coord[rank - 1], count);
                        count += 1;
                    }
                }
                assert_eq!(count, MAX_BATCH_ELEMENTS as u64 + 1);
                assert_eq!(reads.load(Ordering::Relaxed), leaves * 3);
                assert_eq!(CURRENT.with(|m| m.borrow().expanded_coordinates), 0);
                // Exhaustion releases sources even while the empty consumer survives.
                assert!(weak.upgrade().is_none());
            })
            .await;
    }
}

#[tokio::test]
async fn ready_sparse_candidates_can_be_cancelled_and_errors_release_sources() {
    for fail in [false, true] {
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let streams = (0..257)
            .map(|index| {
                let lease = Arc::clone(&lease);
                futures::stream::iter([if fail && index == 256 {
                    Err(Error::InvalidLayout("candidate source failure".into()))
                } else {
                    Ok(BatchRequest::linear(0, MAX_BATCH_ELEMENTS).unwrap())
                }])
                .map(move |item| {
                    let _ = &lease;
                    item
                })
                .boxed()
            })
            .collect();
        drop(lease);
        let mut stream = merge(
            streams,
            crate::Shape::from_slice(&[MAX_BATCH_ELEMENTS as u64]),
        );
        if fail {
            assert!(
                matches!(stream.try_next().await, Err(Error::InvalidLayout(message)) if message == "candidate source failure")
            );
            assert!(weak.upgrade().is_none());
        } else {
            {
                let mut next = std::pin::pin!(stream.try_next());
                assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
                assert!(weak.upgrade().is_some());
            }
            drop(stream);
            assert!(weak.upgrade().is_none());
        }
    }
}

#[tokio::test]
async fn candidate_errors_discard_partial_batches_and_empty_inputs_yield() {
    for next in [
        Err(Error::InvalidLayout("candidate failure".into())),
        Ok(BatchRequest::explicit(vec![vec![2]]).unwrap()),
    ] {
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let source = futures::stream::iter([Ok(BatchRequest::linear(0, 1).unwrap()), next])
            .map(move |item| {
                let _ = &lease;
                item
            })
            .boxed();
        let mut stream = merge(vec![source], crate::Shape::from_slice(&[2]));
        assert!(stream.try_next().await.is_err());
        assert!(weak.upgrade().is_none());
        assert!(stream.try_next().await.unwrap().is_none());
    }

    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let source = futures::stream::repeat_with(move || {
        let _ = &lease;
        Ok(BatchRequest::linear(0, 0).unwrap())
    })
    .boxed();
    let mut stream = merge(vec![source], crate::Shape::from_slice(&[2]));
    assert!(matches!(futures::poll!(stream.try_next()), Poll::Pending));
    assert!(weak.upgrade().is_some());
    drop(stream);
    assert!(weak.upgrade().is_none());
}
