//! Shallow expression traversal. Deferred steps retain borrows, never nested consumers.

use std::sync::Arc;
use std::task::{Poll, ready};

use collate::Collator;
use futures::{StreamExt, TryStreamExt};

use super::{Expression, MAX_BATCH_ELEMENTS, RequestIterator};
use crate::request::{BatchRequest, Cursor, decode_flat};
use crate::schema::Coord;
use crate::slice::{Requests, Slice};
use crate::{Error, Result, TensorElement};

pub type Deferred<'a, T> = Box<dyn FnOnce() -> Result<T> + Send + 'a>;

pub enum Preferred<'a> {
    Ready(Option<RequestIterator>),
    Sources(Vec<Deferred<'a, Preferred<'a>>>),
}

pub enum Ordered<'a> {
    Ready(Requests<'static>),
    Sources(Vec<Deferred<'a, Ordered<'a>>>),
}

pub enum Selection<'a> {
    Ready(Requests<'a>),
    Source(Deferred<'a, Selection<'a>>),
}

/// Runtime descriptions and candidate-planning work are bounded independently of batch width.
pub(crate) const MAX_EXPRESSION_NODES: usize = 65_536;

pub fn node_count(children: impl IntoIterator<Item = usize>) -> Result<usize> {
    children.into_iter().try_fold(1usize, |total, count| {
        total
            .checked_add(count)
            .filter(|&n| n <= MAX_EXPRESSION_NODES)
            .ok_or_else(|| Error::Unsupported("expression description limit exceeded".into()))
    })
}

fn visit(count: &mut usize) -> Result<()> {
    *count = count
        .checked_add(1)
        .filter(|&n| n <= MAX_EXPRESSION_NODES)
        .ok_or_else(|| Error::Unsupported("expression traversal limit exceeded".into()))?;
    Ok(())
}

fn reserve<T>(items: &mut Vec<T>, additional: usize) -> Result<()> {
    items
        .try_reserve(additional)
        .map_err(|_| Error::Unsupported("expression traversal allocation failed".into()))
}

pub fn preferred<'a, E: Expression + ?Sized>(
    source: &'a E,
    shape: &'a [u64],
) -> Result<Option<RequestIterator>>
where
    E::DType: TensorElement,
{
    let mut pending: Vec<Deferred<'a, Preferred<'a>>> = Vec::new();
    reserve(&mut pending, 1)?;
    pending.push(Box::new(move || source.preferred_step(shape)));
    let mut visited = 0;

    while let Some(next) = pending.pop() {
        visit(&mut visited)?;
        match next()? {
            Preferred::Ready(Some(requests)) => return Ok(Some(requests)),
            Preferred::Ready(None) => {}
            Preferred::Sources(children) => {
                reserve(&mut pending, children.len())?;
                pending.extend(children.into_iter().rev());
            }
        }
    }

    Ok(None)
}

pub fn selection<E: Expression + ?Sized>(source: &E, slice: Slice) -> Result<Requests<'_>>
where
    E::DType: TensorElement,
{
    let mut step = source.selection_step(slice)?;
    let mut visited = 0;

    loop {
        visit(&mut visited)?;
        step = match step {
            Selection::Ready(requests) => return Ok(requests),
            Selection::Source(next) => next()?,
        };
    }
}

pub fn ordered<E: Expression + ?Sized>(source: &E, slice: Slice) -> Result<Requests<'static>>
where
    E::DType: TensorElement,
{
    let mut pending: Vec<Deferred<'_, Ordered<'_>>> = Vec::new();
    let mut streams = Vec::new();
    reserve(&mut pending, 1)?;
    pending.push(Box::new(move || source.ordered_step(slice)));
    let mut visited = 0;

    while let Some(next) = pending.pop() {
        visit(&mut visited)?;
        match next()? {
            Ordered::Ready(stream) => {
                reserve(&mut streams, 1)?;
                streams.push(stream);
            }
            Ordered::Sources(children) => {
                reserve(&mut pending, children.len())?;
                pending.extend(children.into_iter().rev());
            }
        }
    }

    if streams.len() == 1 {
        return Ok(streams.pop().expect("one candidate stream"));
    }

    Ok(merge(streams, source.shape().into()))
}

/// Adapt each compact request with reusable scratch; the union retains scalar heads.
/// Coordinates are allocated only for distinct outputs in the bounded result batch.
fn merge(streams: Vec<Requests<'static>>, shape: crate::Shape) -> Requests<'static> {
    let shape: Arc<[u64]> = Arc::from(shape.as_slice());
    let inputs: Vec<_> = streams
        .into_iter()
        .map(|mut stream| {
            let shape = Arc::clone(&shape);
            let mut coordinates: Option<Cursor<BatchRequest, Arc<[u64]>>> = None;
            let mut scratch = Coord::new();
            let mut budget = super::driver::WorkBudget::new();
            futures::stream::poll_fn(move |cx| {
                loop {
                    ready!(budget.poll(cx));
                    if coordinates
                        .as_mut()
                        .is_some_and(|cursor| cursor.next_into(&mut scratch))
                    {
                        let flat = scratch.iter().zip(shape.iter()).try_fold(
                            0u64,
                            |flat, (coordinate, dimension)| {
                                flat.checked_mul(*dimension)
                                    .and_then(|flat| flat.checked_add(*coordinate))
                                    .ok_or_else(|| {
                                        Error::InvalidCoord("candidate position overflow".into())
                                    })
                            },
                        );
                        return Poll::Ready(Some(flat));
                    }

                    match ready!(stream.poll_next_unpin(cx)) {
                        Some(Ok(request)) => match request.into_cursor(Arc::clone(&shape)) {
                            Ok(cursor) => coordinates = Some(cursor),
                            Err(error) => return Poll::Ready(Some(Err(error))),
                        },
                        Some(Err(error)) => return Poll::Ready(Some(Err(error))),
                        None => return Poll::Ready(None),
                    }
                }
            })
        })
        .collect();
    let union = collate::try_union(Collator::default(), inputs).boxed();
    futures::stream::try_unfold(
        (union, shape, Coord::new()),
        |(mut union, shape, mut scratch)| async move {
            let mut output = Vec::with_capacity(MAX_BATCH_ELEMENTS);

            while output.len() < MAX_BATCH_ELEMENTS {
                let Some(flat) = union.try_next().await? else {
                    break;
                };
                decode_flat(flat, &shape, &mut scratch)?;
                output.push(scratch.to_vec());
            }

            if output.is_empty() {
                Ok(None)
            } else {
                Ok(Some((
                    BatchRequest::explicit(output)?,
                    (union, shape, scratch),
                )))
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
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
}
