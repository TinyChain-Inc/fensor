//! Shallow expression traversal. Deferred steps retain borrows, never nested consumers.

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};

use super::{Expression, MAX_BATCH_ELEMENTS, RequestIterator};
use crate::request::{BatchRequest, Cursor};
use crate::schema::Coord;
use crate::slice::{Requests, Slice};
use crate::{Error, Result, TensorElement};

pub type Deferred<'a, T> = Box<dyn FnOnce() -> Result<T> + Send + 'a>;

pub enum Preferred<'a> {
    Ready(Option<RequestIterator>),
    Sources(Vec<Deferred<'a, Preferred<'a>>>),
}

pub enum Support<'a> {
    Ready(Requests<'static>),
    Sources(Vec<(usize, Deferred<'a, Support<'a>>)>),
}

pub enum Selection<'a> {
    Ready(Requests<'a>),
    Source(Deferred<'a, Selection<'a>>),
}

/// Runtime descriptions and support-planning work are bounded independently of batch width.
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

pub fn support<E: Expression + ?Sized>(source: &E, slice: Slice) -> Result<Requests<'static>>
where
    E::DType: TensorElement,
{
    // Slots preserve stream polling order separately from eager provider-error order.
    enum Node {
        Leaf(usize),
        Union(Vec<usize>),
    }
    let mut pending: Vec<(usize, usize, Deferred<'_, Support<'_>>)> = Vec::new();
    let mut nodes = vec![Node::Union(vec![0])];
    let mut streams = Vec::new();
    reserve(&mut pending, 1)?;
    pending.push((0, 0, Box::new(move || source.support_step(slice))));
    let mut visited = 0;
    while let Some((parent, slot, next)) = pending.pop() {
        visit(&mut visited)?;
        let index = nodes.len();
        if let Node::Union(children) = &mut nodes[parent] {
            children[slot] = index;
        }
        reserve(&mut nodes, 1)?;
        match next()? {
            Support::Ready(stream) => {
                nodes.push(Node::Leaf(streams.len()));
                reserve(&mut streams, 1)?;
                streams.push(Some(stream));
            }
            Support::Sources(children) => {
                nodes.push(Node::Union(vec![0; children.len()]));
                reserve(&mut pending, children.len())?;
                pending.extend(
                    children
                        .into_iter()
                        .rev()
                        .map(|(slot, child)| (index, slot, child)),
                );
            }
        }
    }
    let mut ordered = Vec::new();
    reserve(&mut ordered, streams.len())?;
    let mut traversal = vec![0];
    while let Some(index) = traversal.pop() {
        match &nodes[index] {
            Node::Leaf(stream) => {
                ordered.push(streams[*stream].take().expect("unique support leaf"))
            }
            Node::Union(children) => {
                reserve(&mut traversal, children.len())?;
                traversal.extend(children.iter().rev().copied());
            }
        }
    }
    if ordered.len() == 1 {
        return Ok(ordered.pop().expect("one support stream"));
    }
    Ok(merge(ordered, source.shape().into()))
}

/// One compact request and reusable lookahead coordinate per leaf, with one output batch.
/// Flattening the union prevents stream polling and drop from following expression depth.
fn merge(streams: Vec<Requests<'static>>, shape: crate::Shape) -> Requests<'static> {
    struct Input {
        stream: Requests<'static>,
        coordinates: Option<Cursor<BatchRequest, Arc<[u64]>>>,
        next: Coord,
        ready: bool,
        done: bool,
    }
    let inputs: Vec<_> = streams
        .into_iter()
        .map(|stream| Input {
            stream,
            coordinates: None,
            next: Coord::new(),
            ready: false,
            done: false,
        })
        .collect();
    let shape: Arc<[u64]> = Arc::from(shape.as_slice());
    futures::stream::try_unfold(
        (
            inputs,
            shape,
            None::<Coord>,
            super::driver::WorkBudget::new(),
        ),
        |(mut inputs, shape, mut previous, mut budget)| async move {
            let mut output = Vec::with_capacity(MAX_BATCH_ELEMENTS);
            while output.len() < MAX_BATCH_ELEMENTS {
                let mut first: Option<usize> = None;
                for i in 0..inputs.len() {
                    futures::future::poll_fn(|cx| budget.poll(cx)).await;
                    let input = &mut inputs[i];
                    // Retire every matching head in one pass. Repeated operands do
                    // not require another full leaf scan for each duplicate.
                    while !input.done && (!input.ready || previous.as_ref() == Some(&input.next)) {
                        futures::future::poll_fn(|cx| budget.poll(cx)).await;
                        input.ready = input
                            .coordinates
                            .as_mut()
                            .is_some_and(|cursor| cursor.next_into(&mut input.next));
                        if input.ready {
                            continue;
                        }
                        if let Some(request) = input.stream.try_next().await? {
                            input.coordinates = Some(request.into_cursor(Arc::clone(&shape))?);
                        } else {
                            input.coordinates = None;
                            input.done = true;
                        }
                    }
                    if input.ready && first.is_none_or(|first| inputs[i].next < inputs[first].next)
                    {
                        first = Some(i);
                    }
                }
                let Some(first) = first else {
                    break;
                };
                let coord = &inputs[first].next;
                output.push(coord.to_vec());
                previous.get_or_insert_with(Coord::new).clone_from(coord);
            }
            if output.is_empty() {
                Ok(None)
            } else {
                Ok(Some((
                    BatchRequest::explicit(output)?,
                    (inputs, shape, previous, budget),
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
    use std::task::Poll;

    use super::*;
    use crate::read_metrics::{CURRENT, Metrics};

    #[tokio::test]
    async fn wide_sparse_support_retains_compact_requests_and_releases_sources() {
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
    async fn ready_sparse_support_can_be_cancelled_and_errors_release_sources() {
        for fail in [false, true] {
            let lease = Arc::new(());
            let weak = Arc::downgrade(&lease);
            let streams = (0..257)
                .map(|index| {
                    let lease = Arc::clone(&lease);
                    futures::stream::iter([if fail && index == 256 {
                        Err(Error::InvalidLayout("support source failure".into()))
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
                    matches!(stream.try_next().await, Err(Error::InvalidLayout(message)) if message == "support source failure")
                );
                assert!(weak.upgrade().is_none());
            } else {
                let mut next = Box::pin(stream.try_next());
                assert!(matches!(futures::poll!(next.as_mut()), Poll::Pending));
                assert!(weak.upgrade().is_some());
                drop(next);
                drop(stream);
                assert!(weak.upgrade().is_none());
            }
        }
    }
}
