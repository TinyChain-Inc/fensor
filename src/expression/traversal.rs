//! Shallow expression traversal. Steps borrow operands, never nested consumers.

use std::sync::Arc;
use std::task::{Poll, ready};

use collate::Collator;
use futures::{StreamExt, TryStreamExt};

use super::{Expression, MAX_BATCH_ELEMENTS, RequestIterator};
use crate::request::{BatchRequest, Cursor, decode_flat};
use crate::schema::Coord;
use crate::slice::{Requests, Slice};
use crate::{Error, Result, TensorElement};

/// Address discovery is independent of the numerical dtype. Borrowed children
/// let planning remain iterative before the evaluation trampoline starts.
pub trait Plan: Send + Sync {
    /// Describe one preferred-request step for the consumer's logical shape.
    /// None leaves request selection to another source or the consumer.
    fn preferred_step<'a>(&'a self, _shape: &'a [u64]) -> Result<Preferred<'a>> {
        Ok(Preferred::Ready(None))
    }

    /// Describe ordered occupied-request candidates, independent of numerical values.
    fn ordered_step(&self, slice: crate::slice::Slice) -> Result<Ordered<'_>> {
        Ok(Ordered::Ready(slice.stream()))
    }

    /// Describe one selection step; native readers may retain storage order.
    fn selection_step(&self, slice: crate::slice::Slice) -> Result<Selection<'_>> {
        Ok(Selection::Ready(slice.stream()))
    }
}

pub enum Step<'a, I, T> {
    Ready(T),
    Sources(Vec<(&'a dyn Plan, I)>),
}

pub type Preferred<'a> = Step<'a, &'a [u64], Option<RequestIterator>>;
pub type Ordered<'a> = Step<'a, Slice, Requests<'static>>;

pub enum Selection<'a> {
    Ready(Requests<'a>),
    Source(&'a dyn Plan, Slice),
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

/// Stop at the first leaf accepted by the consumer, or visit every leaf.
fn walk<'a, I, T, R>(
    mut step: Step<'a, I, T>,
    mut advance: impl FnMut(&'a dyn Plan, I) -> Result<Step<'a, I, T>>,
    mut consume: impl FnMut(T) -> Result<Option<R>>,
) -> Result<Option<R>> {
    let mut pending = Vec::new();
    let mut visited = 0;
    visit(&mut visited)?;

    loop {
        match step {
            Step::Ready(value) => {
                if let Some(result) = consume(value)? {
                    return Ok(Some(result));
                }
            }
            Step::Sources(children) => {
                reserve(&mut pending, children.len())?;
                pending.extend(children.into_iter().rev());
            }
        }

        let Some((source, input)) = pending.pop() else {
            return Ok(None);
        };
        visit(&mut visited)?;
        step = advance(source, input)?;
    }
}

pub fn preferred<'a, E: Expression + ?Sized>(
    source: &'a E,
    shape: &'a [u64],
) -> Result<Option<RequestIterator>>
where
    E::DType: TensorElement,
{
    walk(
        source.preferred_step(shape)?,
        |source, shape| source.preferred_step(shape),
        Ok,
    )
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
            Selection::Source(source, slice) => source.selection_step(slice)?,
        };
    }
}

pub fn ordered<E: Expression + ?Sized>(source: &E, slice: Slice) -> Result<Requests<'static>>
where
    E::DType: TensorElement,
{
    let mut streams = Vec::new();
    walk(
        source.ordered_step(slice)?,
        |source, slice| source.ordered_step(slice),
        |stream| {
            reserve(&mut streams, 1)?;
            streams.push(stream);
            Ok(None::<()>)
        },
    )?;

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
            futures::stream::poll_fn(move |cx| {
                loop {
                    let progress = ready!(tokio::task::coop::poll_proceed(cx));
                    if coordinates
                        .as_mut()
                        .is_some_and(|cursor| cursor.next_into(&mut scratch))
                    {
                        progress.made_progress();
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

                    let next = ready!(stream.poll_next_unpin(cx));
                    progress.made_progress();
                    match next {
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
#[path = "../../tests/unit/expression/traversal/tests.rs"]
mod tests;
