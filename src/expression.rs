//! Private batch construction shared by typed elementwise expressions.
use futures::StreamExt;
use futures::stream::BoxStream;
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArray, NDArrayRead, NDArrayWhere};

use crate::traits::{BoxFuture, coordinate_batches};
use crate::{
    Error, Layout, Result, Tensor, TensorElement, TensorFileEntry, TensorGeometry, TensorRead,
    TensorView,
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
    fn build<'a>(&'a self, coords: &'a [Vec<u64>]) -> BoxFuture<'a, Result<Batch<Self::DType>>>;
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
    coords: &[Vec<u64>],
) -> Result<EvaluatedBatch<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    validate_bound("coordinate batch", coords.len())?;
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

type CoordinateBatch<T> = (Vec<Vec<u64>>, EvaluatedBatch<T>);

/// Evaluate bounded coordinate batches with at most `num_cpus::get().max(1)`
/// batches in flight. Expression temporaries scale with batch size, expression
/// size, and concurrency, not total tensor size. Collecting the returned stream
/// can still allocate whole-tensor output in the caller.
pub fn evaluated_batches<'a, E, I>(
    expression: &'a E,
    coords: I,
) -> BoxStream<'a, Result<CoordinateBatch<E::DType>>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
    I: Iterator<Item = Vec<u64>> + Send + 'a,
{
    futures::stream::iter(coordinate_batches(coords))
        .map(move |coords| async move {
            let values = evaluate_batch(expression, &coords).await?;

            Ok((coords, values))
        })
        .buffered(num_cpus::get().max(1))
        .boxed()
}

impl<FE, T> Expression for TensorView<'_, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn build<'a>(&'a self, coords: &'a [Vec<u64>]) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let mut values = Vec::with_capacity(coords.len());

            for coord in coords {
                values.push(self.read_value(coord).await?);
            }

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
    fn build<'a>(&'a self, coords: &'a [Vec<u64>]) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move { self.view().build(coords).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut batches = coordinate_batches(coords);
        assert_eq!(consumed.get(), 0);
        assert_eq!(batches.next().unwrap().len(), MAX_BATCH_ELEMENTS);
        assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
        drop(batches);
        assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
    }
}
