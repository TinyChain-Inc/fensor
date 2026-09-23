//! Private batch construction shared by typed elementwise expressions.
use futures::StreamExt;
use futures::stream::BoxStream;
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArrayRead, NDArrayWhere};

use crate::traits::{BoxFuture, coordinate_batches};
use crate::{
    Layout, Result, Tensor, TensorElement, TensorFileEntry, TensorGeometry, TensorRead, TensorView,
};

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

pub fn array<T: TensorElement>(values: Vec<T>) -> Result<ArrayAccess<'static, T>> {
    let shape = std::iter::once(values.len()).collect();

    Ok(ArrayAccess::from(Array::new(Buffer::from(values), shape)?))
}

impl<T: TensorElement> Batch<T> {
    // Selection is part of the lazy expression, not a materialized intermediate.
    pub fn masked(self) -> Result<Self> {
        let values = match &self.support {
            Some(support) if support.contains(&0) => ArrayAccess::from(
                array(support.clone())?.cond(self.array, array(vec![T::ZERO; support.len()])?)?,
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
pub fn union_support(left: Option<Vec<u8>>, right: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (left, right) {
        (Some(left), Some(right)) => {
            Some(left.into_iter().zip(right).map(|(l, r)| l | r).collect())
        }
        _ => None,
    }
}

pub async fn evaluate<E>(expression: &E, coords: &[Vec<u64>]) -> Result<EvaluatedBatch<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    let batch = expression.build(coords).await?;

    Ok(EvaluatedBatch {
        values: batch.array.buffer()?.to_slice()?.into_vec(),
        support: batch.support,
    })
}

pub struct EvaluatedBatch<T> {
    pub values: Vec<T>,
    pub support: Option<Vec<u8>>,
}

impl<T> EvaluatedBatch<T> {
    pub fn populated(self) -> Vec<T> {
        match self.support {
            Some(support) => self
                .values
                .into_iter()
                .zip(support)
                .filter_map(|(v, s)| (s != 0).then_some(v))
                .collect(),
            None => self.values,
        }
    }
}

type CoordinateBatch<T> = (Vec<Vec<u64>>, EvaluatedBatch<T>);

/// Evaluate bounded coordinate batches with at most `num_cpus::get().max(1)`
/// batches in flight. Expression temporaries scale with batch size, expression
/// size, and concurrency, not total tensor size. Collecting the returned stream
/// can still allocate whole-tensor output in the caller.
pub fn batches<'a, E, I>(
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
            let values = evaluate(expression, &coords).await?;

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
                array: array(values)?,
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
