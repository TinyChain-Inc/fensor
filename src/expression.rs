//! Private batch construction shared by typed elementwise expressions.
use futures::StreamExt;
use futures::stream::BoxStream;
use ha_ndarray::{Array, ArrayAccess, Buffer, NDArrayRead, NDArrayWhere};

use crate::traits::{BoxFuture, coordinate_batches};
use crate::{
    Layout, Result, TensorElement, TensorFileEntry, TensorGeometry, TensorRead, TensorView,
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
    // None means full support. This is independent of intermediate numerical zeros.
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
pub fn union_support(left: Option<Vec<u8>>, right: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (left, right) {
        (Some(left), Some(right)) => {
            Some(left.into_iter().zip(right).map(|(l, r)| l | r).collect())
        }
        _ => None,
    }
}

pub async fn evaluate<E>(expression: &E, coords: &[Vec<u64>]) -> Result<Vec<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    let batch = expression.build(coords).await?;

    Ok(batch.array.buffer()?.to_slice()?.into_vec())
}

type EvaluatedBatch<T> = (Vec<Vec<u64>>, Vec<T>);

pub fn batches<'a, E, I>(
    expression: &'a E,
    coords: I,
) -> BoxStream<'a, Result<EvaluatedBatch<E::DType>>>
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
