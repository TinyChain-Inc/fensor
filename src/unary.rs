//! Typed, read-only unary views over geometric tensor views.

use std::iter;

use freqfs::DirLock;
use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{Array, ArrayAccess, Axes, Buffer, NDArrayRead, NDArrayUnary, Range, Shape};
use safecast::AsType;

use crate::Result;
use crate::schema::Layout;
use crate::stream::TensorViewEncoder;
use crate::tensor::{Tensor, TensorElement, TensorFileEntry};
use crate::traits::{
    BoxFuture, SparseElementStream, TensorGeometry, TensorRead, TensorTransform, TensorUnary,
    TensorViewSemantics, ValueBlockStream,
};
use crate::view::{self, TensorView};

mod sealed {
    pub trait Sealed {}
}

/// An operation which builds an ndarray expression without evaluating it.
///
/// This trait is sealed: only the supported unary operations and their typed
/// compositions can be used in a `UnaryView`.
pub trait UnaryOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>>;
}

/// Elementwise exponentiation.
#[derive(Clone, Copy, Debug)]
pub struct Exp;

/// Elementwise natural logarithm.
#[derive(Clone, Copy, Debug)]
pub struct Ln;

/// Elementwise rounding.
#[derive(Clone, Copy, Debug)]
pub struct Round;

/// Apply `Previous`, then `Next`, without evaluating an intermediate buffer.
#[derive(Clone, Copy, Debug)]
pub struct Then<Previous, Next> {
    previous: Previous,
    next: Next,
}

impl sealed::Sealed for Exp {}
impl sealed::Sealed for Ln {}
impl sealed::Sealed for Round {}
impl<P: sealed::Sealed, N: sealed::Sealed> sealed::Sealed for Then<P, N> {}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Exp {
    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.exp()?))
    }
}

impl<T: TensorElement + ha_ndarray::Float> UnaryOp<T> for Ln {
    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.ln()?))
    }
}

impl<T: TensorElement + ha_ndarray::Float + ha_ndarray::Real> UnaryOp<T> for Round {
    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.round()?))
    }
}

impl<T, P, N> UnaryOp<T> for Then<P, N>
where
    T: TensorElement,
    P: UnaryOp<T>,
    N: UnaryOp<T>,
{
    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        self.next.apply(self.previous.apply(array)?)
    }
}

/// A reusable unary expression with a geometric source and a typed operation.
///
/// Unary views implement read and composition traits, but not `TensorWrite`.
/// Unlike a geometric view, a computed result cannot be passed to a writer:
///
/// ```compile_fail,E0277
/// use fensor::{Tensor, TensorFileEntry, TensorUnary, TensorWrite};
/// fn requires_write<V: TensorWrite>(_: &V) {}
/// async fn computed_is_read_only<FE: TensorFileEntry<f32>>(tensor: &Tensor<FE, f32>) {
///     let computed = tensor.view().exp().await.unwrap();
///     requires_write(&computed);
/// }
/// ```
///
/// The geometric source remains writable without requiring `FE: Clone`:
///
/// ```
/// use fensor::{Tensor, TensorFileEntry, TensorUnary, TensorWrite};
/// fn requires_write<V: TensorWrite>(_: &V) {}
/// async fn geometric_is_writable<FE: TensorFileEntry<f32>>(tensor: &Tensor<FE, f32>) {
///     let source = tensor.view().clone();
///     requires_write(&source);
///     let _computed = source.exp().await.unwrap().round().await.unwrap();
/// }
/// ```
#[derive(Clone)]
pub struct UnaryView<Source, Op> {
    source: Source,
    op: Op,
}

impl<'t, FE, T> TensorUnary for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    type ExpOutput = UnaryView<Self, Exp>;
    type LnOutput = UnaryView<Self, Ln>;
    type RoundOutput = UnaryView<Self, Round>;

    fn exp(&self) -> BoxFuture<'_, Result<Self::ExpOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Exp,
            })
        })
    }

    fn ln(&self) -> BoxFuture<'_, Result<Self::LnOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Ln,
            })
        })
    }

    fn round(&self) -> BoxFuture<'_, Result<Self::RoundOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.clone(),
                op: Round,
            })
        })
    }
}

impl<S, O> TensorUnary for UnaryView<S, O>
where
    S: TensorRead + Clone,
    S::DType: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
    O: UnaryOp<S::DType>,
{
    type ExpOutput = UnaryView<S, Then<O, Exp>>;
    type LnOutput = UnaryView<S, Then<O, Ln>>;
    type RoundOutput = UnaryView<S, Then<O, Round>>;

    fn exp(&self) -> BoxFuture<'_, Result<Self::ExpOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.source.clone(),
                op: Then {
                    previous: self.op.clone(),
                    next: Exp,
                },
            })
        })
    }

    fn ln(&self) -> BoxFuture<'_, Result<Self::LnOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.source.clone(),
                op: Then {
                    previous: self.op.clone(),
                    next: Ln,
                },
            })
        })
    }

    fn round(&self) -> BoxFuture<'_, Result<Self::RoundOutput>> {
        Box::pin(async move {
            Ok(UnaryView {
                source: self.source.clone(),
                op: Then {
                    previous: self.op.clone(),
                    next: Round,
                },
            })
        })
    }
}

impl<S, O> TensorGeometry for UnaryView<S, O>
where
    S: TensorGeometry,
    O: Send + Sync,
{
    type DType = S::DType;

    fn dtype(&self) -> Self::DType {
        self.source.dtype()
    }
    fn layout(&self) -> Layout {
        self.source.layout()
    }
    fn shape(&self) -> &[usize] {
        self.source.shape()
    }
}

impl<S: TensorGeometry, O: Send + Sync> TensorViewSemantics for UnaryView<S, O> {
    fn is_base_tensor(&self) -> bool {
        false
    }
    fn supports_write_through(&self) -> bool {
        false
    }
}

impl<S, O> TensorRead for UnaryView<S, O>
where
    S: TensorRead,
    S::DType: TensorElement,
    O: UnaryOp<S::DType>,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let value = self.source.read_value(coord).await?;
            Ok(evaluate(vec![value], self.layout(), &self.op)?[0])
        })
    }

    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        // The geometric source owns the single bounded, ordered read pipeline.
        // Evaluate the entire typed expression once for each consumed batch.
        Ok(self
            .source
            .read_blocks()?
            .and_then(move |values| async move { evaluate(values, self.layout(), &self.op) })
            .boxed())
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let source = self
                .source
                .read_sparse_elements_in_order(range, requested_order)
                .await?;
            Ok(source
                .try_chunks(crate::schema::MAX_BLOCK_CAPACITY)
                .map_err(|error| error.1)
                .and_then(move |elements| async move {
                    let (coords, values): (Vec<_>, Vec<_>) = elements.into_iter().unzip();
                    let values = evaluate(values, self.layout(), &self.op)?;
                    Ok(futures::stream::iter(
                        coords
                            .into_iter()
                            .zip(values)
                            .filter(|(_, value)| *value != Self::DType::default())
                            .map(Ok),
                    ))
                })
                .try_flatten()
                .boxed())
        })
    }
}

impl<S: TensorTransform, O: Send + Sync> TensorTransform for UnaryView<S, O> {
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            source: self.source.reshape(shape)?,
            op: self.op,
        })
    }
    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            source: self.source.broadcast(shape)?,
            op: self.op,
        })
    }
    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            source: self.source.flip(axis)?,
            op: self.op,
        })
    }
    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            source: self.source.slice(range)?,
            op: self.op,
        })
    }
    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            source: self.source.squeeze(axes)?,
            op: self.op,
        })
    }
    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            source: self.source.transpose(permutation)?,
            op: self.op,
        })
    }
    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            source: self.source.unsqueeze(axes)?,
            op: self.op,
        })
    }
}

impl<'t, FE, T, O> UnaryView<TensorView<'t, FE, T>, O>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
    O: UnaryOp<T>,
{
    pub fn view_encoder(&self) -> TensorViewEncoder<'_, Self> {
        TensorViewEncoder::new(self, self.source.tensor().block_shape())
    }

    /// Consume the expression into independent filesystem storage.
    pub async fn materialize(&self, dir: DirLock<FE>, max_capacity: usize) -> Result<Tensor<FE, T>>
    where
        FE: AsType<String> + From<String>,
    {
        view::materialize(self, dir, max_capacity).await
    }
}

// Determine sparse support before constructing any operations. Intermediate
// zeros stay in the expression until its final buffer is evaluated.
fn evaluate<T: TensorElement, O: UnaryOp<T>>(
    mut values: Vec<T>,
    layout: Layout,
    op: &O,
) -> Result<Vec<T>> {
    let sparse = matches!(layout, Layout::Sparse { .. });
    let positions: Vec<usize> = if sparse {
        values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (*v != T::default()).then_some(i))
            .collect()
    } else {
        Vec::new()
    };
    let input = if sparse {
        positions.iter().map(|&i| values[i]).collect::<Vec<_>>()
    } else {
        std::mem::take(&mut values)
    };
    if input.is_empty() {
        return Ok(values);
    }
    let shape = iter::once(input.len()).collect();
    let array = ArrayAccess::from(Array::new(Buffer::from(input), shape)?);
    let output = op.apply(array)?.buffer()?.to_slice()?.into_vec();
    if sparse {
        for (i, value) in positions.into_iter().zip(output) {
            values[i] = value;
        }
        Ok(values)
    } else {
        Ok(output)
    }
}
