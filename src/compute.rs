//! Shared block-streaming compute core for tensor math operations.
//!
//! Fensor stays a storage/indexing substrate: this module only marshals
//! block payloads (`Vec<T>`) into `ha_ndarray` arrays and back so the actual
//! per-block math delegates to `ha-ndarray`, never to a hand-rolled fusion
//! engine here.

use futures::StreamExt;
use ha_ndarray::{Array, Axes, Buffer, NDArrayRead, NDArrayUnary, Shape};
use safecast::AsType;

use crate::validate;
use crate::{
    Error, Layout, Tensor, TensorBlockStore, TensorElement, TensorFileEntry, TensorGeometry,
    TensorRead, TensorShape, TensorWrite,
};

/// The unary math operations `TensorUnary` supports.
pub(crate) enum UnaryOp {
    Exp,
    Ln,
    Round,
}

impl UnaryOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Round => "round",
        }
    }

    /// Zero-preserving ops (`f(0) == 0`) are safe on Sparse tensors, since
    /// unpopulated (implicitly-zero) elements don't need to be visited.
    pub(crate) fn is_zero_preserving(&self) -> bool {
        matches!(self, Self::Round)
    }
}

/// Apply `op` to every element of `values`, delegating the actual math to
/// `ha_ndarray`. `shape.product()` must equal `values.len()`.
pub(crate) fn apply_unary<T>(values: Vec<T>, shape: Shape, op: &UnaryOp) -> crate::Result<Vec<T>>
where
    T: ha_ndarray::Float + ha_ndarray::Real,
{
    let buffer: Buffer<T> = values.into();
    let array = Array::new(buffer, shape)?;
    let result = match op {
        UnaryOp::Exp => array.exp()?,
        UnaryOp::Ln => array.ln()?,
        UnaryOp::Round => array.round()?,
    };
    Ok(result.buffer()?.to_slice()?.into_vec())
}

/// Stream `op` over a Dense tensor's block grid, one block at a time, into a
/// freshly created sibling output tensor. Never buffers the whole tensor in
/// memory: each input block is read, transformed, and written before the
/// next block is read.
async fn stream_unary_dense<FE, T>(
    input: &Tensor<FE, T>,
    op: UnaryOp,
) -> crate::Result<Tensor<FE, T>>
where
    FE: TensorFileEntry<T> + AsType<String> + From<String>,
    T: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    let shape: TensorShape = input.shape().to_vec().into();
    let max_capacity = input.block_len();
    let output = input
        .create_sibling(shape, input.layout(), max_capacity)
        .await?;
    let block_shape: Shape = input.block_shape().to_vec().into();

    for block_id in 0..input.num_blocks() {
        let block = input
            .read_block(block_id)
            .await?
            .ok_or_else(|| Error::InvalidLayout("dense block missing".to_string()))?;
        let result = apply_unary(block, block_shape.clone(), &op)?;
        output.write_block(block_id, result).await?;
    }

    Ok(output)
}

/// Stream `op` over a Sparse tensor's populated elements only. Sparse block
/// ids are randomly assigned per populated row and don't correspond between
/// input and output tensors, so this reuses the identity-order element
/// stream + `write_value` rather than the dense block pipeline.
async fn apply_unary_sparse<FE, T>(
    input: &Tensor<FE, T>,
    op: UnaryOp,
) -> crate::Result<Tensor<FE, T>>
where
    FE: TensorFileEntry<T> + AsType<String> + From<String>,
    T: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    if !op.is_zero_preserving() {
        return Err(Error::Unsupported(format!(
            "{} is not supported for Sparse tensors: the implicit zero elements would be \
             corrupted, since {}(0) != 0",
            op.name(),
            op.name()
        )));
    }

    let shape: TensorShape = input.shape().to_vec().into();
    let output = input
        .create_sibling(shape, input.layout(), input.block_len())
        .await?;

    let full_range = validate::full_range(input.shape());
    let order: Axes = (0..input.ndim()).collect();
    let mut elements = input
        .read_sparse_elements_in_order(full_range, order)
        .await?;

    let one: Shape = std::iter::once(1usize).collect();
    while let Some(item) = elements.next().await {
        let (coord, value) = item?;
        let result = apply_unary(vec![value], one.clone(), &op)?;
        output.write_value(&coord, result[0]).await?;
    }

    Ok(output)
}

/// Dispatch `op` on `input` according to its layout: Dense streams full
/// blocks; Sparse streams populated elements only (and rejects zero-breaking
/// ops with a structured error).
pub(crate) async fn run_unary<FE, T>(
    input: &Tensor<FE, T>,
    op: UnaryOp,
) -> crate::Result<Tensor<FE, T>>
where
    FE: TensorFileEntry<T> + AsType<String> + From<String>,
    T: TensorElement + ha_ndarray::Float + ha_ndarray::Real,
{
    match input.layout() {
        Layout::Dense => stream_unary_dense(input, op).await,
        Layout::Sparse { .. } => apply_unary_sparse(input, op).await,
    }
}

#[cfg(test)]
mod tests {
    use ha_ndarray::shape;

    #[test]
    fn apply_unary_round_known_values() {
        // Round-half-away-from-zero on values whose rounding direction is
        // unambiguous either way.
        let values = vec![1.4f32, 1.6, -1.5];
        let result = super::apply_unary(values, shape![3], &super::UnaryOp::Round)
            .expect("apply_unary should succeed");
        assert_eq!(result, vec![1.0, 2.0, -2.0]);
    }
}
