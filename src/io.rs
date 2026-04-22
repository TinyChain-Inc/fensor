use safecast::AsType;

use crate::{BoxFuture, Result, Tensor};

pub(crate) fn read_block<'a, FE>(
    tensor: &'a Tensor<FE>,
    block_id: u64,
) -> BoxFuture<'a, Result<Option<Vec<f32>>>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.read_block_impl(block_id).await })
}

pub(crate) fn write_block<'a, FE>(
    tensor: &'a Tensor<FE>,
    block_id: u64,
    block: Vec<f32>,
) -> BoxFuture<'a, Result<()>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.write_block_impl(block_id, block).await })
}
