use safecast::AsType;

use crate::{BoxFuture, Result, Tensor};

pub(crate) fn lookup_block_id<'a, FE>(
    tensor: &'a Tensor<FE>,
    key: &'a [u64],
) -> BoxFuture<'a, Result<Option<u64>>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.lookup_block_id_impl(key).await })
}

pub(crate) fn upsert_block_id<'a, FE>(
    tensor: &'a Tensor<FE>,
    key: Vec<u64>,
    block_id: u64,
) -> BoxFuture<'a, Result<()>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.upsert_block_id_impl(key, block_id).await })
}
