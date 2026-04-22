use safecast::AsType;

use crate::{BoxFuture, Result, Tensor};

pub(crate) fn read_value<'a, FE>(
    tensor: &'a Tensor<FE>,
    coord: &'a [u64],
) -> BoxFuture<'a, Result<f32>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.read_value_impl(coord).await })
}

pub(crate) fn write_value<'a, FE>(
    tensor: &'a Tensor<FE>,
    coord: &'a [u64],
    value: f32,
) -> BoxFuture<'a, Result<()>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    Box::pin(async move { tensor.write_value_impl(coord, value).await })
}
