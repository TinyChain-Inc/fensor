use ha_ndarray::{Axes, Range, Shape};
use safecast::AsType;

use crate::{Result, Tensor};

pub(crate) fn reshape<FE>(tensor: Tensor<FE>, shape: Shape) -> Result<Tensor<FE>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    tensor.reshape_impl(shape)
}

pub(crate) fn slice<FE>(tensor: Tensor<FE>, range: Range) -> Result<Tensor<FE>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    tensor.slice_impl(range)
}

pub(crate) fn transpose<FE>(tensor: Tensor<FE>, permutation: Option<Axes>) -> Result<Tensor<FE>>
where
    FE: freqfs::FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    tensor.transpose_impl(permutation)
}
