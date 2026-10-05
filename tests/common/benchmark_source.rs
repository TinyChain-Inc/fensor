//! Shared benchmark setup and consumption; callers own timing and observation.

use fensor::Tensor;
use ha_ndarray::shape;

use super::common::{self, FsEntry};

pub async fn source(
    m: usize,
    n: usize,
    sparse: bool,
    capacity: usize,
    density: usize,
    cache: usize,
) -> (common::Directory, Tensor<FsEntry, f32>) {
    let (root, tensor) = common::fixture::source(
        "benchmark_source",
        shape![m as u64, n as u64],
        common::fixture::layout(sparse),
        capacity,
        1_000_000,
        (0..m * n).map(|i| {
            if density != 0 && i % density == 0 {
                (i % 5 + 1) as f32
            } else {
                0.
            }
        }),
    )
    .await;
    reopen(root, tensor, cache).await
}

pub async fn reopen(
    root: common::Directory,
    tensor: Tensor<FsEntry, f32>,
    cache: usize,
) -> (common::Directory, Tensor<FsEntry, f32>) {
    tensor.sync().await.unwrap();
    drop(tensor);
    // The measured read budget need not hold construction's pinned pages.
    let dir = freqfs::Cache::new(cache, None, 0, std::time::Duration::from_secs(1))
        .load(root.to_path_buf())
        .unwrap();
    let tensor = Tensor::load(dir).await.unwrap();
    (root, tensor)
}
