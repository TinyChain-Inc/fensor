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
) -> Tensor<FsEntry, f32> {
    let (_, tensor) = common::fixture::source(
        "benchmark_source",
        shape![m as u64, n as u64],
        common::fixture::layout(sparse),
        capacity,
        cache,
        (0..m * n).map(|i| {
            if density != 0 && i % density == 0 {
                (i % 5 + 1) as f32
            } else {
                0.
            }
        }),
    )
    .await;
    tensor.sync().await.unwrap();
    tensor
}
