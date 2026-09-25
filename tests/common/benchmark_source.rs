//! Shared benchmark setup and consumption; callers own timing and observation.

use fensor::{Layout, Tensor, TensorSchema, TensorWrite};
use ha_ndarray::shape;
use number_general::DType;

use super::common::{self, FsEntry};

pub async fn source(
    m: usize,
    n: usize,
    sparse: bool,
    capacity: usize,
    density: usize,
    cache: usize,
) -> Tensor<FsEntry, f32> {
    let (root, dir) = common::new_dir("matrix_benchmark").await;
    drop(dir);
    let dir = freqfs::Cache::new(cache, None, 0, std::time::Duration::from_secs(1))
        .load(root)
        .unwrap();
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![m, n]).unwrap(),
        if sparse {
            Layout::Sparse { axis: None }
        } else {
            Layout::Dense
        },
        capacity,
    )
    .await
    .unwrap();

    for (i, c) in common::iter_coords(&[m, n]).enumerate() {
        if density != 0 && i % density == 0 {
            tensor.write_value(&c, (i % 5 + 1) as f32).await.unwrap();
        }
    }
    tensor.sync().await.unwrap();
    tensor
}
