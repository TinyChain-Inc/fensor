//! Focused sparse allocation and final-value reclamation measurements.
use fensor::{Layout, Tensor, TensorRead, TensorSchema, TensorWrite};
use ha_ndarray::shape;
use number_general::DType;

use super::{benchmark, common};

pub async fn run() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    let capacities: &[usize] = if smoke { &[7] } else { &[1, 7, 31, 128, 4096] };
    for &capacity in capacities {
        for axis in [0, 1] {
            for cache in [16_384, 1_000_000] {
                let name = format!("sparse_axis{axis}_cap{capacity}_cache{cache}");
                let root = common::Directory::new(&name).await;
                let dir = freqfs::Cache::<common::FsEntry>::new(
                    cache,
                    None,
                    0,
                    std::time::Duration::from_secs(1),
                )
                .load(root.to_path_buf())
                .unwrap();
                let tensor = Tensor::<common::FsEntry, f32>::create(
                    dir,
                    TensorSchema::new(f32::dtype(), shape![8, 129]).unwrap(),
                    Layout::Sparse { axis: Some(axis) },
                    capacity,
                )
                .await
                .unwrap();
                // One populated value per sparse-axis coordinate: clearing it is
                // a final-value deletion on both baseline and revised layouts.
                for i in 0..8 {
                    tensor.write_value(&[i, i * 16], 1.).await.unwrap();
                }
                tensor.sync().await.unwrap();
                benchmark::storage(&name, "populated", &root).await;
                benchmark::streams(&name, &tensor, &["row", "coordinate"]).await;
                benchmark::copy(&name, &tensor.view(), capacity, cache, false).await;
                benchmark::measure(&name, "clear", "populated", async {
                    for i in 0..8 {
                        tensor.write_value(&[i, i * 16], 0.).await.unwrap();
                    }
                    tensor.sync().await.unwrap();
                    8
                })
                .await;
                for i in 0..8 {
                    assert_eq!(tensor.read_value(&[i, i * 16]).await.unwrap(), 0.);
                }
                benchmark::storage(&name, "cleared", &root).await;
                drop(tensor);
                common::cleanup(&root).await;
            }
        }
    }
}
