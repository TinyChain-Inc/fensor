//! Copy-specific transposed fixtures, including multi-batch overlap.

use super::{benchmark, common};
use fensor::{Layout, TensorTransform};
use ha_ndarray::shape;

pub async fn run() {
    for (rows, cols, caches) in [(9, 17, [8192, 1_000_000]), (65, 65, [32768, 1_000_000])] {
        let fixture = format!("{rows}x{cols}");
        let (rows, cols) = if std::env::var_os("FENSOR_BENCH_SMOKE").is_some() {
            (2, 3)
        } else {
            (rows, cols)
        };
        for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
            let (root, tensor) = common::fixture::source(
                "copy_profile",
                shape![rows, cols],
                layout,
                31,
                1_000_000,
                (0..rows).flat_map(|i| {
                    (0..cols).map(move |j| {
                        if layout == Layout::Dense || j % 3 == 0 {
                            (i + j + 1) as f32
                        } else {
                            0.
                        }
                    })
                }),
            )
            .await;
            tensor.sync().await.unwrap();
            let view = tensor.view().transpose(None).unwrap();
            let name = format!("copy_{fixture}_{layout:?}").replace(',', "-");
            for cache in caches {
                for capacity in [1, 7, 31, 128, 4096] {
                    benchmark::copy(&name, &view, capacity, cache).await;
                }
            }
            drop(view);
            drop(tensor);
            common::cleanup(&root).await;
        }
    }
}
