//! Paired stream and copy measurements; setup remains outside timed phases.

use super::benchmark;
use super::common::{self, FsEntry};
use fensor::{Layout, Tensor, TensorMatMul};
use ha_ndarray::shape;

async fn source(
    rows: usize,
    cols: usize,
    layout: Layout,
    cache: usize,
) -> (std::path::PathBuf, Tensor<FsEntry, f32>) {
    let (root, tensor) = common::fixture::source(
        "pipeline_source",
        shape![rows as u64, cols as u64],
        layout,
        128,
        cache,
        (0..rows).flat_map(|i| {
            (0..cols).map(move |j| {
                if layout == Layout::Dense || (i + j) % 7 == 0 {
                    ((i + j) % 5 + 1) as f32
                } else {
                    0.
                }
            })
        }),
    )
    .await;
    tensor.sync().await.unwrap();
    (root, tensor)
}

pub async fn run() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    let cases = if smoke {
        vec![("smoke", 2, 3, 2)]
    } else {
        vec![
            ("square", 65, 17, 65),
            ("wide", 2, 17, 2051),
            ("narrow", 4097, 3, 1),
        ]
    };
    for (name, rows, inner, cols) in cases {
        for (kind, layout) in [
            ("dense", Layout::Dense),
            ("sparse", Layout::Sparse { axis: None }),
        ] {
            for cache in [32768, 1_000_000] {
                let (a_root, a) = source(rows, inner, layout, cache).await;
                let (b_root, b) = source(inner, cols, layout, cache).await;
                let view = a.view().matmul(&b.view()).await.unwrap();
                let name = format!("pipeline_{name}_{kind}_cache{cache}");
                benchmark::streams(&name, &view, &["coordinate"]).await;
                for capacity in [1, 7, 31, 128, 4096] {
                    benchmark::copy(&name, &view, capacity, cache).await;
                }
                drop(view);
                drop(a);
                drop(b);
                common::cleanup(&a_root).await;
                common::cleanup(&b_root).await;
            }
        }
    }
}
