//! Paired stream and copy measurements; setup remains outside timed phases.

use std::time::{Duration, Instant};

use fensor::{Layout, Tensor, TensorMatMul, TensorSchema, TensorWrite};
use freqfs::Cache;
use ha_ndarray::shape;
use number_general::DType;

#[path = "common/benchmark.rs"]
mod benchmark;

mod common;
use common::{FsEntry, counters};

async fn source(
    rows: usize,
    cols: usize,
    layout: Layout,
    cache: usize,
) -> (std::path::PathBuf, Tensor<FsEntry, f32>) {
    let (root, dir) = common::new_dir("concurrency_source").await;
    drop(dir);
    let dir = Cache::new(cache, None, 0, Duration::from_secs(1))
        .load(root.clone())
        .unwrap();
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![rows, cols]).unwrap(),
        layout,
        128,
    )
    .await
    .unwrap();
    for i in 0..rows {
        for j in 0..cols {
            if layout == Layout::Dense || (i + j) % 7 == 0 {
                tensor
                    .write_value(&[i as u64, j as u64], ((i + j) % 5 + 1) as f32)
                    .await
                    .unwrap();
            }
        }
    }
    tensor.sync().await.unwrap();
    (root, tensor)
}

#[tokio::test]
#[ignore = "paired release measurements, not a timing gate"]
async fn concurrency_benchmark() {
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
    // Reproduce the case flagged by the broad comparison without changing its
    // values, source setup, stream warmup, or measured copy/sync boundary.
    let focus = std::env::var_os("FENSOR_BENCH_FOCUS_WIDE").is_some();
    for (name, rows, inner, cols) in cases {
        if focus && name != "wide" {
            continue;
        }
        for (kind, layout) in [
            ("dense", Layout::Dense),
            ("sparse", Layout::Sparse { axis: None }),
        ] {
            if focus && kind != "dense" {
                continue;
            }
            for cache in [32768, 1_000_000] {
                if focus && cache != 1_000_000 {
                    continue;
                }
                let (a_root, a) = source(rows, inner, layout, cache).await;
                let (b_root, b) = source(inner, cols, layout, cache).await;
                let view = a.view().matmul(&b.view()).await.unwrap();
                for phase in ["first", "warm"] {
                    counters::reset_traffic();
                    let start = Instant::now();
                    let count = benchmark::consume(&view, "coordinate").await;
                    let elapsed = start.elapsed();
                    let traffic = counters::snapshot_traffic();
                    println!(
                        "PIPELINE,{name},{kind},{cache},0,{phase},{},{count},{},{}",
                        elapsed.as_nanos(),
                        traffic.loads,
                        traffic.saves
                    );
                }
                for capacity in [1, 7, 31, 128, 4096] {
                    if focus && capacity != 31 {
                        continue;
                    }
                    let (root, dir) = common::new_dir("concurrency_copy").await;
                    drop(dir);
                    let dir = Cache::<FsEntry>::new(cache, None, 0, Duration::from_secs(1))
                        .load(root.clone())
                        .unwrap();
                    counters::reset_traffic();
                    let start = Instant::now();
                    let copy = Tensor::copy_from(dir, &view, capacity).await.unwrap();
                    copy.sync().await.unwrap();
                    let elapsed = start.elapsed();
                    let traffic = counters::snapshot_traffic();
                    println!(
                        "PIPELINE,{name},{kind},{cache},{capacity},copy,{},{},{},{}",
                        elapsed.as_nanos(),
                        rows * cols,
                        traffic.loads,
                        traffic.saves
                    );
                    drop(copy);
                    common::cleanup(&root).await;
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
