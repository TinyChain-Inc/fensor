//! Completion-order evaluation and numeric terminals, with ordered controls.

use std::time::{Duration, Instant};

use fensor::{
    Tensor, TensorMatMul, TensorRead, TensorReduce, TensorReduceAll, TensorTransform, TensorUnary,
};
use freqfs::Cache;
use ha_ndarray::axes;

#[path = "common/benchmark.rs"]
mod benchmark;

#[path = "common/benchmark_source.rs"]
mod benchmark_source;

mod common;
use benchmark_source::source;
use common::{FsEntry, counters};

async fn measure<V: TensorRead<DType = f32> + TensorReduceAll>(
    name: &str,
    kind: &str,
    cache: usize,
    view: &V,
) {
    for mode in ["row", "coordinate", "sum", "product", "min", "max"] {
        for temperature in ["first", "warm"] {
            counters::reset_traffic();
            let start = Instant::now();
            let count = match mode {
                "sum" => {
                    std::hint::black_box(view.sum_all().await.unwrap());
                    1
                }
                "product" => {
                    std::hint::black_box(view.product_all().await.unwrap());
                    1
                }
                "min" => {
                    std::hint::black_box(view.min_all().await.unwrap());
                    1
                }
                "max" => {
                    std::hint::black_box(view.max_all().await.unwrap());
                    1
                }
                _ => benchmark::consume(view, mode).await,
            };
            let elapsed = start.elapsed().as_nanos();
            let traffic = counters::snapshot_traffic();
            println!(
                "PIPELINE,{name},{kind},{cache},0,{mode}-{temperature},{elapsed},{count},{},{}",
                traffic.loads, traffic.saves
            );
        }
    }

    let (root, dir) = common::new_dir("completion_copy").await;
    drop(dir);
    let dir = Cache::<FsEntry>::new(cache, None, 0, Duration::from_secs(1))
        .load(root.clone())
        .unwrap();
    counters::reset_traffic();
    let start = Instant::now();
    let output = Tensor::copy_from(dir, view, 31).await.unwrap();
    output.sync().await.unwrap();
    let elapsed = start.elapsed().as_nanos();
    let traffic = counters::snapshot_traffic();
    let count = view.shape().iter().product::<usize>();
    println!(
        "PIPELINE,{name},{kind},{cache},31,copy,{elapsed},{count},{},{}",
        traffic.loads, traffic.saves
    );
    drop(output);
    common::cleanup(&root).await;
}

#[tokio::test]
#[ignore = "paired release measurements, not a timing gate"]
async fn completion_benchmark() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();

    for sparse in [false, true] {
        let kind = if sparse { "sparse" } else { "dense" };
        for cache in [32768, 1_000_000] {
            let (rows, cols) = if smoke { (2, 3) } else { (65, 129) };
            let tensor = source(rows, cols, sparse, 128, if sparse { 7 } else { 1 }, cache).await;
            let geometric = tensor.view().transpose(None).unwrap();
            measure("geometric", kind, cache, &geometric).await;
            measure(
                "unary",
                kind,
                cache,
                &geometric.ln().await.unwrap().round().await.unwrap(),
            )
            .await;
            let tensor = source(
                if smoke { 3 } else { 4097 },
                2,
                sparse,
                128,
                if sparse { 7 } else { 1 },
                cache,
            )
            .await;
            measure(
                "reduction",
                kind,
                cache,
                &tensor.view().sum(axes![1], false).await.unwrap(),
            )
            .await;
            let (m, k, n) = if smoke { (2, 3, 2) } else { (65, 17, 65) };
            let left = source(m, k, sparse, 128, if sparse { 7 } else { 1 }, cache).await;
            let right = source(k, n, sparse, 31, if sparse { 7 } else { 1 }, cache).await;
            measure(
                "matrix",
                kind,
                cache,
                &left.view().matmul(&right.view()).await.unwrap(),
            )
            .await;
        }
    }
}
