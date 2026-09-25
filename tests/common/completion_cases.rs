//! Completion-order evaluation and numeric terminals, with ordered controls.

use super::benchmark;
use super::benchmark_source::source;
use fensor::{
    TensorMatMul, TensorRead, TensorReduce, TensorReduceAll, TensorTransform, TensorUnary,
};
use ha_ndarray::axes;

async fn measure<V: TensorRead<DType = f32> + TensorReduceAll>(
    name: &str,
    kind: &str,
    cache: usize,
    view: &V,
) {
    let name = format!("completion_{name}_{kind}_cache{cache}");
    benchmark::streams(
        &name,
        view,
        &["row", "coordinate", "sum", "product", "min", "max"],
    )
    .await;
    benchmark::copy(&name, view, 31, cache).await;
}

pub async fn run() {
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
