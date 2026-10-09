//! Completion-order evaluation and numeric terminals, with ordered controls.

use fensor::{
    TensorMatMul, TensorRead, TensorReduce, TensorReduceAll, TensorTransform, TensorTrig,
    TensorUnary,
};
use ha_ndarray::axes;

use super::benchmark;
use super::benchmark_source::source;

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
    benchmark::copy(&name, view, 31, cache, cache < 1_000_000).await;
}

pub async fn run() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();

    for sparse in [false, true] {
        let kind = if sparse { "sparse" } else { "dense" };
        for cache in [32768, 1_000_000] {
            let (rows, cols) = if smoke { (2, 3) } else { (65, 129) };
            let (_tensor_root, tensor) =
                source(rows, cols, sparse, 128, if sparse { 7 } else { 1 }, cache).await;
            let geometric = tensor.view().transpose(None).unwrap();
            measure("geometric", kind, cache, &geometric).await;
            measure(
                "unary",
                kind,
                cache,
                &geometric.round().await.unwrap().sin().await.unwrap(),
            )
            .await;
            let (_tensor_root, tensor) = source(
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
            let (_left_root, left) =
                source(m, k, sparse, 128, if sparse { 7 } else { 1 }, cache).await;
            let (_right_root, right) =
                source(k, n, sparse, 31, if sparse { 7 } else { 1 }, cache).await;
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

// Isolate mapping construction from consumption; no filesystem setup is timed.
pub async fn owned() {
    use fensor::{AxisRange, Layout, TensorAbs, TensorExpression};

    use super::common;

    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    for sparse in [false, true] {
        for (name, rank, gather) in [
            ("ordinary", 2, false),
            ("high_rank", 12, false),
            ("gather", 2, true),
        ] {
            let mut shape = vec![1; rank];
            shape[rank - 2] = 16;
            shape[rank - 1] = 16;
            let (_root, tensor) = common::fixture::source(
                "owned_mapping",
                shape.clone().into(),
                if sparse {
                    Layout::Sparse { axis: None }
                } else {
                    Layout::Dense
                },
                64,
                1_000_000,
                std::iter::repeat_n(1f32, 256),
            )
            .await;
            let mut expression = TensorExpression::new(tensor).unwrap();
            if gather {
                expression = expression
                    .slice(
                        vec![
                            AxisRange::Of((0..16).rev().collect()),
                            AxisRange::In(0, 16, 1),
                        ]
                        .into(),
                    )
                    .unwrap();
            }
            for _ in 0..16 {
                expression = TensorExpression::new(expression.abs().await.unwrap()).unwrap();
            }
            let name = format!("owned_{name}_{}", if sparse { "sparse" } else { "dense" });
            let builds = if smoke { 2 } else { 50_000 };
            benchmark::measure(&name, "transform", "warm", async {
                for _ in 0..builds {
                    std::hint::black_box(
                        expression
                            .clone()
                            .flip(rank - 1)
                            .unwrap()
                            .transpose(None)
                            .unwrap()
                            .transpose(None)
                            .unwrap(),
                    );
                }
                builds
            })
            .await;
            for mode in ["row", "coordinate"] {
                benchmark::consume(&expression, mode).await;
                benchmark::measure(&name, mode, "warm", async {
                    let mut count = 0;
                    for _ in 0..if smoke { 2 } else { 128 } {
                        count += benchmark::consume(&expression, mode).await;
                    }
                    count
                })
                .await;
            }
        }
    }
}
