//! Identical cases for production timing and test-only structural profiling.

use fensor::{
    AxisRange, TensorRead, TensorReduce, TensorReduceAll, TensorSchema, TensorTransform,
    TensorUnary, TensorWrite,
};
use ha_ndarray::{axes, shape};
use number_general::DType;

use super::common;

async fn measure<V: TensorRead<DType = f32> + TensorReduceAll>(name: &str, view: &V) {
    super::benchmark::streams(
        &format!("slice_{name}"),
        view,
        &["sum", "row", "coordinate"],
    )
    .await;
}

pub async fn run() {
    if std::env::var_os("FENSOR_BENCH_SMOKE").is_some() {
        let (root, dir) = common::new_dir("slice_smoke").await;
        let source = common::create_dense_tensor::<f32>(
            dir,
            TensorSchema::new(f32::dtype(), shape![2, 3]).unwrap(),
        )
        .await;
        source.write_value(&[0, 1], 2.).await.unwrap();
        assert_eq!(source.sum_all().await.unwrap(), 2.);
        measure("smoke", &source.view().sum(axes![1], false).await.unwrap()).await;
        common::cleanup(&root).await;
        return;
    }

    for capacity in [1, 7, 31, 128, 4096] {
        for sparse in [false, true] {
            for cache_size in [1_000_000, 32_768] {
                let (root, tensor) = common::fixture::source(
                    "slice_benchmark",
                    shape![32, 129],
                    common::fixture::layout(sparse),
                    capacity,
                    cache_size,
                    (0..32).flat_map(|i| {
                        (0..129).map(move |j| {
                            if !sparse || (i * 129 + j) % 97 == 0 {
                                ((i + j) % 7) as f32 + 0.2
                            } else {
                                0.
                            }
                        })
                    }),
                )
                .await;
                tensor.sync().await.unwrap();
                // Check the complete fixture outside the timed region. In particular,
                // occupied traversal must not skip entries between index separators.
                let expected: f64 = (0..32)
                    .flat_map(|i| (0..129).map(move |j| (i, j)))
                    .filter(|(i, j)| !sparse || (i * 129 + j) % 97 == 0)
                    .map(|(i, j)| (((i + j) % 7) as f32 + 0.2) as f64)
                    .sum();
                let actual = tensor.sum_all().await.unwrap() as f64;
                assert!((actual - expected).abs() <= expected.abs().max(1.) * 1e-5);
                let label = format!("c{capacity}_s{sparse}_cache{cache_size}");
                measure(&format!("{label}_full"), &tensor).await;
                let view = tensor.view();
                measure(
                    &format!("{label}_rows"),
                    &view.sum(axes![1], false).await.unwrap(),
                )
                .await;
                measure(
                    &format!("{label}_columns"),
                    &view.sum(axes![0], false).await.unwrap(),
                )
                .await;
                let selected = view
                    .clone()
                    .slice(vec![AxisRange::In(0, 32, 2), AxisRange::In(1, 129, 2)].into())
                    .unwrap();
                measure(
                    &format!("{label}_strided"),
                    &selected.sum(axes![0], false).await.unwrap(),
                )
                .await;
                let unary = view.round().await.unwrap().exp().await.unwrap();
                measure(
                    &format!("{label}_unary"),
                    &unary.sum(axes![1], false).await.unwrap(),
                )
                .await;
                let nested = view
                    .sum(axes![1], true)
                    .await
                    .unwrap()
                    .sum(axes![0], false)
                    .await
                    .unwrap();
                measure(&format!("{label}_nested"), &nested).await;
                drop(tensor);
                common::cleanup(&root).await;
            }
        }
    }

    for sparse in [false, true] {
        let (root, tensor) = common::fixture::source(
            "long_slice_benchmark",
            shape![2, 8193],
            common::fixture::layout(sparse),
            128,
            1_000_000,
            (0..2).flat_map(|_| (0..8193).map(|col| if col % 257 == 0 { 1f32 } else { 0. })),
        )
        .await;
        tensor.sync().await.unwrap();
        let reduced = tensor.view().sum(axes![1], false).await.unwrap();
        assert_eq!(reduced.read_value(&[0]).await.unwrap(), 32.);
        assert_eq!(reduced.read_value(&[1]).await.unwrap(), 32.);
        measure(&format!("long_s{sparse}"), &reduced).await;
        drop(tensor);
        common::cleanup(&root).await;
    }
}
