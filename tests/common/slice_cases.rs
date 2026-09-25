//! Identical cases for production timing and test-only structural profiling.

use std::time::Instant;

use fensor::{
    Layout, Tensor, TensorRead, TensorReduce, TensorReduceAll, TensorSchema, TensorTransform,
    TensorUnary, TensorWrite,
};
use ha_ndarray::{AxisRange, axes, shape};
use number_general::DType;

use super::common::{self, FsEntry, counters};

async fn measure<V: TensorRead<DType = f32> + TensorReduceAll>(name: &str, view: &V) {
    for mode in ["sum", "row", "coordinate"] {
        for temperature in ["first", "warm"] {
            counters::reset_traffic();
            let started = Instant::now();
            let count = super::observe(name, mode, temperature, async {
                if mode == "sum" {
                    std::hint::black_box(view.sum_all().await.unwrap());
                    1
                } else {
                    super::benchmark::consume(view, mode).await
                }
            })
            .await;
            let traffic = counters::snapshot_traffic();
            println!(
                "SLICE,{name},{mode},{temperature},{},{count},{},{}",
                started.elapsed().as_nanos(),
                traffic.loads,
                traffic.saves
            );
        }
    }
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
                let (root, _) = common::new_dir("slice_benchmark").await;
                let cache =
                    freqfs::Cache::new(cache_size, None, 0, std::time::Duration::from_secs(1));
                let tensor = Tensor::<FsEntry, f32>::create(
                    cache.load(root.clone()).unwrap(),
                    TensorSchema::new(f32::dtype(), shape![32, 129]).unwrap(),
                    if sparse {
                        Layout::Sparse { axis: None }
                    } else {
                        Layout::Dense
                    },
                    capacity,
                )
                .await
                .unwrap();

                for i in 0..32 {
                    for j in 0..129 {
                        if !sparse || (i * 129 + j) % 97 == 0 {
                            tensor
                                .write_value(&[i, j], ((i + j) % 7) as f32 + 0.2)
                                .await
                                .unwrap();
                        }
                    }
                }
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
        let (root, dir) = common::new_dir("long_slice_benchmark").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir,
            TensorSchema::new(f32::dtype(), shape![2, 8193]).unwrap(),
            if sparse {
                Layout::Sparse { axis: None }
            } else {
                Layout::Dense
            },
            128,
        )
        .await
        .unwrap();

        for row in 0..2 {
            for col in (0..8193).step_by(257) {
                tensor.write_value(&[row, col], 1.).await.unwrap();
            }
        }
        tensor.sync().await.unwrap();
        let reduced = tensor.view().sum(axes![1], false).await.unwrap();
        assert_eq!(reduced.read_value(&[0]).await.unwrap(), 32.);
        assert_eq!(reduced.read_value(&[1]).await.unwrap(), 32.);
        measure(&format!("long_s{sparse}"), &reduced).await;
        drop(tensor);
        common::cleanup(&root).await;
    }
}
