//! Matched resident layout controls, shared by timing and test-local profiling.

use fensor::{
    Layout, TensorAbs, TensorExpression, TensorRead, TensorReduceAll, TensorSource, TensorTransform,
};

use super::{benchmark, common};

async fn consume<E: TensorRead<DType = f32> + TensorReduceAll>(
    name: &str,
    source: &E,
    expected: f32,
    repetitions: usize,
) {
    for mode in ["row", "coordinate", "sum"] {
        if mode == "sum" {
            assert_eq!(source.sum_all().await.unwrap(), expected);
        } else {
            benchmark::consume(source, mode).await;
        }
        benchmark::measure(name, mode, "warm", async {
            let mut count = 0;
            for _ in 0..repetitions {
                if mode == "sum" {
                    assert_eq!(source.sum_all().await.unwrap(), expected);
                    count += 1;
                } else {
                    count += benchmark::consume(source, mode).await;
                }
            }
            count
        })
        .await;
    }
}

pub async fn run() {
    println!("layout: Tokio current-thread runtime; cache 67108864 bytes");
    let repetitions = if std::env::var_os("FENSOR_BENCH_SMOKE").is_some() {
        1
    } else {
        4
    };
    for layout in [
        Layout::Dense,
        Layout::Sparse { axis: Some(0) },
        Layout::Sparse { axis: None },
    ] {
        let capacities: &[usize] = if matches!(layout, Layout::Sparse { axis: None }) {
            &[1]
        } else {
            &[64, 512, 4096]
        };
        for &capacity in capacities {
            for full in [false, true] {
                let shape = [8, 2049];
                let values =
                    (0..shape[0] * shape[1]).map(|i| if full || i % 257 == 0 { 1f32 } else { 0.0 });
                let expected = values.clone().sum();
                let (root, tensor) = common::fixture::source(
                    "layout",
                    shape.as_slice().into(),
                    layout,
                    capacity,
                    67108864,
                    values,
                )
                .await;
                tensor.sync().await.unwrap();
                let kind = match layout {
                    Layout::Dense => "dense",
                    Layout::Sparse { axis: Some(_) } => "sparse",
                    Layout::Sparse { axis: None } => "scalar_sparse",
                };
                let occupancy = if full { "full" } else { "isolated" };
                let name = format!("layout/{kind}/cap{capacity}/{occupancy}");
                let geometry = tensor.storage_geometry();
                for (metric, value) in [
                    ("block_elements", geometry.block_len()),
                    ("blocks", geometry.block_count() as usize),
                    ("repetitions", repetitions),
                ] {
                    benchmark::record(
                        &name,
                        "geometry",
                        "resident",
                        metric,
                        "count",
                        value as u128,
                    );
                }
                benchmark::storage(&name, "resident", &root).await;
                {
                    let unary = tensor.view().transpose(None).unwrap().abs().await.unwrap();
                    consume(&format!("{name}/borrowed"), &unary, expected, repetitions).await;
                    let owned = TensorExpression::new(tensor.view().abs().await.unwrap())
                        .unwrap()
                        .transpose(None)
                        .unwrap();
                    consume(&format!("{name}/owned"), &owned, expected, repetitions).await;
                    let mut nested = TensorExpression::new(tensor.view()).unwrap();
                    for _ in 0..4 {
                        nested = TensorExpression::new(
                            nested.transpose(None).unwrap().abs().await.unwrap(),
                        )
                        .unwrap();
                    }
                    consume(&format!("{name}/nested"), &nested, expected, repetitions).await;
                }
                drop(tensor);
                common::cleanup(&root).await;
            }
        }
    }
}
