//! Shared evaluation-only cases; production timing and test-only profiling use the same inputs.

use std::time::Instant;

use fensor::{TensorMatMul, TensorRead, TensorTransform, TensorUnary};
use ha_ndarray::shape;

use super::benchmark::consume;
use super::benchmark_source::source;
use super::common::counters;

async fn measure<V: TensorRead<DType = f32>>(name: &str, view: &V) {
    for mode in ["row", "coordinate"] {
        for temperature in ["first", "warm"] {
            counters::reset_traffic();
            let start = Instant::now();
            let count = super::observe(name, mode, temperature, consume(view, mode)).await;
            let traffic = counters::snapshot_traffic();
            println!(
                "READ,{name},{mode},{temperature},{},{count},{},{}",
                start.elapsed().as_micros(),
                traffic.loads,
                traffic.saves
            );
        }
    }
}

pub async fn run() {
    if std::env::var_os("FENSOR_BENCH_SMOKE").is_some() {
        let a = source(2, 3, false, 7, 1, 1_000_000).await;
        let b = source(3, 2, false, 7, 1, 1_000_000).await;
        measure("smoke", &a.view().matmul(&b.view()).await.unwrap()).await;
        return;
    }

    for (name, m, k, n, cap, ls, rs, density, cache) in [
        ("square", 32, 129, 32, 128, false, false, 1, 1_000_000),
        ("square_128", 128, 129, 128, 128, false, false, 1, 1_000_000),
        (
            "square_256",
            256,
            257,
            256,
            4096,
            false,
            false,
            1,
            1_000_000,
        ),
        ("wide", 32, 17, 4097, 4096, false, false, 1, 1_000_000),
        ("tall", 129, 129, 1, 31, false, false, 1, 1_000_000),
        ("dot", 1, 4097, 1, 128, false, false, 1, 1_000_000),
        ("tiny_blocks", 8, 17, 8, 1, false, false, 1, 1_000_000),
        ("odd_blocks", 9, 127, 7, 7, false, false, 1, 1_000_000),
        ("mixed_left", 32, 128, 32, 31, true, false, 7, 1_000_000),
        ("mixed_right", 32, 128, 32, 128, false, true, 3, 1_000_000),
        ("sparse", 32, 129, 32, 31, true, true, 7, 1_000_000),
        ("empty", 8, 129, 8, 7, true, true, 0, 1_000_000),
        ("pressure", 9, 129, 7, 128, false, false, 1, 2048),
    ] {
        let left = source(m, k, ls, cap, density, cache).await;
        let right = source(k, n, rs, cap.max(7), density, cache).await;
        measure(name, &left.view().matmul(&right.view()).await.unwrap()).await;
    }

    let a = source(18, 17, false, 31, 1, 1_000_000).await;
    let b = source(34, 7, false, 128, 1, 1_000_000).await;
    let av = a.view().reshape(shape![2, 9, 17]).unwrap();
    let bv = b.view().reshape(shape![2, 17, 7]).unwrap();
    let product = av.matmul(&bv).await.unwrap();
    measure("batched", &product).await;
    measure(
        "transposed",
        &product
            .clone()
            .transpose(Some(ha_ndarray::axes![0, 2, 1]))
            .unwrap(),
    )
    .await;
    measure(
        "repeated",
        &product
            .clone()
            .slice(ha_ndarray::range![
                ha_ndarray::AxisRange::At(0),
                ha_ndarray::AxisRange::Of(vec![8, 0, 8, 1].into()),
                ha_ndarray::AxisRange::In(0, 7, 1)
            ])
            .unwrap(),
    )
    .await;
    measure("unary", &product.round().await.unwrap()).await;
    let c = source(7, 3, false, 7, 1, 1_000_000).await;
    let cv = c.view().broadcast(shape![2, 7, 3]).unwrap();
    measure("nested", &product.matmul(&cv).await.unwrap()).await;
    let a = source(32, 17, false, 31, 1, 1_000_000).await;
    let b = source(17, 32, false, 128, 1, 1_000_000).await;
    let diagonal = a
        .view()
        .matmul(&b.view())
        .await
        .unwrap()
        .reshape(shape![1024])
        .unwrap()
        .slice(ha_ndarray::range![ha_ndarray::AxisRange::In(0, 1024, 33)])
        .unwrap();
    measure("diagonal", &diagonal).await;
    let left = source(17, 9, false, 31, 1, 1_000_000).await;
    let right = source(17, 7, false, 128, 1, 1_000_000).await;
    let left = left.view().transpose(None).unwrap();
    measure(
        "operand_transpose",
        &left.matmul(&right.view()).await.unwrap(),
    )
    .await;
    measure(
        "operand_flip",
        &left
            .flip(1)
            .unwrap()
            .matmul(&right.view().flip(0).unwrap())
            .await
            .unwrap(),
    )
    .await;
    let leaf = source(32, 128, false, 4096, 1, 1_000_000).await;
    measure("leaf_contiguous", &leaf.view()).await;
    measure("leaf_transpose", &leaf.view().transpose(None).unwrap()).await;
    measure("leaf_flip", &leaf.view().flip(1).unwrap()).await;
    measure(
        "leaf_broadcast",
        &leaf
            .view()
            .slice(ha_ndarray::range![
                ha_ndarray::AxisRange::In(0, 32, 1),
                ha_ndarray::AxisRange::In(0, 1, 1)
            ])
            .unwrap()
            .broadcast(shape![32, 128])
            .unwrap(),
    )
    .await;
    measure(
        "leaf_gather",
        &leaf
            .view()
            .slice(ha_ndarray::range![
                ha_ndarray::AxisRange::Of(vec![31, 0, 31, 4].into()),
                ha_ndarray::AxisRange::In(0, 128, 1)
            ])
            .unwrap(),
    )
    .await;
}
