//! Shared benchmark stream consumption; callers own timing and observation.

use std::time::Instant;

use fensor::{Tensor, TensorRead, TensorReduceAll};
use futures::TryStreamExt;

use super::common::{self, FsEntry, counters};

pub async fn consume<V: TensorRead>(view: &V, mode: &str) -> usize {
    let mut count = 0;
    match mode {
        "row" => {
            let mut stream = view.read_blocks().unwrap();

            while let Some(values) = stream.try_next().await.unwrap() {
                count += values.len();
                std::hint::black_box(values);
            }
        }
        "coordinate" => {
            let mut stream = view.read_coordinate_blocks().unwrap();

            while let Some((coords, values)) = stream.try_next().await.unwrap() {
                count += values.len();
                std::hint::black_box((coords, values));
            }
        }
        _ => panic!("unknown benchmark stream mode {mode}"),
    }
    assert_eq!(
        count as u64,
        view.size().unwrap(),
        "complete {mode} consumption"
    );
    count
}

/// Long-form records keep timing and structural observations in one schema.
pub fn record(
    name: &str,
    operation: &str,
    temperature: &str,
    metric: &str,
    unit: &str,
    value: u128,
) {
    println!("FENSOR,{name},{operation},{temperature},{metric},{unit},{value}");
}

async fn measure(
    name: &str,
    operation: &str,
    temperature: &str,
    work: impl std::future::Future<Output = usize>,
) {
    counters::reset_traffic();
    let start = Instant::now();
    let count = super::observe(name, operation, temperature, work).await;
    let elapsed = start.elapsed();
    let traffic = counters::snapshot_traffic();
    for (metric, value) in [
        ("elements", count as u128),
        ("loads", traffic.loads as u128),
        ("saves", traffic.saves as u128),
    ] {
        record(name, operation, temperature, metric, "count", value);
    }
    record(
        name,
        operation,
        temperature,
        "wall",
        "ns",
        elapsed.as_nanos(),
    );
}

pub async fn streams<V: TensorRead<DType = f32> + TensorReduceAll>(
    name: &str,
    view: &V,
    modes: &[&str],
) {
    for mode in modes {
        for temperature in ["first", "warm"] {
            measure(name, mode, temperature, async {
                let value = match *mode {
                    "sum" => view.sum_all().await.unwrap(),
                    "product" => view.product_all().await.unwrap(),
                    "min" => view.min_all().await.unwrap(),
                    "max" => view.max_all().await.unwrap(),
                    _ => return consume(view, mode).await,
                };
                std::hint::black_box(value);
                1
            })
            .await;
        }
    }
}

pub async fn copy<V: TensorRead<DType = f32>>(name: &str, view: &V, capacity: usize, cache: usize) {
    let (root, dir) = common::new_dir("benchmark_copy").await;
    drop(dir);
    let dir = freqfs::Cache::<FsEntry>::new(cache, None, 0, std::time::Duration::from_secs(1))
        .load(root.clone())
        .unwrap();
    let name = format!("{name}_dst{capacity}_cache{cache}");
    measure(&name, "copy-sync", "new", async {
        let output = Tensor::copy_from(dir, view, capacity).await.unwrap();
        let sync = Instant::now();
        output.sync().await.unwrap();
        record(
            &name,
            "copy-sync",
            "new",
            "sync",
            "ns",
            sync.elapsed().as_nanos(),
        );
        view.size().unwrap() as usize
    })
    .await;
    common::cleanup(&root).await;
}
