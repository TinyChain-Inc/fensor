//! Uninstrumented production evaluation and persistence measurements.
#[path = "common/benchmark.rs"]
mod benchmark;

mod common;

#[path = "common/statistics_benchmark.rs"]
mod statistics;

#[path = "common/workloads.rs"]
mod workloads;

async fn observe(
    _: &str,
    _: &str,
    _: &str,
    work: impl std::future::Future<Output = usize>,
) -> usize {
    work.await
}

#[tokio::test]
#[ignore = "isolated release measurements, not a timing gate"]
async fn benchmark() {
    workloads::run().await;
}

/// Isolate accumulator cost from filesystem traversal. Seven samples of 2048
/// execution-sized batches include the adapter's owned-Vec allocation and report
/// median nanoseconds per value. Run with --release --nocapture --ignored.
#[test]
#[ignore = "isolated release measurements, not a timing gate"]
fn statistical_accumulators() {
    use std::hint::black_box;
    use std::time::Instant;

    use fensor::reduce::{Mean, Norm, ReduceOp, StandardDeviation};

    fn measure<R: ReduceOp<f64>>(name: &str, op: R, values: &[f64]) {
        let mut samples = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..2048 {
                let state = op.partial(black_box(values.to_vec())).unwrap();
                black_box(R::finish(state).unwrap());
            }

            samples.push(start.elapsed().as_secs_f64() * 1e9 / (2048 * values.len()) as f64);
        }

        samples.sort_by(f64::total_cmp);
        println!("{name}: median_ns_per_value={:.4}", samples[3]);
    }

    let values: Vec<f64> = (0..4096).map(|i| (i % 127) as f64 - 63.).collect();
    measure("mean", Mean, &values);
    measure("norm", Norm, &values);
    measure("std", StandardDeviation, &values);
}
