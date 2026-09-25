//! Uninstrumented production evaluation and persistence measurements.
#[path = "common/benchmark.rs"]
mod benchmark;
mod common;

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
