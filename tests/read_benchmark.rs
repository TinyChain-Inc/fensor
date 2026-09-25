//! Evaluation-only release benchmark. No destination tensors are created.

#[path = "common/benchmark.rs"]
mod benchmark;

#[path = "common/benchmark_source.rs"]
mod benchmark_source;

#[path = "common/read_cases.rs"]
mod cases;

mod common;

async fn observe(
    _: &str,
    _: &str,
    _: &str,
    work: impl std::future::Future<Output = usize>,
) -> usize {
    work.await
}

#[tokio::test]
#[ignore = "release evaluation benchmark, not a timing gate"]
async fn read_benchmark() {
    cases::run().await;
}
