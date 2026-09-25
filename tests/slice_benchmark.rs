#[path = "common/benchmark.rs"]
mod benchmark;

#[path = "common/slice_cases.rs"]
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
#[ignore = "release slice benchmark, not a timing gate"]
async fn slice_benchmark() {
    cases::run().await;
}
