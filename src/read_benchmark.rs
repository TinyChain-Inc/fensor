//! The same filesystem cases with test-only, task-local phase observations.

use crate::test_support as common;

#[path = "../tests/common/benchmark.rs"]
mod benchmark;

#[path = "../tests/common/benchmark_source.rs"]
mod benchmark_source;

#[path = "../tests/common/read_cases.rs"]
mod cases;

#[path = "../tests/common/slice_cases.rs"]
mod slice_cases;

async fn observe(
    name: &str,
    mode: &str,
    temperature: &str,
    work: impl std::future::Future<Output = usize>,
) -> usize {
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let count = work.await;

            crate::read_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                println!(
                    "SLICE_PROFILE,{name},{mode},{temperature},{},{},{}",
                    m.slice_requests,
                    m.index_entries,
                    m.reduction_calls,
                );
                println!(
                    "PROFILE,{name},{mode},{temperature},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    m.runs,
                    m.coordinate_resolutions,
                    m.boundary_decodes,
                    m.lookups,
                    m.borrows,
                    m.requested,
                    m.borrowed,
                    m.backend_calls,
                    m.mapping.as_nanos(),
                    m.index.as_nanos(),
                    m.access.as_nanos(),
                    m.scatter.as_nanos(),
                    m.planning.as_nanos(),
                    m.operands.as_nanos(),
                    m.backend.as_nanos(),
                    m.accumulate.as_nanos(),
                );
            });

            count
        })
        .await
}

#[tokio::test]
#[ignore = "release profiling benchmark, not a timing gate"]
async fn read_profile() {
    cases::run().await;
}

#[tokio::test]
#[ignore = "release slice profiling, not a timing gate"]
async fn slice_profile() {
    slice_cases::run().await;
}
