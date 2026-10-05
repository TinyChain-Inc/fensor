//! Same workload code with task-local observations; absent from production builds.
use crate::test_support as common;

#[path = "../tests/common/benchmark.rs"]
mod benchmark;

#[path = "../tests/common/workloads.rs"]
mod workloads;

async fn observe(
    name: &str,
    operation: &str,
    temperature: &str,
    work: impl std::future::Future<Output = usize>,
) -> usize {
    crate::read_metrics::CURRENT
        .scope(
            Default::default(),
            crate::tensor::copy_metrics::CURRENT.scope(Default::default(), async {
                let count = work.await;
                crate::read_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    for (metric, value) in [
                        ("slice_requests", m.slice_requests as u128),
                        ("index_entries", m.index_entries as u128),
                        ("reduction_calls", m.reduction_calls as u128),
                        ("runs", m.runs as u128),
                        ("coordinate_resolutions", m.coordinate_resolutions as u128),
                        ("expanded_coordinates", m.expanded_coordinates as u128),
                        ("boundary_decodes", m.boundary_decodes as u128),
                        ("logical_payload_reads", m.logical_payload_reads as u128),
                        ("borrows", m.borrows as u128),
                        ("requested", m.requested as u128),
                        ("borrowed", m.borrowed as u128),
                        ("backend_calls", m.backend_calls as u128),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "count", value);
                    }
                });
                crate::tensor::copy_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    for (metric, value) in [
                        ("copy_groups", m.groups as u128),
                        ("copy_block_updates", m.block_updates as u128),
                        ("copy_constructed_blocks", m.constructed_blocks as u128),
                        ("copy_staged_elements", m.staged_elements as u128),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "count", value);
                    }
                });
                count
            }),
        )
        .await
}

#[tokio::test]
#[ignore = "isolated release profiling, not a timing gate"]
async fn profile() {
    workloads::run().await;
}
