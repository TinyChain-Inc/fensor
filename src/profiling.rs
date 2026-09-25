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
                        ("boundary_decodes", m.boundary_decodes as u128),
                        ("lookups", m.lookups as u128),
                        ("borrows", m.borrows as u128),
                        ("requested", m.requested as u128),
                        ("borrowed", m.borrowed as u128),
                        ("backend_calls", m.backend_calls as u128),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "count", value);
                    }
                    for (metric, value) in [
                        ("mapping", m.mapping.as_nanos()),
                        ("index", m.index.as_nanos()),
                        ("access", m.access.as_nanos()),
                        ("scatter", m.scatter.as_nanos()),
                        ("planning", m.planning.as_nanos()),
                        ("operands", m.operands.as_nanos()),
                        ("backend", m.backend.as_nanos()),
                        ("accumulate", m.accumulate.as_nanos()),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "ns", value);
                    }
                });
                crate::tensor::copy_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    for (metric, value) in [
                        ("copy_groups", m.groups as u128),
                        ("copy_block_updates", m.block_updates as u128),
                        ("copy_sparse_lookups", m.sparse_lookups as u128),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "count", value);
                    }
                    for (metric, value) in [
                        ("copy_initialize", m.initialize.as_nanos()),
                        ("copy_consume", m.consume.as_nanos()),
                        ("copy_update", m.update.as_nanos()),
                        ("copy_overlap", m.overlap.as_nanos()),
                    ] {
                        benchmark::record(name, operation, temperature, metric, "ns", value);
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
