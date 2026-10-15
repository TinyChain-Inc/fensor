//! Workload selection is confined to the benchmark executable, not tensor execution.
use super::{benchmark, common};

#[path = "benchmark_source.rs"]
mod benchmark_source;

#[path = "completion_cases.rs"]
mod completion_cases;

#[path = "copy_cases.rs"]
mod copy_cases;

#[path = "pipeline_cases.rs"]
mod pipeline_cases;

#[path = "read_ahead_cases.rs"]
mod read_ahead_cases;

#[path = "read_cases.rs"]
mod read_cases;

#[path = "slice_cases.rs"]
mod slice_cases;

#[path = "sparse_cases.rs"]
mod sparse_cases;

#[path = "storage_cases.rs"]
mod storage_cases;

pub async fn run() {
    let selected = std::env::var("FENSOR_BENCH_SUITE").unwrap_or_else(|_| "all".into());
    match selected.as_str() {
        "read-ahead" => read_ahead_cases::run().await,
        "matrix" => read_cases::run().await,
        "reduction" => slice_cases::run().await,
        "completion" => completion_cases::run().await,
        "owned" => completion_cases::owned().await,
        "pipeline" => pipeline_cases::run().await,
        "copy" => copy_cases::run().await,
        "sparse" => sparse_cases::run().await,
        "storage" => storage_cases::storage().await,
        "mutation" => {
            storage_cases::mutation().await;
            storage_cases::replacement().await;
        }
        "traversal" => storage_cases::traversal().await,
        "reopen" => storage_cases::reopen().await,
        "all" => {
            read_ahead_cases::run().await;
            read_cases::run().await;
            slice_cases::run().await;
            completion_cases::run().await;
            completion_cases::owned().await;
            pipeline_cases::run().await;
            copy_cases::run().await;
            sparse_cases::run().await;
            storage_cases::storage().await;
            storage_cases::mutation().await;
            storage_cases::replacement().await;
            storage_cases::traversal().await;
            storage_cases::reopen().await;
        }
        _ => panic!("unknown benchmark suite {selected}"),
    }
}
