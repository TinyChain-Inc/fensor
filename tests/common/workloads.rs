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

#[path = "read_cases.rs"]
mod read_cases;

#[path = "slice_cases.rs"]
mod slice_cases;

pub async fn run() {
    let selected = std::env::var("FENSOR_BENCH_SUITE").unwrap_or_else(|_| "all".into());
    match selected.as_str() {
        "matrix" => read_cases::run().await,
        "reduction" => slice_cases::run().await,
        "completion" => completion_cases::run().await,
        "pipeline" => pipeline_cases::run().await,
        "copy" => copy_cases::run().await,
        "all" => {
            read_cases::run().await;
            slice_cases::run().await;
            completion_cases::run().await;
            pipeline_cases::run().await;
            copy_cases::run().await;
        }
        _ => panic!("unknown benchmark suite {selected}"),
    }
}
