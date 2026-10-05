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

pub async fn measure(
    name: &str,
    operation: &str,
    temperature: &str,
    work: impl std::future::Future<Output = usize>,
) {
    // Keep the measured future out of nested task-local instrumentation frames.
    // Allocation belongs to setup, outside the timed and counted operation.
    let work = Box::pin(work);
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

pub async fn copy<V: TensorRead<DType = f32>>(
    name: &str,
    view: &V,
    capacity: usize,
    cache: usize,
    allow_capacity_rejection: bool,
) {
    let (root, dir) = common::new_dir("benchmark_copy").await;
    drop(dir);
    let dir = freqfs::Cache::<FsEntry>::new(cache, None, 0, std::time::Duration::from_secs(1))
        .load(root.to_path_buf())
        .unwrap();
    let name = format!("{name}_dst{capacity}_cache{cache}");
    // A rejected attempt is not successful-copy latency. Keep all original
    // pressure cases, and compare wall time only when both attempts complete.
    measure(&name, "copy-attempt", "new", async {
        let mut sync_ns = 0;
        let result = async {
            let output = Tensor::copy_from(dir, view, capacity).await?;
            let sync = Instant::now();
            let result = output.sync().await;
            sync_ns = sync.elapsed().as_nanos();
            result
        }
        .await;
        let admitted = match result {
            Ok(()) => true,
            Err(fensor::Error::Io(error))
                if allow_capacity_rejection
                    && matches!(
                        (error.kind(), error.to_string().as_str()),
                        (
                            std::io::ErrorKind::OutOfMemory,
                            "retained file exceeds cache capacity"
                        ) | (
                            std::io::ErrorKind::ResourceBusy,
                            "cache capacity is exhausted"
                        )
                    ) =>
            {
                eprintln!("capacity rejection for {name}: {error}");
                false
            }
            Err(error) => panic!("copy failed for {name}: {error}"),
        };
        for (metric, value) in [
            ("admitted", u128::from(admitted)),
            ("capacity_rejected", u128::from(!admitted)),
            ("sync", sync_ns),
        ] {
            let unit = if metric == "sync" { "ns" } else { "count" };
            record(&name, "copy-attempt", "new", metric, unit, value);
        }
        if admitted {
            view.size().unwrap() as usize
        } else {
            0
        }
    })
    .await;
    common::cleanup(&root).await;
}

/// Inspect synchronized storage one native page at a time, outside measurement.
/// Encoded bytes and retained node memory are distinct costs. Row-backed sparse
/// payloads reside in table pages, so dense payload files alone are not storage.
pub async fn storage(name: &str, phase: &str, root: &std::path::Path) {
    let mut dirs = vec![root.to_owned()];
    let (mut files, mut nodes, mut payload_files) = (0u128, 0u128, 0u128);
    let (mut bytes, mut payload_bytes, mut table_bytes) = (0u128, 0u128, 0u128);
    let (mut leaves, mut leaf_rows, mut index_rows, mut retained, mut max_node) =
        (0u128, 0u128, 0u128, 0u128, 0u128);
    while let Some(dir) = dirs.pop() {
        let mut entries = tokio::fs::read_dir(dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let metadata = entry.metadata().await.unwrap();
            if metadata.is_dir() {
                dirs.push(entry.path());
                continue;
            }
            files += 1;
            bytes += u128::from(metadata.len());
            if entry.path().parent().unwrap().ends_with("blocks") {
                if entry.file_name() != "metadata" {
                    payload_files += 1;
                    payload_bytes += u128::from(metadata.len());
                }
                continue;
            }
            nodes += 1;
            table_bytes += u128::from(metadata.len());
            let file = tokio::fs::File::open(entry.path()).await.unwrap();
            let node: FsEntry = tbon::de::read_from((), file).await.unwrap();
            // The benchmark workloads use f32/f64; this is adapter-owned
            // accounting, not a native storage introspection API.
            let (leaf, rows, memory) = match node {
                FsEntry::Node(node) => node_metrics(&node),
                FsEntry::SparseF32(node) => node_metrics(&node),
                FsEntry::SparseF64(node) => node_metrics(&node),
                _ => panic!("unexpected benchmark node"),
            };
            leaves += u128::from(leaf);
            if leaf {
                leaf_rows += rows as u128;
            } else {
                index_rows += rows as u128;
            }
            retained += memory as u128;
            max_node = max_node.max(memory as u128);
        }
    }
    for (metric, value) in [
        ("files", files),
        ("nodes", nodes),
        ("payload_files", payload_files),
        ("persisted_bytes", bytes),
        ("payload_file_bytes", payload_bytes),
        ("table_bytes", table_bytes),
        ("node_leaves", leaves),
        ("node_leaf_rows", leaf_rows),
        ("node_index_rows", index_rows),
        ("decoded_node_bytes_sum", retained),
        ("max_decoded_node_bytes", max_node),
    ] {
        record(name, "storage", phase, metric, "count", value);
    }
}

fn node_metrics<V>(node: &b_table::Node<V>) -> (bool, usize, usize) {
    let (leaf, rows, children) = match node {
        b_table::Node::Leaf(rows) => (true, rows, 0),
        b_table::Node::Index(rows, children) => (false, rows, children.capacity() * 16),
    };
    let bytes = std::mem::size_of_val(node)
        + rows.capacity() * std::mem::size_of::<Vec<V>>()
        + rows
            .iter()
            .map(|row| row.capacity() * std::mem::size_of::<V>())
            .sum::<usize>()
        + children;
    (leaf, rows.len(), bytes)
}
