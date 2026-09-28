//! Focused sparse allocation and final-value reclamation measurements.
use b_table::{TableLock, collate::Collator};
use fensor::{
    Layout, SparseIndexSchema, SparseTableSchema, Tensor, TensorMetadata, TensorRead, TensorSchema,
    TensorWrite,
};
use ha_ndarray::shape;
use number_general::DType;

use super::{benchmark, common};

async fn storage(name: &str, phase: &str, root: &std::path::Path) {
    let dir = common::open_dir(root).unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let metadata = blocks.read().await.get_file("metadata").unwrap().clone();
    let capacity = metadata
        .read::<TensorMetadata<f32>>()
        .await
        .unwrap()
        .block_shape()
        .iter()
        .product::<u64>();
    let files = blocks
        .read()
        .await
        .iter()
        .filter(|(name, _)| name.as_str() != "metadata")
        .count();
    let index = dir.read().await.get_dir("index").unwrap().clone();
    let table =
        TableLock::<SparseTableSchema, SparseIndexSchema, Collator<u64>, common::FsEntry>::load(
            SparseTableSchema::default(),
            Collator::default(),
            index,
        )
        .unwrap();
    let rows = table.read().await.count(Default::default()).await.unwrap();
    for (metric, value) in [
        ("data_files", files as u128),
        ("index_rows", rows as u128),
        (
            "allocated_payload_bytes",
            files as u128 * capacity as u128 * std::mem::size_of::<f32>() as u128,
        ),
    ] {
        benchmark::record(name, "storage", phase, metric, "count", value);
    }
}

pub async fn run() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    let capacities: &[usize] = if smoke { &[7] } else { &[1, 7, 31, 128, 4096] };
    for &capacity in capacities {
        for axis in [0, 1] {
            for cache in [16_384, 1_000_000] {
                let name = format!("sparse_axis{axis}_cap{capacity}_cache{cache}");
                let root = common::unique_tmp_dir(&name);
                tokio::fs::create_dir(&root).await.unwrap();
                let dir = freqfs::Cache::<common::FsEntry>::new(
                    cache,
                    None,
                    0,
                    std::time::Duration::from_secs(1),
                )
                .load(root.clone())
                .unwrap();
                let tensor = Tensor::<common::FsEntry, f32>::create(
                    dir,
                    TensorSchema::new(f32::dtype(), shape![8, 129]).unwrap(),
                    Layout::Sparse { axis: Some(axis) },
                    capacity,
                )
                .await
                .unwrap();
                // One populated value per sparse-axis coordinate: clearing it is
                // a final-value deletion on both baseline and revised layouts.
                for i in 0..8 {
                    tensor.write_value(&[i, i * 16], 1.).await.unwrap();
                }
                tensor.sync().await.unwrap();
                storage(&name, "populated", &root).await;
                benchmark::streams(&name, &tensor, &["row", "coordinate"]).await;
                benchmark::copy(&name, &tensor.view(), capacity, cache).await;
                benchmark::measure(&name, "clear", "populated", async {
                    for i in 0..8 {
                        tensor.write_value(&[i, i * 16], 0.).await.unwrap();
                    }
                    tensor.sync().await.unwrap();
                    8
                })
                .await;
                for i in 0..8 {
                    assert_eq!(tensor.read_value(&[i, i * 16]).await.unwrap(), 0.);
                }
                storage(&name, "cleared", &root).await;
                drop(tensor);
                common::cleanup(&root).await;
            }
        }
    }
}
