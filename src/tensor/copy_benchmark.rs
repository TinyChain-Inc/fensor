//! Isolated copy-phase measurements; no production instrumentation.

use std::time::{Duration, Instant};

use freqfs::Cache;
use ha_ndarray::shape;
use number_general::{FloatType, NumberType};

use crate::test_support::{FsEntry as TestFE, cleanup, counters, new_dir};
use crate::{Layout, Tensor, TensorSchema, TensorTransform, TensorWrite};

use super::copy_metrics;

#[tokio::test]
#[ignore = "release copy phase benchmark, not a timing gate"]
async fn copy_phase_benchmark() {
    profile_copy(9, 17, [8192, 1_000_000]).await;
}

#[tokio::test]
#[ignore = "multi-batch copy profiling, not a timing gate"]
async fn copy_pipeline_profile() {
    profile_copy(65, 65, [32768, 1_000_000]).await;
}

async fn profile_copy(rows: usize, columns: usize, caches: [usize; 2]) {
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (source_root, source_dir) = new_dir("copy_phase_source").await;
        let source = Tensor::<TestFE, f32>::create(
            source_dir,
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![rows, columns]).unwrap(),
            layout,
            31,
        )
        .await
        .unwrap();

        for row in 0..rows {
            for column in 0..columns {
                if column % 3 == 0 || matches!(layout, Layout::Dense) {
                    source
                        .write_value(&[row as u64, column as u64], (row + column + 1) as f32)
                        .await
                        .unwrap();
                }
            }
        }
        source.sync().await.unwrap();
        let view = source.view().transpose(None).unwrap();

        for cache in caches {
            for capacity in [1, 7, 31, 128, 4096] {
                let (root, dir) = new_dir("copy_phase_output").await;
                drop(dir);
                let dir = Cache::<TestFE>::new(cache, None, 0, Duration::from_secs(1))
                    .load(root.clone())
                    .unwrap();
                copy_metrics::CURRENT
                    .scope(Default::default(), async {
                        counters::reset_traffic();
                        let start = Instant::now();
                        let output = Tensor::copy_from(dir, &view, capacity).await.unwrap();
                        let sync = Instant::now();
                        output.sync().await.unwrap();
                        let sync = sync.elapsed();
                        copy_metrics::CURRENT.with(|m| {
                            let m = m.borrow();
                            let traffic = counters::snapshot_traffic();
                            println!(
                                "COPY_OVERLAP,{layout:?},{capacity},{cache},{}",
                                m.overlap.as_nanos(),
                            );
                            println!(
                                "COPY_PHASE,{layout:?},{capacity},{cache},{},{},{},{},{},{},{},{},{},{}",
                                start.elapsed().as_micros(),
                                m.initialize.as_micros(),
                                m.consume.as_micros(),
                                m.update.as_micros(),
                                sync.as_micros(),
                                m.groups,
                                m.block_updates,
                                m.sparse_lookups,
                                traffic.loads,
                                traffic.saves,
                            );
                        });
                    })
                    .await;

                cleanup(&root).await;
            }
        }
        cleanup(&source_root).await;
    }
}
