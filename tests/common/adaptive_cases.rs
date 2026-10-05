//! Reproducible storage-shape workloads; measurements are evidence, not thresholds.
use std::time::Instant;

use fensor::{Layout, Tensor, TensorRead, TensorSchema, TensorSource, TensorWrite};
use number_general::DType;

use super::{benchmark, common};

pub async fn storage() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    for (name, mut shape, axis, full) in [
        ("vector", vec![128], 0, false),
        ("narrow", vec![128, 3], 0, false),
        ("trailing", vec![3, 128], 1, false),
        ("low_occupancy", vec![16, 4096], 0, false),
        ("high_occupancy", vec![16, 4096], 0, true),
    ] {
        if smoke && shape[0] == 4096 {
            shape[0] = 16;
        }
        common::counters::reset_traffic();
        let (root, dir) = common::new_dir(name).await;
        let t = Tensor::<common::FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), shape.into()).unwrap(),
            Layout::Sparse { axis: Some(axis) },
            4096,
        )
        .await
        .unwrap();
        let g = t.storage_geometry();
        let start = Instant::now();
        for id in 0..g.block_count() {
            let mut values = vec![0.; g.block_len()];
            if full {
                values.fill(1.);
            } else {
                values[0] = 1.;
            }
            t.replace_logical_block(id, values).await.unwrap();
        }
        let write_only = start.elapsed();
        t.sync().await.unwrap();
        let write = start.elapsed();
        let write_sync = write - write_only;
        let start = Instant::now();
        for id in 0..g.block_count() {
            t.read_logical_block(id).await.unwrap();
        }
        let scan = start.elapsed();
        let start = Instant::now();
        for _ in 0..workload_repetitions(1) {
            for id in 0..g.block_count() {
                let coord: Vec<_> = g
                    .block_bounds(id)
                    .unwrap()
                    .into_iter()
                    .map(|(start, _)| start)
                    .collect();
                t.write_value(&coord, 2.).await.unwrap();
            }
        }
        t.sync().await.unwrap();
        let point = start.elapsed();
        let (copy_root, copy_dir) = common::new_dir("adaptive_copy").await;
        let start = Instant::now();
        let copy = Tensor::<common::FsEntry, f64>::copy_from(copy_dir, &t, 4096)
            .await
            .unwrap();
        let construct_time = start.elapsed();
        copy.sync().await.unwrap();
        let sync_time = start.elapsed() - construct_time;
        let copy_time = start.elapsed();
        benchmark::storage(name, "copy", &copy_root).await;
        let copy_geometry = copy.storage_geometry();
        for id in 0..g.block_count() {
            let coord: Vec<_> = g
                .block_bounds(id)
                .unwrap()
                .into_iter()
                .map(|(start, _)| start)
                .collect();
            assert_eq!(
                copy.read_value(&coord).await.unwrap(),
                t.read_value(&coord).await.unwrap()
            );
        }
        for (metric, value) in [
            ("blocks", copy_geometry.block_count() as u128),
            ("block_elements", copy_geometry.block_len() as u128),
        ] {
            benchmark::record(name, "storage", "copy", metric, "count", value);
        }
        common::cleanup(&copy_root).await;
        benchmark::storage(name, "populated", &root).await;
        let start = Instant::now();
        for id in 0..g.block_count() {
            t.replace_logical_block(id, vec![0.; g.block_len()])
                .await
                .unwrap();
        }
        t.sync().await.unwrap();
        let deletion = start.elapsed();
        let traffic = common::counters::snapshot_traffic();
        for (operation, elapsed) in [
            ("write", write_only),
            ("sync", write_sync),
            ("write_sync", write),
            ("scan", scan),
            ("point_sync", point),
            ("copy_construct", construct_time),
            ("copy_sync", sync_time),
            ("copy_total", copy_time),
            ("delete_sync", deletion),
        ] {
            benchmark::record(name, operation, "first", "wall", "ns", elapsed.as_nanos());
        }
        for (metric, value) in [
            ("blocks", g.block_count() as u128),
            ("block_elements", g.block_len() as u128),
            ("cache_budget", 1_000_000),
            ("loads", traffic.loads as u128),
            ("saves", traffic.saves as u128),
        ] {
            benchmark::record(name, "storage", "populated", metric, "count", value);
        }
        common::cleanup(&root).await;
    }
}

pub async fn mutation() {
    use fensor::{StorageGeometry, TensorWriteBulk};
    for (layout, shapes) in [
        (Layout::Dense, [vec![4, 16], vec![16, 16]]),
        (Layout::Sparse { axis: Some(0) }, [vec![1, 16], vec![1, 64]]),
    ] {
        for block_shape in shapes {
            for scalar in [true, false] {
                common::counters::reset_traffic();
                let (root, dir) = common::new_dir("bulk_mutation").await;
                let geometry = StorageGeometry::new(
                    TensorSchema::new(f64::dtype(), vec![16, 64].into()).unwrap(),
                    layout,
                    block_shape.clone().into(),
                )
                .unwrap();
                let tensor = Tensor::<common::FsEntry, f64>::create_with_geometry(dir, geometry)
                    .await
                    .unwrap();
                let start = Instant::now();
                for value in [1., 0., 2.] {
                    if scalar {
                        for coord in fensor::row_major_coords(&[16, 64]).unwrap() {
                            tensor.write_value(&coord, value).await.unwrap();
                        }
                    } else {
                        tensor.fill(value).await.unwrap();
                    }
                }
                let mutation = start.elapsed();
                tensor.sync().await.unwrap();
                let elapsed = start.elapsed();
                let sync = elapsed - mutation;
                assert_eq!(tensor.read_value(&[15, 63]).await.unwrap(), 2.);
                let traffic = common::counters::snapshot_traffic();
                let name = format!(
                    "mutation_{}_{}x{}_scalar{scalar}",
                    layout_name(layout),
                    block_shape[0],
                    block_shape[1]
                );
                for (operation, duration) in [
                    ("mutate", mutation),
                    ("sync", sync),
                    ("mutate_sync", elapsed),
                ] {
                    benchmark::record(&name, operation, "first", "wall", "ns", duration.as_nanos());
                }
                for (metric, value) in [("loads", traffic.loads), ("saves", traffic.saves)] {
                    benchmark::record(
                        &name,
                        "mutate_sync",
                        "first",
                        metric,
                        "count",
                        value as u128,
                    );
                }
                common::cleanup(&root).await;
            }
        }
    }
}

pub async fn replacement() {
    let (root, dir) = common::new_dir("dense_replace_measurement").await;
    let tensor = Tensor::<common::FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), vec![65536].into()).unwrap(),
        Layout::Dense,
        4096,
    )
    .await
    .unwrap();
    common::counters::reset_traffic();
    let start = Instant::now();
    for _ in 0..workload_repetitions(64) {
        for id in 0..tensor.storage_geometry().block_count() {
            tensor
                .replace_logical_block(id, vec![1.; 4096])
                .await
                .unwrap();
        }
    }
    let replace = start.elapsed();
    tensor.sync().await.unwrap();
    let traffic = common::counters::snapshot_traffic();
    benchmark::record(
        "dense_replacement",
        "replace",
        "warm",
        "wall",
        "ns",
        replace.as_nanos(),
    );
    for (metric, value) in [("loads", traffic.loads), ("saves", traffic.saves)] {
        benchmark::record(
            "dense_replacement",
            "replace_sync",
            "warm",
            metric,
            "count",
            value as u128,
        );
    }
    common::cleanup(&root).await;
}

pub async fn traversal() {
    use fensor::{TensorExpression, TensorReduceAll, TensorTransform};
    use futures::TryStreamExt;
    let repeats = workload_repetitions(4);
    for layout in [
        Layout::Dense,
        Layout::Sparse { axis: Some(0) },
        Layout::Sparse { axis: Some(1) },
    ] {
        let (root, dir) = common::new_dir("traversal_read").await;
        let tensor = Tensor::<common::FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), vec![128, 129].into()).unwrap(),
            layout,
            256,
        )
        .await
        .unwrap();
        let g = tensor.storage_geometry();
        for id in 0..g.block_count() {
            let mut values = vec![0.; g.block_len()];
            for (offset, coord) in common::block_entries(&g, id) {
                values[offset] = if coord[1] % 4 == 0 { 2. } else { 0. };
            }
            tensor.replace_logical_block(id, values).await.unwrap();
        }
        let start = Instant::now();
        for _ in 0..repeats.div_ceil(4) {
            for i in 0..128 {
                assert_eq!(tensor.read_value(&[i, 0]).await.unwrap(), 2.);
            }
        }
        let point = start.elapsed();
        let start = Instant::now();
        for _ in 0..repeats {
            assert_eq!(
                tensor
                    .read_blocks()
                    .unwrap()
                    .try_concat()
                    .await
                    .unwrap()
                    .len(),
                128 * 129
            );
        }
        let scan = start.elapsed();
        let start = Instant::now();
        for _ in 0..repeats {
            assert_eq!(tensor.sum_all().await.unwrap(), 128. * 33. * 2.);
        }
        let reduction = start.elapsed();
        let start = Instant::now();
        for _ in 0..repeats {
            let value = TensorExpression::new(tensor.view())
                .unwrap()
                .transpose(None)
                .unwrap()
                .flip(0)
                .unwrap();
            assert_eq!(
                value
                    .into_blocks()
                    .unwrap()
                    .try_concat()
                    .await
                    .unwrap()
                    .len(),
                128 * 129
            );
        }
        let transformed = start.elapsed();
        let name = format!("traversal_{}", layout_name(layout));
        for (operation, elapsed) in [
            ("point", point),
            ("scan", scan),
            ("reduction", reduction),
            ("transformed", transformed),
        ] {
            benchmark::record(&name, operation, "warm", "wall", "ns", elapsed.as_nanos());
        }
        benchmark::record(
            &name,
            "traversal",
            "warm",
            "repeats",
            "count",
            repeats as u128,
        );
        common::cleanup(&root).await;
    }
    Box::pin(sparse_output()).await;
    Box::pin(sparse_selections()).await;
}

async fn sparse_output() {
    use fensor::TensorExpression;
    use futures::TryStreamExt;

    let repeats = workload_repetitions(32);
    for rank in [1, 20] {
        let mut shape = vec![1; rank];
        shape[rank - 1] = 32768;
        let (_root, tensor) = common::fixture::source(
            "sparse_output",
            shape.into(),
            Layout::Dense,
            4096,
            1_000_000,
            (0..32768).map(|offset| if offset % 1024 == 0 { 2f64 } else { 0. }),
        )
        .await;
        let expression = TensorExpression::new(tensor).unwrap();
        benchmark::measure(
            &format!("sparse_output_rank{rank}"),
            "owned_sparse",
            "warm",
            async {
                let mut count = 0;
                for _ in 0..repeats {
                    let mut stream = expression.clone().into_sparse_elements().unwrap();
                    let mut emitted = 0;
                    while let Some((coord, value)) = stream.try_next().await.unwrap() {
                        assert_eq!(coord.len(), rank);
                        assert!(coord[..rank - 1].iter().all(|&axis| axis == 0));
                        assert_eq!(coord[rank - 1], emitted * 1024);
                        assert_eq!(value, 2.);
                        emitted += 1;
                    }
                    assert_eq!(emitted, 32);
                    count += emitted as usize;
                }
                count
            },
        )
        .await;
    }
}

async fn sparse_selections() {
    use fensor::{AxisRange, TensorExpression, TensorTransform};
    use futures::TryStreamExt;

    let repeats = workload_repetitions(128);
    let (_root, tensor) = common::fixture::source(
        "sparse_dense_selections",
        vec![16, 4096].into(),
        Layout::Sparse { axis: Some(0) },
        4096,
        1_000_000,
        std::iter::repeat_n(1f64, 65536),
    )
    .await;
    assert_eq!(tensor.storage_geometry().block_len(), 4096);
    let expression = TensorExpression::new(tensor).unwrap();
    // Each selected physical chunk contains 32 KiB of dense numerical values.
    // Visit every block repeatedly while consuming only a point or eight values.
    for width in [1, 8] {
        let selections: Vec<_> = (0..16)
            .map(|block| {
                expression
                    .clone()
                    .slice(vec![AxisRange::At(block), AxisRange::In(31, 31 + width, 1)].into())
                    .unwrap()
            })
            .collect();
        benchmark::measure(
            &format!("sparse_dense_selection{width}"),
            "read",
            "warm",
            async {
                let mut count = 0;
                for _ in 0..repeats {
                    for selection in &selections {
                        if width == 1 {
                            assert_eq!(selection.read_value(&[0]).await.unwrap(), 1.);
                            count += 1;
                        } else {
                            let mut stream = selection.read_blocks().unwrap();
                            let mut selected = 0;
                            while let Some(values) = stream.try_next().await.unwrap() {
                                assert!(values.iter().all(|&value| value == 1.));
                                selected += values.len();
                            }
                            assert_eq!(selected, width as usize);
                            count += selected;
                        }
                    }
                }
                count
            },
        )
        .await;
    }
}

fn workload_repetitions(default: usize) -> usize {
    if std::env::var_os("FENSOR_BENCH_SMOKE").is_some() {
        return 1;
    }
    let count = std::env::var("FENSOR_WORKLOAD_REPEATS")
        .ok()
        .map(|value| value.parse().expect("positive repeat count"))
        .unwrap_or(default);
    assert!(count > 0);
    count
}

pub async fn reopen() {
    for axis in [0, 1] {
        let (root, dir) = common::new_dir("strict_reopen_workload").await;
        let tensor = Tensor::<common::FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), vec![64, 65].into()).unwrap(),
            Layout::Sparse { axis: Some(axis) },
            256,
        )
        .await
        .unwrap();
        let geometry = tensor.storage_geometry();
        for id in 0..geometry.block_count() {
            let mut values = vec![0.; geometry.block_len()];
            for offset in geometry.block_offsets(id).unwrap() {
                values[offset] = 1.;
            }
            tensor.replace_logical_block(id, values).await.unwrap();
        }
        tensor.sync_all().await.unwrap();
        drop(tensor);
        let start = Instant::now();
        for _ in 0..10 {
            let loaded = Tensor::<common::FsEntry, f64>::load(common::open_dir(&root).unwrap())
                .await
                .unwrap();
            let reopened = loaded.storage_geometry();
            assert_eq!(reopened.schema(), geometry.schema());
            assert_eq!(reopened.layout(), geometry.layout());
            assert_eq!(reopened.block_shape(), geometry.block_shape());
        }
        benchmark::record(
            &format!("reopen_axis{axis}"),
            "reopen",
            "first",
            "wall",
            "ns",
            start.elapsed().as_nanos(),
        );
        common::cleanup(&root).await;
    }
}

fn layout_name(layout: Layout) -> String {
    match layout {
        Layout::Dense => "dense".into(),
        Layout::Sparse { axis } => format!("sparse_axis{}", axis.unwrap_or(0)),
    }
}
