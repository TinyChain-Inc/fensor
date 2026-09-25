//! Integration tests for `TensorUnary` (`exp`, `ln`, `round`) via lazy
//! typed `UnaryView` chains over filesystem-backed geometric views.

use fensor::unary::{Exp, Round};
use fensor::{
    AxisRange, Error, Layout, Tensor, TensorRead, TensorSchema, TensorTransform, TensorUnary,
    TensorView, TensorViewSemantics, TensorWrite, UnaryView,
};
use futures::TryStreamExt;
use ha_ndarray::{axes, range, shape};
use number_general::{FloatType, NumberType};

use common::{FsEntry, create_dense_tensor, create_sparse_tensor, iter_coords, new_dir};

mod common;

#[tokio::test]
async fn dense_single_block_round_matches_expected_values() {
    let (_root, dir) = new_dir("dense_round_single_block").await;
    let (_out_root, out_dir) = new_dir("dense_round_single_block_out").await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, schema).await;

    let values: [((u64, u64), f32); 4] =
        [((0, 0), 1.4), ((0, 1), 1.6), ((1, 0), -1.5), ((1, 1), 2.5)];
    for ((r, c), value) in values {
        tensor.write_value(&[r, c], value).await.expect("write");
    }

    let rounded = Tensor::copy_from(
        out_dir,
        &tensor.view().round().await.expect("round should succeed"),
        1000,
    )
    .await
    .expect("copy should succeed");

    for ((r, c), value) in values {
        let expected = value.round();
        let actual = rounded.read_value(&[r, c]).await.expect("read");
        assert_eq!(actual, expected, "coord ({r},{c})");
    }
}

#[tokio::test]
async fn dense_multi_block_exp_and_ln_stream_correctly() {
    // shape [10] with max_capacity 3 -> block_shape [3], grid ceil(10/3) = 4
    // blocks, forcing the block-level copy path to walk multiple blocks.
    let (_root, dir) = new_dir("dense_multi_block_exp_ln").await;
    let (_exp_root, exp_dir) = new_dir("dense_multi_block_exp_ln_exp_out").await;
    let (_ln_root, ln_dir) = new_dir("dense_multi_block_exp_ln_ln_out").await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape![10]).expect("schema");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema, Layout::Dense, 3)
        .await
        .expect("create multi-block dense tensor");

    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.expect("write");
    }

    let expd = Tensor::copy_from(
        exp_dir,
        &tensor.view().exp().await.expect("exp should succeed"),
        1000,
    )
    .await
    .expect("copy exp");
    let lnd = Tensor::copy_from(
        ln_dir,
        &tensor.view().ln().await.expect("ln should succeed"),
        1000,
    )
    .await
    .expect("copy ln");

    for coord in iter_coords(&[10]) {
        let i = coord[0] as usize;
        let expected_exp = values[i].exp();
        let expected_ln = values[i].ln();
        assert_eq!(
            expd.read_value(&coord).await.expect("read exp"),
            expected_exp
        );
        let actual_ln = lnd.read_value(&coord).await.expect("read ln");
        if expected_ln.is_nan() {
            assert!(actual_ln.is_nan());
        } else {
            assert_eq!(actual_ln, expected_ln);
        }
    }
}

#[tokio::test]
async fn chain_is_lazy_until_copy() {
    let (_root, dir) = new_dir("lazy_chain_source").await;
    let (out_root, out_dir) = new_dir("lazy_chain_out").await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, schema).await;
    tensor.write_value(&[0, 0], 1.0).await.expect("write");

    let chain = tensor
        .view()
        .exp()
        .await
        .expect("exp should succeed")
        .ln()
        .await
        .expect("ln should succeed");

    assert!(
        !out_root.join("blocks").exists(),
        "copy must not have run any I/O yet"
    );

    let copied = Tensor::copy_from(out_dir.clone(), &chain, 1000)
        .await
        .expect("copy should succeed");
    copied.sync().await.expect("sync");

    assert!(
        out_root.join("blocks").exists(),
        "copy must create the blocks directory"
    );
    assert_eq!(
        copied.read_value(&[0, 0]).await.expect("read"),
        1.0f32.exp().ln()
    );
}

#[tokio::test]
async fn ln_domain_edges_produce_ieee754_values_not_errors() {
    let (_root, dir) = new_dir("ln_domain_edges").await;
    let (_out_root, out_dir) = new_dir("ln_domain_edges_out").await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape![2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, schema).await;

    tensor.write_value(&[0], 0.0).await.expect("write zero");
    tensor
        .write_value(&[1], -1.0)
        .await
        .expect("write negative");

    let lnd = Tensor::copy_from(
        out_dir,
        &tensor
            .view()
            .ln()
            .await
            .expect("ln should succeed, not error"),
        1000,
    )
    .await
    .expect("copy should succeed, not error");

    assert_eq!(lnd.read_value(&[0]).await.expect("read"), f32::NEG_INFINITY);
    assert!(lnd.read_value(&[1]).await.expect("read").is_nan());
}

#[tokio::test]
async fn sparse_round_exp_ln_all_supported_on_populated_elements_only() {
    let (_root, dir) = new_dir("sparse_unary_supported").await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3, 4]).expect("schema");
    let tensor = create_sparse_tensor::<f32>(dir, schema, Some(1)).await;

    tensor
        .write_value(&[0, 1, 2], 1.6)
        .await
        .expect("write nz 1");
    tensor
        .write_value(&[1, 2, 3], -1.5)
        .await
        .expect("write nz 2");

    for (op_name, expected_a, expected_b) in [
        ("round", 1.6f32.round(), (-1.5f32).round()),
        ("exp", 1.6f32.exp(), (-1.5f32).exp()),
        ("ln", 1.6f32.ln(), f32::NAN),
    ] {
        let (_out_root, out_dir) = new_dir(&format!("sparse_unary_supported_{op_name}_out")).await;
        let view = tensor.view();
        let result = match op_name {
            "round" => Tensor::copy_from(out_dir, &view.round().await.expect("round"), 1000).await,
            "exp" => Tensor::copy_from(out_dir, &view.exp().await.expect("exp"), 1000).await,
            "ln" => Tensor::copy_from(out_dir, &view.ln().await.expect("ln"), 1000).await,
            _ => unreachable!(),
        }
        .unwrap_or_else(|err| panic!("copy {op_name} should succeed: {err}"));

        let actual_a = result.read_value(&[0, 1, 2]).await.expect("read a");
        if expected_a.is_nan() {
            assert!(actual_a.is_nan(), "{op_name} at populated coord a");
        } else {
            assert_eq!(actual_a, expected_a, "{op_name} at populated coord a");
        }

        let actual_b = result.read_value(&[1, 2, 3]).await.expect("read b");
        if expected_b.is_nan() {
            assert!(actual_b.is_nan(), "{op_name} at populated coord b");
        } else {
            assert_eq!(actual_b, expected_b, "{op_name} at populated coord b");
        }

        assert_eq!(
            result
                .read_value(&[0, 0, 0])
                .await
                .expect("read unpopulated"),
            0.0,
            "{op_name} must leave unpopulated coordinates as the implicit zero"
        );
    }
}

#[tokio::test]
async fn transformed_sparse_unary_view_copies() {
    let (_root, dir) = new_dir("sparse_non_identity_unsupported").await;
    let (_out_root, out_dir) = new_dir("sparse_non_identity_unsupported_out").await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3, 4]).expect("schema");
    let tensor = create_sparse_tensor::<f32>(dir, schema, Some(1)).await;

    tensor.write_value(&[0, 1, 2], 1.6).await.expect("write");

    let sliced = tensor
        .view()
        .slice(
            [
                AxisRange::In(0, 1, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1),
            ]
            .into_iter()
            .collect(),
        )
        .expect("slice should succeed");

    let chain = sliced.exp().await.expect("exp should succeed lazily");
    let result = Tensor::copy_from(out_dir, &chain, 2)
        .await
        .expect("copy transformed sparse");
    assert_eq!(
        result.read_value(&[0, 1, 2]).await.expect("populated"),
        1.6f32.exp()
    );
    assert_eq!(
        result.read_value(&[0, 0, 0]).await.expect("implicit zero"),
        0.0
    );
}

#[tokio::test]
async fn chained_exp_then_round_matches_per_coordinate_computation_multi_block() {
    // shape [10] with max_capacity 3 -> 4 blocks, exercising the block-level
    // copy path under a real multi-op chain, not just a single op.
    let (_root, dir) = new_dir("chained_exp_round_multi_block").await;
    let (_out_root, out_dir) = new_dir("chained_exp_round_multi_block_out").await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape![10]).expect("schema");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema, Layout::Dense, 3)
        .await
        .expect("create multi-block dense tensor");

    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.3 - 1.0).collect();
    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.expect("write");
    }

    let chained = Tensor::copy_from(
        out_dir,
        &tensor
            .view()
            .exp()
            .await
            .expect("exp should succeed")
            .round()
            .await
            .expect("round should succeed"),
        1000,
    )
    .await
    .expect("copy should succeed");

    for coord in iter_coords(&[10]) {
        let i = coord[0] as usize;
        let expected = values[i].exp().round();
        let actual = chained.read_value(&coord).await.expect("read");
        assert_eq!(actual, expected, "coord {coord:?}");
    }
}

#[tokio::test]
async fn computed_f64_view_agrees_across_consumers_and_reuse() {
    let (root, dir) = new_dir("unary_consumers").await;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F64), shape![2, 5]).unwrap(),
        Layout::Dense,
        3,
    )
    .await
    .unwrap();
    for coord in iter_coords(&[2, 5]) {
        tensor
            .write_value(&coord, (coord[0] * 5 + coord[1]) as f64 / 4.0)
            .await
            .unwrap();
    }
    let expression: UnaryView<UnaryView<TensorView<'_, FsEntry, f64>, Round>, Exp> = tensor
        .view()
        .round()
        .await
        .unwrap()
        .transpose(Some(axes![1, 0]))
        .unwrap()
        .exp()
        .await
        .unwrap();
    assert!(!expression.is_base_tensor());
    assert!(!expression.supports_write_through());
    let (first, second): (Vec<Vec<f64>>, Vec<Vec<f64>>) = futures::try_join!(
        expression.read_blocks().unwrap().try_collect(),
        expression.read_blocks().unwrap().try_collect(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.iter().map(Vec::len).collect::<Vec<_>>(), vec![10]);
    let values: Vec<_> = first.into_iter().flatten().collect();
    let (out_root, out_dir) = new_dir("unary_consumers_out").await;
    let output = Tensor::copy_from(out_dir, &expression, 2).await.unwrap();
    for (i, coord) in iter_coords(&[5, 2]).enumerate() {
        let expected = (((coord[1] * 5 + coord[0]) as f64 / 4.0).round()).exp();
        assert_eq!(values[i], expected);
        assert_eq!(expression.read_value(&coord).await.unwrap(), expected);
        assert_eq!(output.read_value(&coord).await.unwrap(), expected);
    }
    assert_eq!(tensor.read_value(&[0, 0]).await.unwrap(), 0.0);
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}

#[tokio::test]
async fn sparse_chain_preserves_input_support_through_intermediate_zero() {
    let (root, dir) = new_dir("sparse_chain_support").await;
    let tensor = create_sparse_tensor::<f32>(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3]).unwrap(),
        Some(1),
    )
    .await;
    tensor.write_value(&[1, 2], -0.2).await.unwrap();
    let expression: UnaryView<UnaryView<TensorView<'_, FsEntry, f32>, Round>, Exp> = tensor
        .view()
        .round()
        .await
        .unwrap()
        .exp()
        .await
        .unwrap()
        .transpose(Some(axes![1, 0]))
        .unwrap();
    assert_eq!(expression.read_value(&[2, 1]).await.unwrap(), 1.0);
    assert_eq!(expression.read_value(&[0, 0]).await.unwrap(), 0.0);
    let rows: Vec<_> = expression
        .read_sparse_elements_in_order(
            range![AxisRange::In(0, 3, 1), AxisRange::In(0, 2, 1)],
            axes![0, 1],
        )
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows, vec![(vec![2, 1], 1.0)]);
    let (out_root, out_dir) = new_dir("sparse_chain_support_out").await;
    let output = Tensor::copy_from(out_dir, &expression, 2).await.unwrap();
    for coord in iter_coords(&[3, 2]) {
        assert_eq!(
            output.read_value(&coord).await.unwrap(),
            expression.read_value(&coord).await.unwrap()
        );
    }
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}

#[tokio::test]
async fn stream_is_demand_driven_and_errors_on_corrupt_tail() {
    let (root, dir) = new_dir("unary_corrupt_tail").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![8192]).unwrap(),
        Layout::Dense,
        4096,
    )
    .await
    .unwrap();
    let expression = tensor.view().exp().await.unwrap();
    let mut stream = expression.read_blocks().unwrap();
    // Constructing the stream must not read or retain the data it will consume.
    tensor.write_value(&[0], 1.0).await.unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete("1").await;
    let first = stream.try_next().await.unwrap().unwrap();
    assert_eq!(first.len(), 4096);
    assert_eq!(first[0], 1.0f32.exp());
    assert!(first[1..].iter().all(|&value| value == 1.0));
    drop(stream);
    // A fresh consumer starts from the beginning, and a later read failure is surfaced.
    let mut stream = expression.read_blocks().unwrap();
    assert!(stream.try_next().await.unwrap().is_some());
    assert!(stream.try_next().await.is_err());
    assert!(expression.read_value(&[8191]).await.is_err());
    common::cleanup(&root).await;
}

#[tokio::test]
async fn copying_spills_beyond_cache_budget_and_reloads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("unary_small_cache").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root.clone()).unwrap();
        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![1024]).unwrap(),
            Layout::Dense,
            16,
        )
        .await
        .unwrap();
        // A zero-filled tensor larger than cache capacity must spill during creation.
        let mut files = tokio::fs::read_dir(root.join("blocks")).await.unwrap();
        let mut count = 0;
        while files.next_entry().await.unwrap().is_some() {
            count += 1;
        }
        assert!(count > 1, "blocks must spill before explicit sync");
        tensor.write_value(&[1023], 2.0).await.unwrap();
        let (out_root, _) = new_dir("unary_small_cache_out").await;
        let out_cache =
            freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let out_dir = out_cache.load(out_root.clone()).unwrap();
        let expression = tensor.view().exp().await.unwrap();
        let output = Tensor::copy_from(out_dir.clone(), &expression, 16)
            .await
            .unwrap();

        assert_eq!(output.read_value(&[1023]).await.unwrap(), 2.0f32.exp());
        output.sync().await.unwrap();
        drop(output);
        drop(out_dir);
        let reloaded = Tensor::<FsEntry, f32>::load(common::open_dir(&out_root).unwrap())
            .await
            .unwrap();
        assert_eq!(reloaded.read_value(&[0]).await.unwrap(), 1.0);
        assert_eq!(reloaded.read_value(&[1023]).await.unwrap(), 2.0f32.exp());
        common::cleanup(&root).await;
        common::cleanup(&out_root).await;
    })
    .await
    .expect("cache pressure must make progress");
}

#[tokio::test]
async fn block_larger_than_cache_is_a_recoverable_error() {
    let (root, _) = new_dir("unary_oversized_block").await;
    let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
    let dir = cache.load(root.clone()).unwrap();
    let result = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![1024]).unwrap(),
        Layout::Dense,
        1024,
    )
    .await;
    assert!(
        matches!(result, Err(Error::Io(ref error)) if error.kind() == std::io::ErrorKind::OutOfMemory)
    );
    common::cleanup(&root).await;
}

#[tokio::test]
async fn large_sparse_shape_constructs_streams_without_expanding_axes() {
    let (root, dir) = new_dir("unary_large_sparse_shape").await;
    let length = 1_000_000_000;
    let tensor = create_sparse_tensor::<f32>(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![length]).unwrap(),
        None,
    )
    .await;
    let expression = tensor.view().exp().await.unwrap();
    let sparse = expression
        .read_sparse_elements_in_order(range![AxisRange::In(0, length, 1)], axes![0])
        .await
        .unwrap();
    drop(sparse);
    let mut blocks = expression.read_blocks().unwrap();
    assert_eq!(blocks.try_next().await.unwrap().unwrap(), vec![0.0; 4096]);
    drop(blocks);
    common::cleanup(&root).await;
}

#[tokio::test]
async fn scalar_view_streaming_is_explicitly_unsupported() {
    let (root, dir) = new_dir("unary_scalar_view").await;
    let tensor = create_dense_tensor::<f32>(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2]).unwrap(),
    )
    .await;
    let scalar = tensor
        .view()
        .exp()
        .await
        .unwrap()
        .slice(range![AxisRange::At(1)])
        .unwrap();
    assert!(matches!(scalar.read_blocks(), Err(Error::InvalidSchema(_))));
    common::cleanup(&root).await;
}

#[tokio::test]
async fn typed_unary_composition_preserves_all_geometric_transforms() {
    for (name, layout) in [
        ("typed_transforms_dense", Layout::Dense),
        ("typed_transforms_sparse", Layout::Sparse { axis: Some(1) }),
    ] {
        let (root, dir) = new_dir(name).await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir,
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 1, 3]).unwrap(),
            layout,
            3,
        )
        .await
        .unwrap();
        for coord in iter_coords(&[2, 1, 3]) {
            tensor
                .write_value(&coord, (coord[0] * 3 + coord[2]) as f32 + 0.25)
                .await
                .unwrap();
        }
        let expression = tensor
            .view()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 1, 1),
                AxisRange::In(0, 3, 1)
            ])
            .unwrap()
            .round()
            .await
            .unwrap()
            .reshape(shape![2, 3])
            .unwrap()
            .unsqueeze(axes![1])
            .unwrap()
            .broadcast(shape![2, 4, 3])
            .unwrap()
            .flip(2)
            .unwrap()
            .slice(range![
                AxisRange::At(1),
                AxisRange::In(1, 4, 2),
                AxisRange::In(0, 3, 1)
            ])
            .unwrap()
            .exp()
            .await
            .unwrap()
            .transpose(Some(axes![1, 0]))
            .unwrap()
            .unsqueeze(axes![1])
            .unwrap()
            .squeeze(axes![1])
            .unwrap()
            .ln()
            .await
            .unwrap();
        let blocks: Vec<Vec<f32>> = expression
            .read_blocks()
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let values: Vec<_> = blocks.into_iter().flatten().collect();
        assert_eq!(values.len(), 6);
        let (out_root, out_dir) = new_dir(&format!("{name}_out")).await;
        let output = Tensor::copy_from(out_dir, &expression, 2).await.unwrap();
        if matches!(layout, Layout::Sparse { .. }) {
            let rows: Vec<_> = expression
                .read_sparse_elements_in_order(
                    range![AxisRange::In(0, 3, 1), AxisRange::In(0, 2, 1)],
                    axes![0, 1],
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(
                rows,
                iter_coords(&[3, 2])
                    .zip(values.iter().copied())
                    .collect::<Vec<_>>()
            );
        }
        for (i, coord) in iter_coords(&[3, 2]).enumerate() {
            // Row 1 of the original tensor, reversed along its final axis;
            // the two output columns both select the broadcast singleton.
            let expected = (5.0 - coord[0] as f32).exp().ln();
            assert_eq!(values[i], expected, "{name}: {coord:?}");
            assert_eq!(expression.read_value(&coord).await.unwrap(), expected);
            assert_eq!(output.read_value(&coord).await.unwrap(), expected);
        }
        common::cleanup(&root).await;
        common::cleanup(&out_root).await;
    }
}

#[tokio::test]
async fn sparse_ranges_are_ordered_unique_and_bounded_by_selection() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, dir) = new_dir("sparse_selected_tail").await;
        let length = 1_000_000_000;
        let tensor = create_sparse_tensor::<f32>(
            dir,
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![length]).unwrap(),
            None,
        )
        .await;
        for offset in [1, 3, 5] {
            tensor.write_value(&[(length - offset)], 0.2).await.unwrap();
        }
        let view = tensor.view();
        let unary = view.round().await.unwrap().exp().await.unwrap();
        let selection = range![AxisRange::Of(vec![
            length - 1,
            length - 5,
            length - 1,
            length - 3
        ])];
        let expected_coords = vec![vec![(length - 5)], vec![(length - 3)], vec![(length - 1)]];
        async fn check(
            reader: &impl TensorRead<DType = f32>,
            length: u64,
            selection: fensor::Range,
            expected_coords: &[Vec<u64>],
        ) {
            let rows: Vec<_> = reader
                .read_sparse_elements_in_order(selection.clone(), axes![0])
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(
                rows.iter()
                    .map(|(coord, _)| coord.clone())
                    .collect::<Vec<_>>(),
                expected_coords
            );
            let stepped: Vec<_> = reader
                .read_sparse_elements_in_order(
                    range![AxisRange::In(length - 5, length, 2)],
                    axes![0],
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(rows, stepped);
            let empty: Vec<_> = reader
                .read_sparse_elements_in_order(range![AxisRange::In(length, length, 1)], axes![0])
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert!(empty.is_empty());
            assert!(
                reader
                    .read_sparse_elements_in_order(range![AxisRange::At(length)], axes![0])
                    .await
                    .is_err()
            );
            assert!(matches!(
                reader
                    .read_sparse_elements_in_order(selection.clone(), axes![1])
                    .await,
                Err(Error::UnsupportedSparseIterationOrder { .. })
            ));
        }
        check(&tensor, length, selection.clone(), &expected_coords).await;
        check(&view, length, selection.clone(), &expected_coords).await;
        check(&unary, length, selection.clone(), &expected_coords).await;
        let rows: Vec<_> = unary
            .read_sparse_elements_in_order(selection, axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert!(rows.iter().all(|(_, value)| *value == 1.0));
        common::cleanup(&root).await;
    })
    .await
    .expect("small sparse selections must not scan the full shape");
}

#[tokio::test]
async fn sparse_range_reads_only_selected_storage() {
    use fensor::TensorSparseIndex;

    let (root, dir) = new_dir("sparse_range_corruption").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![8]).unwrap(),
        Layout::Sparse { axis: None },
        2,
    )
    .await
    .unwrap();
    tensor.write_value(&[0], 0.2).await.unwrap();
    tensor.write_value(&[7], 0.2).await.unwrap();
    let block_id = tensor.lookup_block_id(&[7, 3]).await.unwrap().unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete(&block_id.to_string()).await;
    let view = tensor.view();
    let unary = view.round().await.unwrap().exp().await.unwrap();
    async fn check(reader: &impl TensorRead<DType = f32>) {
        let rows: Vec<_> = reader
            .read_sparse_elements_in_order(range![AxisRange::At(0)], axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        let mut corrupt = reader
            .read_sparse_elements_in_order(range![AxisRange::At(7)], axes![0])
            .await
            .unwrap();
        assert!(corrupt.try_next().await.is_err());
    }
    check(&tensor).await;
    check(&view).await;
    check(&unary).await;
    common::cleanup(&root).await;
}

#[tokio::test]
async fn sparse_unary_batches_preserve_support_and_independent_consumption() {
    let (root, dir) = new_dir("sparse_unary_batch_boundary").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![1, 4100]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        4096,
    )
    .await
    .unwrap();
    for i in 0..4100 {
        tensor
            .write_value(&[0, i], if i % 2 == 0 { 0.2 } else { 1.2 })
            .await
            .unwrap();
    }
    let expression = tensor.view().round().await.unwrap().exp().await.unwrap();
    let range = range![AxisRange::At(0), AxisRange::In(0, 4100, 1)];
    let mut dropped = expression
        .read_sparse_elements_in_order(range.clone(), axes![0, 1])
        .await
        .unwrap();
    assert_eq!(dropped.try_next().await.unwrap(), Some((vec![0, 0], 1.0)));
    drop(dropped);
    let first = expression
        .read_sparse_elements_in_order(range.clone(), axes![0, 1])
        .await
        .unwrap();
    let second = expression
        .read_sparse_elements_in_order(range.clone(), axes![0, 1])
        .await
        .unwrap();
    let (first, second) = futures::try_join!(
        first.try_collect::<Vec<_>>(),
        second.try_collect::<Vec<_>>()
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 4100);
    for (coord, value) in &first {
        assert_eq!(*value, expression.read_value(coord).await.unwrap());
    }
    let final_zeros = expression.ln().await.unwrap();
    let rows: Vec<_> = final_zeros
        .read_sparse_elements_in_order(range, axes![0, 1])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows.len(), 2050);
    assert!(
        rows.iter()
            .all(|(coord, value)| coord[1] % 2 == 1 && *value != 0.0)
    );
    common::cleanup(&root).await;
}

#[tokio::test]
async fn dense_scalar_write_rejects_malformed_existing_block() {
    let (root, dir) = new_dir("dense_write_malformed_block").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![4]).unwrap(),
        Layout::Dense,
        4,
    )
    .await
    .unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let file = blocks.read().await.get_file("0").unwrap().clone();
    file.write::<Vec<f32>>().await.unwrap().truncate(1);
    assert!(matches!(
        tensor.write_value(&[0], 2.0).await,
        Err(Error::InvalidLayout(_))
    ));
    assert_eq!(*file.read::<Vec<f32>>().await.unwrap(), vec![0.0]);
    common::cleanup(&root).await;
}
