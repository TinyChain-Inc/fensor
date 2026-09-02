//! Integration tests for `TensorUnary` (`exp`, `ln`, `round`) via lazy
//! `TensorView` chains, materialized with an explicit terminal call.

use fensor::{
    DType, Error, Layout, Tensor, TensorRead, TensorSchema, TensorTransform, TensorUnary,
    TensorWrite,
};
use ha_ndarray::{AxisRange, shape};

use common::{FsEntry, create_dense_tensor, create_sparse_tensor, iter_coords, new_dir};

mod common;

#[tokio::test]
async fn dense_single_block_round_matches_expected_values() {
    let (_root, dir) = new_dir("dense_round_single_block").await;
    let (_out_root, out_dir) = new_dir("dense_round_single_block_out").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, schema).await;

    let values: [((u64, u64), f32); 4] =
        [((0, 0), 1.4), ((0, 1), 1.6), ((1, 0), -1.5), ((1, 1), 2.5)];
    for ((r, c), value) in values {
        tensor.write_value(&[r, c], value).await.expect("write");
    }

    let rounded = tensor
        .view()
        .round()
        .await
        .expect("round should succeed")
        .materialize(out_dir, 1000)
        .await
        .expect("materialize should succeed");

    for ((r, c), value) in values {
        let expected = value.round();
        let actual = rounded.read_value(&[r, c]).await.expect("read");
        assert_eq!(actual, expected, "coord ({r},{c})");
    }
}

#[tokio::test]
async fn dense_multi_block_exp_and_ln_stream_correctly() {
    // shape [10] with max_capacity 3 -> block_shape [3], grid ceil(10/3) = 4
    // blocks, forcing the block-level materialize path to walk multiple blocks.
    let (_root, dir) = new_dir("dense_multi_block_exp_ln").await;
    let (_exp_root, exp_dir) = new_dir("dense_multi_block_exp_ln_exp_out").await;
    let (_ln_root, ln_dir) = new_dir("dense_multi_block_exp_ln_ln_out").await;
    let schema = TensorSchema::new(DType::F32, shape![10]).expect("schema");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema, Layout::Dense, 3)
        .await
        .expect("create multi-block dense tensor");

    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.expect("write");
    }

    let expd = tensor
        .view()
        .exp()
        .await
        .expect("exp should succeed")
        .materialize(exp_dir, 1000)
        .await
        .expect("materialize exp");
    let lnd = tensor
        .view()
        .ln()
        .await
        .expect("ln should succeed")
        .materialize(ln_dir, 1000)
        .await
        .expect("materialize ln");

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
async fn chain_is_lazy_until_materialize() {
    let (_root, dir) = new_dir("lazy_chain_source").await;
    let (out_root, out_dir) = new_dir("lazy_chain_out").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 2]).expect("schema");
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
        "materialize must not have run any I/O yet"
    );

    let materialized = chain
        .materialize(out_dir.clone(), 1000)
        .await
        .expect("materialize should succeed");
    out_dir.sync().await.expect("sync");

    assert!(
        out_root.join("blocks").exists(),
        "materialize must create the blocks directory"
    );
    assert_eq!(
        materialized.read_value(&[0, 0]).await.expect("read"),
        1.0f32.exp().ln()
    );
}

#[tokio::test]
async fn ln_domain_edges_produce_ieee754_values_not_errors() {
    let (_root, dir) = new_dir("ln_domain_edges").await;
    let (_out_root, out_dir) = new_dir("ln_domain_edges_out").await;
    let schema = TensorSchema::new(DType::F32, shape![2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, schema).await;

    tensor.write_value(&[0], 0.0).await.expect("write zero");
    tensor
        .write_value(&[1], -1.0)
        .await
        .expect("write negative");

    let lnd = tensor
        .view()
        .ln()
        .await
        .expect("ln should succeed, not error")
        .materialize(out_dir, 1000)
        .await
        .expect("materialize should succeed, not error");

    assert_eq!(lnd.read_value(&[0]).await.expect("read"), f32::NEG_INFINITY);
    assert!(lnd.read_value(&[1]).await.expect("read").is_nan());
}

#[tokio::test]
async fn sparse_round_exp_ln_all_supported_on_populated_elements_only() {
    let (_root, dir) = new_dir("sparse_unary_supported").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 3, 4]).expect("schema");
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
        let chain = match op_name {
            "round" => view.round().await,
            "exp" => view.exp().await,
            "ln" => view.ln().await,
            _ => unreachable!(),
        }
        .unwrap_or_else(|err| panic!("{op_name} should be supported for Sparse: {err}"));
        let result = chain
            .materialize(out_dir, 1000)
            .await
            .unwrap_or_else(|err| panic!("materialize {op_name} should succeed: {err}"));

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
async fn non_identity_sparse_view_with_pending_op_is_unsupported() {
    let (_root, dir) = new_dir("sparse_non_identity_unsupported").await;
    let (_out_root, out_dir) = new_dir("sparse_non_identity_unsupported_out").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 3, 4]).expect("schema");
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
    match chain.materialize(out_dir, 1000).await {
        Err(Error::Unsupported(_)) => {}
        Err(_other) => panic!("expected Error::Unsupported, got a different error variant"),
        Ok(_) => panic!("materialize of a non-identity Sparse view with pending ops must fail"),
    }
}

#[tokio::test]
async fn chained_exp_then_round_matches_per_coordinate_computation_multi_block() {
    // shape [10] with max_capacity 3 -> 4 blocks, exercising the block-level
    // materialize path under a real multi-op chain, not just a single op.
    let (_root, dir) = new_dir("chained_exp_round_multi_block").await;
    let (_out_root, out_dir) = new_dir("chained_exp_round_multi_block_out").await;
    let schema = TensorSchema::new(DType::F32, shape![10]).expect("schema");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema, Layout::Dense, 3)
        .await
        .expect("create multi-block dense tensor");

    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.3 - 1.0).collect();
    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.expect("write");
    }

    let chained = tensor
        .view()
        .exp()
        .await
        .expect("exp should succeed")
        .round()
        .await
        .expect("round should succeed")
        .materialize(out_dir, 1000)
        .await
        .expect("materialize should succeed");

    for coord in iter_coords(&[10]) {
        let i = coord[0] as usize;
        let expected = values[i].exp().round();
        let actual = chained.read_value(&coord).await.expect("read");
        assert_eq!(actual, expected, "coord {coord:?}");
    }
}
