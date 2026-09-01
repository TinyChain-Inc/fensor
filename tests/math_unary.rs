//! Integration tests for `TensorUnary` (`exp`, `ln`, `round`).

use fensor::{DType, Error, Layout, Tensor, TensorRead, TensorSchema, TensorUnary, TensorWrite};
use ha_ndarray::shape;

use common::{FsEntry, create_dense_tensor, create_sparse_tensor, iter_coords, new_dir};

mod common;

#[tokio::test]
async fn dense_single_block_round_matches_expected_values() {
    let (_root, dir) = new_dir("dense_round_single_block").await;
    let (_workspace_root, workspace) = new_dir("dense_round_single_block_workspace").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, workspace, schema).await;

    let values: [((u64, u64), f32); 4] =
        [((0, 0), 1.4), ((0, 1), 1.6), ((1, 0), -1.5), ((1, 1), 2.5)];
    for ((r, c), value) in values {
        tensor.write_value(&[r, c], value).await.expect("write");
    }

    let rounded = tensor.round().await.expect("round should succeed");

    for ((r, c), value) in values {
        let expected = value.round();
        let actual = rounded.read_value(&[r, c]).await.expect("read");
        assert_eq!(actual, expected, "coord ({r},{c})");
    }
}

#[tokio::test]
async fn dense_multi_block_exp_and_ln_stream_correctly() {
    // shape [10] with max_capacity 3 -> block_shape [3], grid ceil(10/3) = 4
    // blocks, forcing `stream_unary_dense` to walk multiple blocks.
    let (_root, dir) = new_dir("dense_multi_block_exp_ln").await;
    let (_workspace_root, workspace) = new_dir("dense_multi_block_exp_ln_workspace").await;
    let schema = TensorSchema::new(DType::F32, shape![10]).expect("schema");
    let tensor = Tensor::<FsEntry, f32>::create(dir, workspace, schema, Layout::Dense, 3)
        .await
        .expect("create multi-block dense tensor");

    let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    for (i, &value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.expect("write");
    }

    let expd = tensor.exp().await.expect("exp should succeed");
    let lnd = tensor.ln().await.expect("ln should succeed");

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
async fn ln_domain_edges_produce_ieee754_values_not_errors() {
    // Documents/locks in the decision that unary ops validate shape/layout
    // support only, not numeric domain -- domain edges propagate as ordinary
    // IEEE754 float values (matching numpy-style semantics), not errors.
    let (_root, dir) = new_dir("ln_domain_edges").await;
    let (_workspace_root, workspace) = new_dir("ln_domain_edges_workspace").await;
    let schema = TensorSchema::new(DType::F32, shape![2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, workspace, schema).await;

    tensor.write_value(&[0], 0.0).await.expect("write zero");
    tensor
        .write_value(&[1], -1.0)
        .await
        .expect("write negative");

    let lnd = tensor.ln().await.expect("ln should succeed, not error");

    assert_eq!(lnd.read_value(&[0]).await.expect("read"), f32::NEG_INFINITY);
    assert!(lnd.read_value(&[1]).await.expect("read").is_nan());
}

#[tokio::test]
async fn sparse_round_supported_exp_ln_structured_unsupported() {
    let (_root, dir) = new_dir("sparse_round_supported").await;
    let (_workspace_root, workspace) = new_dir("sparse_round_supported_workspace").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 3, 4]).expect("schema");
    let tensor = create_sparse_tensor::<f32>(dir, workspace, schema, Some(1)).await;

    tensor
        .write_value(&[0, 1, 2], 1.6)
        .await
        .expect("write nz 1");
    tensor
        .write_value(&[1, 2, 3], -1.5)
        .await
        .expect("write nz 2");

    let rounded = tensor
        .round()
        .await
        .expect("round should be supported for Sparse");

    assert_eq!(rounded.read_value(&[0, 1, 2]).await.expect("read"), 2.0);
    assert_eq!(rounded.read_value(&[1, 2, 3]).await.expect("read"), -2.0);
    // Unpopulated coordinates must still read as the implicit zero.
    assert_eq!(rounded.read_value(&[0, 0, 0]).await.expect("read"), 0.0);

    for (op_name, result) in [("exp", tensor.exp().await), ("ln", tensor.ln().await)] {
        match result {
            Ok(_) => panic!("expected {op_name} on Sparse to be Unsupported"),
            Err(Error::Unsupported(message)) => {
                assert!(
                    message.contains(op_name),
                    "error message {message:?} should name the op {op_name:?}"
                );
            }
            Err(other) => panic!("unexpected error variant for {op_name}: {other}"),
        }
    }
}

#[tokio::test]
async fn chained_exp_then_round_matches_per_coordinate_computation() {
    let (_root, dir) = new_dir("chained_exp_round").await;
    let (_workspace_root, workspace) = new_dir("chained_exp_round_workspace").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 2]).expect("schema");
    let tensor = create_dense_tensor::<f32>(dir, workspace, schema).await;

    let values: [((u64, u64), f32); 4] =
        [((0, 0), 0.1), ((0, 1), 1.0), ((1, 0), -0.5), ((1, 1), 2.3)];
    for ((r, c), value) in values {
        tensor.write_value(&[r, c], value).await.expect("write");
    }

    let chained = tensor
        .exp()
        .await
        .expect("exp should succeed")
        .round()
        .await
        .expect("round should succeed");

    for ((r, c), value) in values {
        let expected = value.exp().round();
        let actual = chained.read_value(&[r, c]).await.expect("read");
        assert_eq!(actual, expected, "coord ({r},{c})");
    }
}
