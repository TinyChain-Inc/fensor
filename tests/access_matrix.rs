//! Test matrix for the `fensor` access layer per Requirements_1.md.
//!
//! Status: **many tests in this file are EXPECTED TO FAIL today.** They pin
//! down behavior that is not yet implemented (sparse zero-write lifecycle,
//! bulk I/O, view writeability, in-order sparse iteration, corruption
//! fail-closed). They serve as a living specification for the access-layer
//! rollout, not as gating CI checks.
//!
//! Tests that require API which does not yet exist (e.g. a `SparseZeroPolicy`
//! setter) are marked `#[ignore]` and document the needed surface in the body.

#![allow(unused_imports, dead_code, clippy::needless_range_loop)]

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use fensor::{
    BoxFuture, DType, Error, Layout, SparseZeroPolicy, Tensor, TensorArray, TensorBlockStore,
    TensorRead, TensorReadBulk, TensorSchema, TensorSparseIndex, TensorSparseLifecycle,
    TensorTransform, TensorViewSemantics, TensorWrite, TensorWriteBulk, contiguous_strides,
};
use ha_ndarray::{Axes, AxisRange, Range, Shape, axes, range, shape};

use common::{FsEntry, block_id_for_coord, cleanup, iter_coords, new_dir, open_dir};

// ---------- helpers ----------

fn dense_schema_f32(shape: Shape, block_shape: Shape) -> TensorSchema {
    let strides = contiguous_strides(&shape);
    TensorSchema::new(DType::F32, shape, Layout::Dense, block_shape, strides).expect("schema")
}

fn sparse_schema_f32(shape: Shape, block_shape: Shape, axis: Option<usize>) -> TensorSchema {
    let strides = contiguous_strides(&shape);
    TensorSchema::new(
        DType::F32,
        shape,
        Layout::Sparse { axis },
        block_shape,
        strides,
    )
    .expect("schema")
}

async fn create_dense(
    name: &str,
    shape: Shape,
    block_shape: Shape,
) -> (PathBuf, Tensor<FsEntry, f32>, TensorSchema) {
    let (root, dir) = new_dir(name).await;
    let schema = dense_schema_f32(shape, block_shape);
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema.clone())
        .await
        .expect("create dense");
    (root, tensor, schema)
}

async fn create_sparse(
    name: &str,
    shape: Shape,
    block_shape: Shape,
    axis: Option<usize>,
) -> (PathBuf, Tensor<FsEntry, f32>, TensorSchema) {
    let (root, dir) = new_dir(name).await;
    let schema = sparse_schema_f32(shape, block_shape, axis);
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema.clone())
        .await
        .expect("create sparse");
    (root, tensor, schema)
}

fn encode_value(coord: &[u64]) -> f32 {
    (coord[0] as f32) * 100.0 + (coord[1] as f32) * 10.0 + (coord[2] as f32)
}

async fn seed_values(tensor: &Tensor<FsEntry, f32>) {
    for coord in iter_coords(tensor.shape()) {
        tensor
            .write_value(&coord, encode_value(&coord))
            .await
            .expect("seed write");
    }
}

fn transpose_range(range: &Range, permutation: &[usize]) -> Range {
    let mut remapped = Vec::with_capacity(permutation.len());
    for axis in permutation {
        remapped.push(range[*axis].clone());
    }
    remapped.into()
}

// ====================================================================
// Section A — Access parity on real `Tensor<FE,T>`
// ====================================================================

mod section_a_access_parity {
    use crate::common::block_key_for_coord;

    use super::*;

    #[tokio::test]
    async fn dense_read_write_value_roundtrip() {
        let (root, tensor, _) = create_dense("a_dense_rw", shape![2, 3, 4], shape![1, 1, 2]).await;

        for coord in iter_coords(tensor.shape()) {
            tensor
                .write_value(&coord, encode_value(&coord))
                .await
                .expect("write");
        }

        for coord in iter_coords(tensor.shape()) {
            let actual = tensor.read_value(&coord).await.expect("read");
            assert_eq!(actual, encode_value(&coord), "coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_read_write_value_roundtrip() {
        let (root, tensor, schema) =
            create_sparse("a_sparse_rw", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        // Default-valued reads should return T::default() without materializing blocks.
        for coord in iter_coords(tensor.shape()) {
            let v = tensor.read_value(&coord).await.expect("read default");
            assert_eq!(v, 0.0, "expected default at {:?}", coord);
        }

        // A small sparse write set.
        let writes: HashMap<Vec<u64>, f32> = [
            (vec![0, 0, 0], 1.0),
            (vec![1, 2, 3], 9.0),
            (vec![1, 0, 1], 4.0),
            (vec![0, 2, 2], 7.0),
        ]
        .into_iter()
        .collect();

        let full_sparse_blocks: [Vec<u64>; 2] = [vec![0, 1, 0], vec![1, 1, 0]];

        for coords in full_sparse_blocks.iter() {
            assert!(
                block_id_for_coord(&tensor, &schema, coords).await.is_none(),
                "Block should not be materialized before writes"
            );
        }

        for (coord, value) in &writes {
            assert!(
                block_id_for_coord(&tensor, &schema, coord).await.is_none(),
                "Block should not be materialized before writes"
            );

            tensor.write_value(coord, *value).await.expect("write");

            assert!(
                block_id_for_coord(&tensor, &schema, coord).await.is_some(),
                "Block should be materialized for non-zero writes"
            );
        }

        for coords in full_sparse_blocks.iter() {
            assert!(
                block_id_for_coord(&tensor, &schema, coords).await.is_none(),
                "Block should not be materialized for full-sparse blocks"
            );
        }

        for coord in iter_coords(tensor.shape()) {
            let actual = tensor.read_value(&coord).await.expect("read");
            let expected = writes.get(&coord).copied().unwrap_or(0.0);
            assert_eq!(actual, expected, "coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn dense_sparse_parity_after_writes() {
        let (dense_root, dense, _) =
            create_dense("a_par_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        let (sparse_root, sparse, _) =
            create_sparse("a_par_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        let writes: HashMap<Vec<u64>, f32> = [
            (vec![0u64, 0, 0], 1.0),
            (vec![1, 2, 3], 9.0),
            (vec![1, 0, 1], 4.0),
            (vec![0, 2, 2], 7.0),
        ]
        .into_iter()
        .collect();

        for (coord, value) in &writes {
            dense.write_value(coord, *value).await.expect("dense write");
            sparse
                .write_value(coord, *value)
                .await
                .expect("sparse write");
        }

        for coord in iter_coords(dense.shape()) {
            let d = dense.read_value(&coord).await.expect("dense read");
            let s = sparse.read_value(&coord).await.expect("sparse read");
            assert_eq!(d, s, "parity at {:?}", coord);
        }

        cleanup(&dense_root).await;
        cleanup(&sparse_root).await;
    }

    #[tokio::test]
    async fn rank_one_tensor_read_write() {
        let (dense_root, dense, _) = create_dense("a_rank_one_d", shape![4], shape![2]).await;
        for i in 0..4u64 {
            dense
                .write_value(&[i], i as f32 + 1.0)
                .await
                .expect("dense write");
        }
        for i in 0..4u64 {
            let v = dense.read_value(&[i]).await.expect("dense read");
            assert_eq!(v, i as f32 + 1.0, "dense coord [{}]", i);
        }
        cleanup(&dense_root).await;

        let (sparse_none_root, sparse_none, _) =
            create_sparse("a_rank_one_sn", shape![4], shape![2], None).await;
        for i in 0..4u64 {
            sparse_none
                .write_value(&[i], i as f32 + 1.0)
                .await
                .expect("sparse_none write");
        }
        for i in 0..4u64 {
            let v = sparse_none
                .read_value(&[i])
                .await
                .expect("sparse_none read");
            assert_eq!(v, i as f32 + 1.0, "sparse_none coord [{}]", i);
        }
        cleanup(&sparse_none_root).await;

        let (sparse_root, sparse, _) =
            create_sparse("a_rank_one_s", shape![4], shape![2], Some(0)).await;
        for i in 0..4u64 {
            sparse
                .write_value(&[i], i as f32 + 1.0)
                .await
                .expect("sparse write");
        }
        for i in 0..4u64 {
            let v = sparse.read_value(&[i]).await.expect("sparse read");
            assert_eq!(v, i as f32 + 1.0, "sparse coord [{}]", i);
        }
        cleanup(&sparse_root).await;
    }

    #[tokio::test]
    async fn boundary_coordinates_dense_and_sparse() {
        let (dense_root, dense, _) =
            create_dense("a_boundary_dense", shape![2, 3, 4], shape![1, 1, 4]).await;

        dense
            .write_value(&[0, 0, 0], 1.0)
            .await
            .expect("dense write min");
        assert_eq!(
            dense.read_value(&[0, 0, 0]).await.expect("dense read min"),
            1.0
        );

        dense
            .write_value(&[1, 2, 3], 2.0)
            .await
            .expect("dense write max");
        assert_eq!(
            dense.read_value(&[1, 2, 3]).await.expect("dense read max"),
            2.0
        );

        let oob_read = dense
            .read_value(&[2, 0, 0])
            .await
            .expect_err("oob read dense");
        assert!(
            matches!(oob_read, Error::InvalidCoord(_)),
            "got {oob_read:?}"
        );
        let oob_write = dense
            .write_value(&[2, 0, 0], 1.0)
            .await
            .expect_err("oob write dense");
        assert!(
            matches!(oob_write, Error::InvalidCoord(_)),
            "got {oob_write:?}"
        );

        cleanup(&dense_root).await;

        let (sparse_root, sparse, _) = create_sparse(
            "a_boundary_sparse",
            shape![2, 3, 4],
            shape![1, 1, 4],
            Some(1),
        )
        .await;

        sparse
            .write_value(&[0, 0, 0], 1.0)
            .await
            .expect("sparse write min");
        assert_eq!(
            sparse
                .read_value(&[0, 0, 0])
                .await
                .expect("sparse read min"),
            1.0
        );

        sparse
            .write_value(&[1, 2, 3], 2.0)
            .await
            .expect("sparse write max");
        assert_eq!(
            sparse
                .read_value(&[1, 2, 3])
                .await
                .expect("sparse read max"),
            2.0
        );

        let oob_read = sparse
            .read_value(&[2, 0, 0])
            .await
            .expect_err("oob read sparse");
        assert!(
            matches!(oob_read, Error::InvalidCoord(_)),
            "got {oob_read:?}"
        );
        let oob_write = sparse
            .write_value(&[2, 0, 0], 1.0)
            .await
            .expect_err("oob write sparse");
        assert!(
            matches!(oob_write, Error::InvalidCoord(_)),
            "got {oob_write:?}"
        );

        cleanup(&sparse_root).await;
    }

    #[tokio::test]
    async fn sparse_axis_none_uses_axis_zero() {
        let (none_root, none_tensor, none_schema) =
            create_sparse("a_axis_none", shape![2, 3, 4], shape![1, 1, 4], None).await;

        none_tensor
            .write_value(&[1, 2, 3], 5.0)
            .await
            .expect("write");

        let key = block_key_for_coord(&none_schema, &[1, 2, 3]);
        assert_eq!(key[0], 1, "axis=None must use coord[0] as key");
        assert!(
            block_id_for_coord(&none_tensor, &none_schema, &[1, 2, 3])
                .await
                .is_some(),
            "block must exist after write"
        );

        let (some0_root, _, some0_schema) =
            create_sparse("a_axis_some0", shape![2, 3, 4], shape![1, 1, 4], Some(0)).await;

        assert_eq!(
            block_key_for_coord(&none_schema, &[1, 2, 3]),
            block_key_for_coord(&some0_schema, &[1, 2, 3]),
            "axis=None and axis=Some(0) must produce identical keys"
        );

        cleanup(&none_root).await;
        cleanup(&some0_root).await;
    }

    #[tokio::test]
    async fn sparse_axis_some_n_keys_by_n() {
        let (axis1_root, axis1_tensor, axis1_schema) =
            create_sparse("a_axis_some1", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        axis1_tensor
            .write_value(&[1, 2, 3], 5.0)
            .await
            .expect("axis1 write");

        let key1 = block_key_for_coord(&axis1_schema, &[1, 2, 3]);
        assert_eq!(key1[0], 2, "axis=Some(1) must use coord[1] as key");
        assert!(
            block_id_for_coord(&axis1_tensor, &axis1_schema, &[1, 2, 3])
                .await
                .is_some(),
            "block must exist after write (axis 1)"
        );

        cleanup(&axis1_root).await;

        let (axis2_root, axis2_tensor, axis2_schema) =
            create_sparse("a_axis_some2", shape![2, 3, 4], shape![1, 1, 4], Some(2)).await;

        axis2_tensor
            .write_value(&[1, 2, 3], 7.0)
            .await
            .expect("axis2 write");

        let key2 = block_key_for_coord(&axis2_schema, &[1, 2, 3]);
        assert_eq!(key2[0], 3, "axis=Some(2) must use coord[2] as key");
        assert!(
            block_id_for_coord(&axis2_tensor, &axis2_schema, &[1, 2, 3])
                .await
                .is_some(),
            "block must exist after write (axis 2)"
        );

        cleanup(&axis2_root).await;
    }
}

// ====================================================================
// Section B — Transform parity (chained transforms)
// ====================================================================

mod section_b_transforms {
    use super::*;

    #[tokio::test]
    async fn slice_then_read_dense() {
        let (root, tensor, _) =
            create_dense("b_slice_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let r: Range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(1, 3, 1),
            AxisRange::In(0, 4, 2)
        ];
        let sliced = tensor.clone().slice(r).expect("slice");
        assert_eq!(sliced.shape(), &[2, 2, 2]);

        for s_coord in iter_coords(sliced.shape()) {
            let src = vec![s_coord[0], s_coord[1] + 1, s_coord[2] * 2];
            let expected = tensor.read_value(&src).await.expect("read original");
            let actual = sliced.read_value(&s_coord).await.expect("read sliced");
            assert_eq!(actual, expected, "coord {:?}", s_coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn transpose_then_read_dense() {
        let (root, tensor, _) = create_dense("b_tx_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let perm = axes![2, 0, 1];
        let transposed = tensor.clone().transpose(Some(perm.clone())).expect("tx");
        assert_eq!(transposed.shape(), &[4, 2, 3]);

        let mut inverse = vec![0usize; perm.len()];
        for (i, axis) in perm.iter().enumerate() {
            inverse[*axis] = i;
        }

        for t_coord in iter_coords(transposed.shape()) {
            let mut src = vec![0u64; t_coord.len()];
            for old_axis in 0..t_coord.len() {
                src[old_axis] = t_coord[inverse[old_axis]];
            }
            let expected = tensor.read_value(&src).await.expect("read original");
            let actual = transposed.read_value(&t_coord).await.expect("read tx");
            assert_eq!(actual, expected, "coord {:?}", t_coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn reshape_then_read_dense() {
        let (root, tensor, _) = create_dense("b_reshape", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let reshaped = tensor.clone().reshape(shape![6, 4]).expect("reshape");
        assert_eq!(reshaped.shape(), &[6, 4]);

        let from_reshape = reshaped.read_value(&[0, 0]).await.expect("reshape read");
        let from_original = tensor.read_value(&[0, 0, 0]).await.expect("orig read");
        assert_eq!(from_reshape, from_original);

        // Row-major equivalence: index k in [6,4] maps to (k/4, k%4) and
        // to original (k/12, (k%12)/4, k%4) — same linear offset.
        for k in 0..24u64 {
            let r = k / 4;
            let c = k % 4;
            let a = k / 12;
            let b = (k % 12) / 4;
            let cc = k % 4;
            let from_reshape = reshaped.read_value(&[r, c]).await.expect("reshape read");
            let from_original = tensor.read_value(&[a, b, cc]).await.expect("orig read");
            assert_eq!(from_reshape, from_original, "k={k}");
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn transpose_then_read_sparse() {
        let (root, tensor, _) =
            create_sparse("b_tx_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;
        seed_values(&tensor).await;

        let perm = axes![1, 2, 0];
        let transposed = tensor.clone().transpose(Some(perm.clone())).expect("tx");
        assert_eq!(transposed.shape(), &[3, 4, 2]);

        let mut inverse = vec![0usize; perm.len()];
        for (i, axis) in perm.iter().enumerate() {
            inverse[*axis] = i;
        }

        for t_coord in iter_coords(transposed.shape()) {
            let mut src = vec![0u64; t_coord.len()];
            for old_axis in 0..t_coord.len() {
                src[old_axis] = t_coord[inverse[old_axis]];
            }
            let expected = tensor.read_value(&src).await.expect("orig");
            let actual = transposed.read_value(&t_coord).await.expect("tx");
            assert_eq!(actual, expected, "sparse tx coord {:?}", t_coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn slice_then_read_sparse() {
        let (root, tensor, _) =
            create_sparse("b_slice_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;
        seed_values(&tensor).await;

        let r: Range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 3, 2),
            AxisRange::In(1, 4, 1)
        ];
        let sliced = tensor.clone().slice(r).expect("slice");

        for s_coord in iter_coords(sliced.shape()) {
            let src = vec![s_coord[0], s_coord[1] * 2, s_coord[2] + 1];
            let expected = tensor.read_value(&src).await.expect("orig");
            let actual = sliced.read_value(&s_coord).await.expect("sliced");
            assert_eq!(actual, expected, "sparse slice coord {:?}", s_coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_transpose_slice_consistency_dense() {
        let (root, tensor, _) =
            create_dense("b_chain_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let perm = axes![2, 0, 1];
        let r: Range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(1, 3, 1),
            AxisRange::In(0, 4, 2)
        ];

        let left = tensor
            .clone()
            .slice(r.clone())
            .expect("slice")
            .transpose(Some(perm.clone()))
            .expect("tx");

        let remapped = transpose_range(&r, &perm);
        let right = tensor
            .clone()
            .transpose(Some(perm))
            .expect("tx")
            .slice(remapped)
            .expect("slice");

        assert_eq!(left.shape(), right.shape());
        for coord in iter_coords(left.shape()) {
            let l = left.read_value(&coord).await.expect("left");
            let r_v = right.read_value(&coord).await.expect("right");
            assert_eq!(l, r_v, "chain coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_transpose_slice_consistency_sparse() {
        let (root, tensor, _) =
            create_sparse("b_chain_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;
        seed_values(&tensor).await;

        let perm = axes![2, 0, 1];
        let r: Range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 3, 1),
            AxisRange::In(0, 4, 2)
        ];

        let left = tensor
            .clone()
            .slice(r.clone())
            .expect("slice")
            .transpose(Some(perm.clone()))
            .expect("tx");

        let remapped = transpose_range(&r, &perm);
        let right = tensor
            .clone()
            .transpose(Some(perm))
            .expect("tx")
            .slice(remapped)
            .expect("slice");

        assert_eq!(left.shape(), right.shape());
        for coord in iter_coords(left.shape()) {
            let l = left.read_value(&coord).await.expect("left");
            let r_v = right.read_value(&coord).await.expect("right");
            assert_eq!(l, r_v, "sparse chain coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_slice_transpose_slice_dense() {
        let (root, tensor, _) =
            create_dense("b_chain3_dense", shape![4, 4, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let s1: Range = range![
            AxisRange::In(0, 4, 1),
            AxisRange::In(0, 4, 2),
            AxisRange::In(0, 4, 1)
        ];
        let perm = axes![2, 0, 1];
        let s2: Range = range![
            AxisRange::In(0, 4, 2),
            AxisRange::In(0, 4, 1),
            AxisRange::In(0, 2, 1)
        ];

        let chained = tensor
            .clone()
            .slice(s1)
            .expect("s1")
            .transpose(Some(perm))
            .expect("tx")
            .slice(s2)
            .expect("s2");

        for coord in iter_coords(chained.shape()) {
            // No oracle here — just assert reads do not error and are
            // consistent with themselves on a repeat call.
            let v1 = chained.read_value(&coord).await.expect("chain read");
            let v2 = chained.read_value(&coord).await.expect("chain read");
            assert_eq!(v1, v2, "non-deterministic at {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn reshape_over_transformed_view_rejected() {
        let (root, tensor, _) =
            create_dense("b_reshape_view", shape![2, 3, 4], shape![1, 1, 4]).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");

        let err = sliced.reshape(shape![16]).err().expect("must reject");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn slice_axis_range_at() {
        let (root, tensor, _) = create_dense("b_at_dense", shape![3, 4, 5], shape![1, 1, 5]).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::At(1),
                AxisRange::In(0, 4, 1),
                AxisRange::In(0, 5, 1)
            ])
            .expect("slice At dense");
        assert_eq!(sliced.shape(), &[4, 5]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![1u64, coord[0], coord[1]];
            let expected = tensor.read_value(&src).await.expect("orig dense At");
            let actual = sliced.read_value(&coord).await.expect("sliced dense At");
            assert_eq!(actual, expected, "At dense coord {:?}", coord);
        }
        cleanup(&root).await;

        let (root, tensor, _) =
            create_sparse("b_at_sparse", shape![3, 4, 5], shape![1, 1, 5], Some(0)).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::At(1),
                AxisRange::In(0, 4, 1),
                AxisRange::In(0, 5, 1)
            ])
            .expect("slice At sparse");
        assert_eq!(sliced.shape(), &[4, 5]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![1u64, coord[0], coord[1]];
            let expected = tensor.read_value(&src).await.expect("orig sparse At");
            let actual = sliced.read_value(&coord).await.expect("sliced sparse At");
            assert_eq!(actual, expected, "At sparse coord {:?}", coord);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn slice_axis_range_in_with_step_gt_one() {
        let (root, tensor, _) =
            create_dense("b_step2_dense", shape![4, 6, 8], shape![1, 1, 8]).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 4, 2),
                AxisRange::In(0, 6, 2),
                AxisRange::In(0, 8, 2)
            ])
            .expect("slice step=2 dense");
        assert_eq!(sliced.shape(), &[2, 3, 4]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![coord[0] * 2, coord[1] * 2, coord[2] * 2];
            let expected = tensor.read_value(&src).await.expect("orig dense step2");
            let actual = sliced.read_value(&coord).await.expect("sliced dense step2");
            assert_eq!(actual, expected, "step2 dense coord {:?}", coord);
        }
        cleanup(&root).await;

        let (root, tensor, _) =
            create_sparse("b_step2_sparse", shape![4, 6, 8], shape![1, 1, 8], Some(1)).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 4, 2),
                AxisRange::In(0, 6, 2),
                AxisRange::In(0, 8, 2)
            ])
            .expect("slice step=2 sparse");
        assert_eq!(sliced.shape(), &[2, 3, 4]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![coord[0] * 2, coord[1] * 2, coord[2] * 2];
            let expected = tensor.read_value(&src).await.expect("orig sparse step2");
            let actual = sliced
                .read_value(&coord)
                .await
                .expect("sliced sparse step2");
            assert_eq!(actual, expected, "step2 sparse coord {:?}", coord);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn slice_axis_range_of_gather() {
        let src_indices: &[u64] = &[0, 2, 3];

        let (root, tensor, _) =
            create_dense("b_gather_dense", shape![4, 5, 6], shape![1, 1, 6]).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::Of([0usize, 2, 3].iter().copied().collect()),
                AxisRange::In(0, 5, 1),
                AxisRange::In(0, 6, 1)
            ])
            .expect("slice Of dense");
        assert_eq!(sliced.shape(), &[3, 5, 6]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![src_indices[coord[0] as usize], coord[1], coord[2]];
            let expected = tensor.read_value(&src).await.expect("orig dense gather");
            let actual = sliced
                .read_value(&coord)
                .await
                .expect("sliced dense gather");
            assert_eq!(actual, expected, "Of dense coord {:?}", coord);
        }
        cleanup(&root).await;

        let (root, tensor, _) =
            create_sparse("b_gather_sparse", shape![4, 5, 6], shape![1, 1, 6], Some(0)).await;
        seed_values(&tensor).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::Of([0usize, 2, 3].iter().copied().collect()),
                AxisRange::In(0, 5, 1),
                AxisRange::In(0, 6, 1)
            ])
            .expect("slice Of sparse");
        assert_eq!(sliced.shape(), &[3, 5, 6]);

        for coord in iter_coords(sliced.shape()) {
            let src = vec![src_indices[coord[0] as usize], coord[1], coord[2]];
            let expected = tensor.read_value(&src).await.expect("orig sparse gather");
            let actual = sliced
                .read_value(&coord)
                .await
                .expect("sliced sparse gather");
            assert_eq!(actual, expected, "Of sparse coord {:?}", coord);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn degenerate_slice_empty_axis() {
        let (root, tensor, _) =
            create_dense("b_empty_axis", shape![4, 5, 6], shape![1, 1, 6]).await;

        // In(2, 2, 1) produces extent = 0 on axis 0. The schema rejects zero-dim
        // shapes, so slice must return an error rather than panic.
        let result = tensor.clone().slice(range![
            AxisRange::In(2, 2, 1),
            AxisRange::In(0, 5, 1),
            AxisRange::In(0, 6, 1)
        ]);
        assert!(result.is_err(), "zero-extent slice must fail, got Ok");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn transpose_identity_permutation_is_noop() {
        let (root, tensor, _) = create_dense("b_id_perm", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let transposed = tensor
            .clone()
            .transpose(Some(axes![0, 1, 2]))
            .expect("identity tx");
        assert_eq!(transposed.shape(), tensor.shape());

        for coord in iter_coords(tensor.shape()) {
            let expected = tensor.read_value(&coord).await.expect("orig");
            let actual = transposed.read_value(&coord).await.expect("identity tx");
            assert_eq!(actual, expected, "identity perm coord {:?}", coord);
        }

        cleanup(&root).await;
    }
}

// ====================================================================
// Section C — New transforms in scope (broadcast, flip, squeeze, unsqueeze)
// All currently return Error::Unsupported via trait defaults. Tests pin
// the *intended* behavior; they will fail until those transforms land.
// ====================================================================

mod section_c_new_transforms {
    use super::*;

    #[tokio::test]
    async fn broadcast_repeats_size_one_axis() {
        let (root, tensor, _) = create_dense("c_broadcast", shape![1, 3, 4], shape![1, 1, 4]).await;
        for b in 0..3u64 {
            for c in 0..4u64 {
                tensor
                    .write_value(&[0, b, c], (b * 10 + c) as f32)
                    .await
                    .expect("seed");
            }
        }

        let broadcasted = tensor
            .clone()
            .broadcast(shape![2, 3, 4])
            .expect("broadcast must be supported");

        for a in 0..2u64 {
            for b in 0..3u64 {
                for c in 0..4u64 {
                    let v = broadcasted
                        .read_value(&[a, b, c])
                        .await
                        .expect("broadcast read");
                    let expected = (b * 10 + c) as f32;
                    assert_eq!(v, expected, "broadcast [{a},{b},{c}]");
                }
            }
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn flip_axis_reverses_indexing() {
        let (root, tensor, _) = create_dense("c_flip", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let flipped = tensor.clone().flip(2).expect("flip must be supported");

        for coord in iter_coords(flipped.shape()) {
            let src = vec![coord[0], coord[1], 3 - coord[2]];
            let expected = tensor.read_value(&src).await.expect("orig");
            let actual = flipped.read_value(&coord).await.expect("flipped");
            assert_eq!(actual, expected, "flip coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_removes_size_one_axes() {
        let (root, tensor, _) =
            create_dense("c_squeeze", shape![1, 3, 1, 4], shape![1, 1, 1, 4]).await;
        for b in 0..3u64 {
            for c in 0..4u64 {
                tensor
                    .write_value(&[0, b, 0, c], (b * 10 + c) as f32)
                    .await
                    .expect("seed");
            }
        }

        let squeezed = tensor
            .clone()
            .squeeze(axes![0, 2])
            .expect("squeeze must be supported");
        assert_eq!(squeezed.shape(), &[3, 4]);

        for b in 0..3u64 {
            for c in 0..4u64 {
                let v = squeezed.read_value(&[b, c]).await.expect("squeezed read");
                assert_eq!(v, (b * 10 + c) as f32);
            }
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unsqueeze_inserts_size_one_axes() {
        let (root, tensor, _) = create_dense("c_unsqueeze", shape![3, 4], shape![1, 4]).await;
        for b in 0..3u64 {
            for c in 0..4u64 {
                tensor
                    .write_value(&[b, c], (b * 10 + c) as f32)
                    .await
                    .expect("seed");
            }
        }

        let unsqueezed = tensor
            .clone()
            .unsqueeze(axes![0, 2])
            .expect("unsqueeze must be supported");
        assert_eq!(unsqueezed.shape(), &[1, 3, 1, 4]);

        for b in 0..3u64 {
            for c in 0..4u64 {
                let v = unsqueezed
                    .read_value(&[0, b, 0, c])
                    .await
                    .expect("unsqueeze read");
                assert_eq!(v, (b * 10 + c) as f32);
            }
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_non_size_one_axis_rejected() {
        let (root, tensor, _) =
            create_dense("c_squeeze_bad", shape![2, 3, 4], shape![1, 1, 4]).await;
        let err = tensor
            .clone()
            .squeeze(axes![1])
            .err()
            .expect("must reject squeeze on size>1 axis");
        assert!(
            matches!(err, Error::InvalidLayout(_) | Error::Unsupported(_)),
            "got {err:?}"
        );
        cleanup(&root).await;
    }
}

// ====================================================================
// Section D — Bulk read/write (TensorReadBulk / TensorWriteBulk)
// Currently inherit `Error::Unsupported` defaults.
// ====================================================================

mod section_d_bulk_io {
    use super::*;

    #[tokio::test]
    async fn read_values_dense_contiguous_range() {
        let (root, tensor, _) =
            create_dense("d_read_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let values = tensor
            .read_values(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .await
            .expect("bulk read must be supported");

        assert_eq!(values.len(), 24);
        for (idx, coord) in iter_coords(tensor.shape()).enumerate() {
            assert_eq!(values[idx], encode_value(&coord), "coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn read_values_sparse_returns_zeros_for_missing() {
        let (root, tensor, _) =
            create_sparse("d_read_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;
        tensor.write_value(&[0, 0, 0], 1.0).await.expect("write");
        tensor.write_value(&[1, 2, 3], 9.0).await.expect("write");

        let values = tensor
            .read_values(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .await
            .expect("bulk sparse read must be supported");

        assert_eq!(values.len(), 24);
        let mut nonzero = 0;
        for v in &values {
            if *v != 0.0 {
                nonzero += 1;
            }
        }
        assert_eq!(nonzero, 2);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn read_all_returns_full_tensor_row_major() {
        let (root, tensor, _) = create_dense("d_read_all", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let values = tensor.read_all().await.expect("read_all must be supported");
        assert_eq!(values.len(), 24);

        for (idx, coord) in iter_coords(tensor.shape()).enumerate() {
            assert_eq!(values[idx], encode_value(&coord));
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn write_values_round_trips() {
        let (root, tensor, _) =
            create_dense("d_write_vals", shape![2, 3, 4], shape![1, 1, 4]).await;

        let mut buf = Vec::with_capacity(24);
        for coord in iter_coords(tensor.shape()) {
            buf.push(encode_value(&coord));
        }

        tensor
            .write_values(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                buf,
            )
            .await
            .expect("bulk write must be supported");

        for coord in iter_coords(tensor.shape()) {
            let v = tensor.read_value(&coord).await.expect("read");
            assert_eq!(v, encode_value(&coord), "coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn fill_writes_default_then_nonzero() {
        let (root, tensor, schema) =
            create_sparse("d_fill", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        tensor.fill(0.0).await.expect("fill default supported");
        // After fill(default) on a `RemoveRow` sparse tensor the index should
        // be empty. Probe one expected coord.
        let pre = block_id_for_coord(&tensor, &schema, &[0, 0, 0]).await;
        assert!(pre.is_none(), "expected no rows after fill(default)");

        tensor.fill(1.0).await.expect("fill nonzero supported");
        for coord in iter_coords(tensor.shape()) {
            let v = tensor.read_value(&coord).await.expect("read");
            assert_eq!(v, 1.0, "coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn write_tensor_dense_to_sparse() {
        let (dense_root, dense, _) =
            create_dense("d_wt_dense", shape![2, 3, 4], shape![1, 1, 4]).await;
        let (sparse_root, sparse, _) =
            create_sparse("d_wt_sparse", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;
        seed_values(&dense).await;

        sparse
            .write_tensor(&dense)
            .await
            .expect("write_tensor must be supported");

        for coord in iter_coords(dense.shape()) {
            let d = dense.read_value(&coord).await.expect("dense");
            let s = sparse.read_value(&coord).await.expect("sparse");
            assert_eq!(d, s, "coord {:?}", coord);
        }

        cleanup(&dense_root).await;
        cleanup(&sparse_root).await;
    }
}

// ====================================================================
// Section E — Sparse zero-write lifecycle
// ====================================================================

mod section_e_sparse_lifecycle {
    use super::*;

    // With default `SparseZeroPolicy::RemoveRow`, writing default to a
    // never-present coord is a no-op (current behavior is correct here).
    #[tokio::test]
    async fn sparse_zero_to_nonzero_creates_row_and_block() {
        let (root, tensor, schema) = create_sparse(
            "e_zero_to_nonzero",
            shape![2, 3, 4],
            shape![1, 1, 4],
            Some(1),
        )
        .await;

        // No row before write.
        assert!(
            block_id_for_coord(&tensor, &schema, &[0, 1, 2])
                .await
                .is_none()
        );

        tensor.write_value(&[0, 1, 2], 1.0).await.expect("write");
        let block_id = block_id_for_coord(&tensor, &schema, &[0, 1, 2])
            .await
            .expect("row must exist");
        assert!(
            tensor.read_block(block_id).await.expect("block").is_some(),
            "block must be materialized after first nonzero write"
        );
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 1.0);

        cleanup(&root).await;
    }

    // Default policy `RemoveRow`: after nonzero→zero, the row must be gone.
    // Currently FAILS — the row is left in place after writing default.
    #[tokio::test]
    async fn sparse_nonzero_to_zero_remove_row_policy() {
        let (root, tensor, schema) =
            create_sparse("e_remove_row", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("write nz");
        assert!(
            block_id_for_coord(&tensor, &schema, &[0, 1, 2])
                .await
                .is_some()
        );

        tensor.write_value(&[0, 1, 2], 0.0).await.expect("write z");

        assert!(
            block_id_for_coord(&tensor, &schema, &[0, 1, 2])
                .await
                .is_none(),
            "RemoveRow policy must drop the row when value becomes default"
        );
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 0.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    #[ignore = "requires Tensor::with_sparse_zero_policy / create_with_policy"]
    async fn sparse_nonzero_to_zero_retain_zero_policy() {
        // Intended shape once the policy setter lands:
        //
        //   let tensor = Tensor::<FsEntry, f32>::create_with_policy(
        //       dir, schema, SparseZeroPolicy::RetainZero,
        //   ).await.expect("create");
        //   tensor.write_value(&[0, 1, 2], 5.0).await.expect("write nz");
        //   tensor.write_value(&[0, 1, 2], 0.0).await.expect("write z");
        //   assert!(tensor.lookup_block_id(&[1, 1]).await.unwrap().is_some());
        //   assert_eq!(tensor.read_value(&[0, 1, 2]).await.unwrap(), 0.0);
        panic!("requires Tensor::with_sparse_zero_policy / create_with_policy");
    }

    #[tokio::test]
    #[ignore = "requires Tensor::with_sparse_zero_policy / create_with_policy"]
    async fn sparse_nonzero_to_zero_tombstone_policy() {
        // Same shape as above, but with `SparseZeroPolicy::Tombstone`.
        // Exact contract (sentinel block_id vs zero-filled block) to be
        // specified during implementation.
        panic!("requires Tensor::with_sparse_zero_policy / create_with_policy");
    }

    #[tokio::test]
    async fn sparse_overwrite_nonzero_preserves_row_id() {
        let (root, tensor, schema) =
            create_sparse("e_overwrite", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        tensor.write_value(&[0, 1, 2], 1.0).await.expect("write 1");
        let id_a = block_id_for_coord(&tensor, &schema, &[0, 1, 2])
            .await
            .expect("must exist");

        tensor.write_value(&[0, 1, 2], 2.0).await.expect("write 2");
        let id_b = block_id_for_coord(&tensor, &schema, &[0, 1, 2])
            .await
            .expect("must exist");

        assert_eq!(id_a, id_b, "block_id must be stable across nonzero updates");
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 2.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn compact_sparse_removes_all_zero_rows() {
        let (root, tensor, schema) =
            create_sparse("e_compact", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("nz");
        tensor.write_value(&[0, 1, 2], 0.0).await.expect("zero");

        tensor
            .compact_sparse()
            .await
            .expect("compact must be supported");

        assert!(
            block_id_for_coord(&tensor, &schema, &[0, 1, 2])
                .await
                .is_none(),
            "compaction must drop all-zero rows"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    #[ignore = "requires policy persistence + setter"]
    async fn sparse_zero_policy_persists_across_reload() {
        // Once the policy setter lands and is persisted in metadata:
        //   1. Create tensor with non-default policy
        //   2. Write values, sync, drop
        //   3. Reload, assert sparse_zero_policy() matches what was set
        panic!("requires SparseZeroPolicy persistence");
    }
}

// ====================================================================
// Section F — View semantics / writeability
// ====================================================================

mod section_f_view_semantics {
    use super::*;

    #[tokio::test]
    async fn is_base_tensor_true_only_for_identity_view() {
        let (root, tensor, _) = create_dense("f_is_base", shape![2, 3, 4], shape![1, 1, 4]).await;

        assert!(tensor.is_base_tensor(), "fresh tensor is the base");

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");
        assert!(!sliced.is_base_tensor(), "sliced view is not the base");

        let transposed = tensor.clone().transpose(Some(axes![2, 0, 1])).expect("tx");
        assert!(
            !transposed.is_base_tensor(),
            "transposed view is not the base"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn supports_write_through_true_for_slice_and_transpose() {
        let (root, tensor, _) =
            create_dense("f_writethrough", shape![2, 3, 4], shape![1, 1, 4]).await;

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(1, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");
        assert!(sliced.supports_write_through());

        sliced
            .write_value(&[0, 0, 0], 42.0)
            .await
            .expect("write through slice");

        // The slice maps view-coord [0,0,0] to base-coord [0,1,0].
        let from_base = tensor.read_value(&[0, 1, 0]).await.expect("base read");
        assert_eq!(from_base, 42.0, "write must reach base storage");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn write_through_rejected_for_broadcast() {
        let (root, tensor, _) = create_dense("f_no_wt", shape![1, 3, 4], shape![1, 1, 4]).await;
        let broadcasted = match tensor.clone().broadcast(shape![2, 3, 4]) {
            Ok(b) => b,
            Err(_) => {
                // Broadcast not yet implemented — skip the rest of the test;
                // the writeability assertion is moot until broadcast exists.
                cleanup(&root).await;
                return;
            }
        };

        assert!(
            !broadcasted.supports_write_through(),
            "broadcast view must NOT be write-through"
        );

        let err = broadcasted
            .write_value(&[1, 0, 0], 1.0)
            .await
            .expect_err("write must be rejected on broadcast view");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn reshape_view_writeability_documented() {
        let (root, tensor, _) =
            create_dense("f_reshape_view", shape![2, 3, 4], shape![1, 1, 4]).await;

        // Reshape over identity is fine.
        let reshaped = tensor.clone().reshape(shape![6, 4]).expect("reshape ok");
        assert!(reshaped.supports_write_through());

        // Reshape over a transformed view is rejected (current contract).
        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");
        let err = sliced.reshape(shape![16]).err().expect("must reject");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

        cleanup(&root).await;
    }
}

// ====================================================================
// Section G — `read_sparse_elements_in_order` semantics
// ====================================================================

mod section_g_sparse_iteration {
    use super::*;

    #[tokio::test]
    async fn in_order_iteration_matches_base_order() {
        let (root, tensor, _) =
            create_sparse("g_in_order", shape![2, 3, 4], shape![1, 1, 4], Some(0)).await;

        tensor.write_value(&[0, 0, 0], 1.0).await.expect("w");
        tensor.write_value(&[1, 2, 3], 9.0).await.expect("w");
        tensor.write_value(&[1, 0, 1], 4.0).await.expect("w");

        let rows = tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![0, 1, 2],
            )
            .await
            .expect("compatible order must succeed");

        assert_eq!(rows.len(), 3, "should yield exactly the written coords");
        // First coordinate in axis-0 order: [0,0,0]
        assert_eq!(rows[0].0, vec![0u64, 0, 0]);
        assert_eq!(rows[0].1, 1.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn in_order_iteration_with_partial_range() {
        let (root, tensor, _) =
            create_sparse("g_partial", shape![2, 3, 4], shape![1, 1, 4], Some(0)).await;

        tensor.write_value(&[0, 0, 0], 1.0).await.expect("w");
        tensor.write_value(&[1, 2, 3], 9.0).await.expect("w");

        let rows = tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(1, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![0, 1, 2],
            )
            .await
            .expect("partial range supported");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, vec![1u64, 2, 3]);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn incompatible_order_returns_structured_error() {
        let (root, tensor, _) =
            create_sparse("g_bad_order", shape![2, 3, 4], shape![1, 1, 4], Some(0)).await;

        let err = tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![2, 0, 1],
            )
            .await
            .expect_err("incompatible order must fail");

        match err {
            Error::UnsupportedSparseIterationOrder {
                requested_order,
                base_order,
                hint,
            } => {
                assert_eq!(requested_order, vec![2, 0, 1]);
                assert_eq!(base_order, vec![0, 1, 2]);
                assert!(!hint.is_empty(), "hint must be actionable");
            }
            other => panic!("unexpected error variant: {other}"),
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn in_order_iteration_over_sliced_view() {
        let (root, tensor, _) =
            create_sparse("g_sliced", shape![4, 3, 4], shape![1, 1, 4], Some(0)).await;
        tensor.write_value(&[0, 0, 0], 1.0).await.expect("w");
        tensor.write_value(&[1, 0, 0], 2.0).await.expect("w");
        tensor.write_value(&[2, 0, 0], 3.0).await.expect("w");

        let sliced = tensor
            .clone()
            .slice(range![
                AxisRange::In(1, 3, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");

        let result = sliced
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![0, 1, 2],
            )
            .await;

        // v1 may choose either: support this case OR return a structured
        // `UnsupportedSparseIterationOrder` with a clear hint.
        match result {
            Ok(rows) => {
                assert_eq!(rows.len(), 2);
            }
            Err(Error::UnsupportedSparseIterationOrder { hint, .. }) => {
                assert!(!hint.is_empty(), "hint must be actionable");
            }
            Err(other) => panic!("unexpected error variant: {other}"),
        }

        cleanup(&root).await;
    }
}

// ====================================================================
// Section H — Persistence + corruption (real freqfs)
// ====================================================================

mod section_h_persistence {
    use super::*;

    #[tokio::test]
    async fn dense_full_roundtrip_reload() {
        let root = common::unique_tmp_dir("h_dense_reload");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4], shape![1, 1, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let tensor = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            for coord in iter_coords(tensor.shape()) {
                tensor
                    .write_value(&coord, encode_value(&coord))
                    .await
                    .expect("write");
            }
            dir.sync().await.expect("sync");
        }

        let dir2 = open_dir(&root).expect("reopen");
        let loaded = Tensor::<FsEntry, f32>::load_with_schema(dir2, &schema)
            .await
            .expect("reload");
        for coord in iter_coords(loaded.shape()) {
            let v = loaded.read_value(&coord).await.expect("read");
            assert_eq!(v, encode_value(&coord), "post-reload coord {:?}", coord);
        }

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_full_roundtrip_reload() {
        let root = common::unique_tmp_dir("h_sparse_reload");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = sparse_schema_f32(shape![2, 3, 4], shape![1, 1, 4], Some(1));

        {
            let dir = open_dir(&root).expect("open");
            let tensor = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            tensor.write_value(&[0, 0, 0], 1.0).await.expect("w");
            tensor.write_value(&[1, 2, 3], 9.0).await.expect("w");
            dir.sync().await.expect("sync");
        }

        let dir2 = open_dir(&root).expect("reopen");
        let loaded = Tensor::<FsEntry, f32>::load_with_schema(dir2, &schema)
            .await
            .expect("reload");
        assert_eq!(loaded.read_value(&[0, 0, 0]).await.expect("read"), 1.0);
        assert_eq!(loaded.read_value(&[1, 2, 3]).await.expect("read"), 9.0);
        assert_eq!(loaded.read_value(&[1, 0, 0]).await.expect("read"), 0.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn view_schema_persists_via_with_view_schema() {
        let root = common::unique_tmp_dir("h_view_persist");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4], shape![1, 1, 4]);

        let view_schema = {
            let dir = open_dir(&root).expect("open");
            let tensor = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            seed_values(&tensor).await;
            let transposed = tensor.clone().transpose(Some(axes![2, 0, 1])).expect("tx");
            let vs = transposed.view_schema().expect("view schema");
            dir.sync().await.expect("sync");
            vs
        };

        let dir2 = open_dir(&root).expect("reopen");
        let loaded = Tensor::<FsEntry, f32>::load_with_schema(dir2, &schema)
            .await
            .expect("reload");
        let rehydrated = loaded
            .with_view_schema(&view_schema)
            .expect("rehydrate view");
        assert_eq!(rehydrated.shape(), &[4, 2, 3]);
        // sample one coord to verify the view still resolves correctly
        let _ = rehydrated.read_value(&[0, 0, 0]).await.expect("read");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn metadata_file_missing_fails_closed() {
        let root = common::unique_tmp_dir("h_meta_missing");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4], shape![1, 1, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            dir.sync().await.expect("sync");
        }

        // Delete the metadata file on disk.
        let meta = root.join("blocks").join("metadata");
        // Best-effort: freqfs may name the file with an extension or
        // wrapper; try the bare name first and fall back to wildcard remove.
        if tokio::fs::remove_file(&meta).await.is_err() {
            // Wildcard removal: drop the whole blocks dir contents.
            let _ = tokio::fs::remove_dir_all(root.join("blocks")).await;
        }

        let dir2 = open_dir(&root).expect("reopen");
        let err = Tensor::<FsEntry, f32>::load(dir2)
            .await
            .err()
            .expect("missing metadata must fail closed");
        assert!(
            matches!(err, Error::InvalidSchema(_) | Error::Io(_)),
            "got {err:?}"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn metadata_file_tampered_fails_closed() {
        let root = common::unique_tmp_dir("h_meta_tampered");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4], shape![1, 1, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            dir.sync().await.expect("sync");
        }

        // Overwrite the metadata file with garbage. freqfs encodes via
        // destream, so a raw write produces a structured-decode failure.
        let meta = root.join("blocks").join("metadata");
        let _ = tokio::fs::write(&meta, b"not a tensor metadata file").await;

        let dir2 = open_dir(&root).expect("reopen");
        let err = Tensor::<FsEntry, f32>::load(dir2)
            .await
            .err()
            .expect("tampered metadata must fail closed");
        assert!(
            matches!(err, Error::InvalidSchema(_) | Error::Io(_)),
            "got {err:?}"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_index_points_to_missing_block_on_read() {
        let root = common::unique_tmp_dir("h_index_orphan");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = sparse_schema_f32(shape![2, 3, 4], shape![1, 1, 4], Some(1));

        let block_id = {
            let dir = open_dir(&root).expect("open");
            let tensor = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            tensor.write_value(&[0, 1, 2], 5.0).await.expect("write");
            let id = block_id_for_coord(&tensor, &schema, &[0, 1, 2])
                .await
                .expect("row must exist");
            dir.sync().await.expect("sync");
            id
        };

        // Delete the orphan block file directly.
        let blocks_dir = root.join("blocks");
        let target = blocks_dir.join(block_id.to_string());
        let _ = tokio::fs::remove_file(&target).await;

        let dir2 = open_dir(&root).expect("reopen");
        let loaded = Tensor::<FsEntry, f32>::load_with_schema(dir2, &schema)
            .await
            .expect("reload");
        let err = loaded
            .read_value(&[0, 1, 2])
            .await
            .expect_err("orphan index row must fail closed");
        assert!(
            matches!(err, Error::SparseIndex(_) | Error::Io(_)),
            "got {err:?}"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unknown_metadata_version_rejected_on_reload() {
        let root = common::unique_tmp_dir("h_meta_version");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4], shape![1, 1, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone())
                .await
                .expect("create");
            dir.sync().await.expect("sync");
        }

        // Rewrite the metadata file with a structurally valid but unsupported version.
        let meta = root.join("blocks").join("metadata");
        let bad = "version=999\ndtype=f32\nlayout=dense\nshape=2,3,4\nblock_shape=1,1,4\nstrides=12,4,1\n";
        let _ = tokio::fs::write(&meta, bad).await;

        let dir2 = open_dir(&root).expect("reopen");
        let err = Tensor::<FsEntry, f32>::load(dir2)
            .await
            .err()
            .expect("unknown version must fail closed");
        assert!(
            matches!(err, Error::InvalidSchema(_) | Error::Io(_)),
            "got {err:?}"
        );

        cleanup(&root).await;
    }
}
