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
    BoxFuture, DType, Error, Layout, Tensor, TensorArray, TensorBlockStore, TensorGeometry,
    TensorRead, TensorReadBulk, TensorSchema, TensorSparseIndex, TensorTransform,
    TensorViewSemantics, TensorWrite, TensorWriteBulk, contiguous_strides,
};
use futures::TryStreamExt;
use ha_ndarray::{Axes, AxisRange, Range, Shape, axes, range, shape};

use common::{FsEntry, cleanup, iter_coords, new_dir, open_dir};

fn dense_schema_f32(shape: Shape) -> TensorSchema {
    TensorSchema::new(DType::F32, shape).expect("schema")
}

fn sparse_schema_f32(shape: Shape) -> TensorSchema {
    TensorSchema::new(DType::F32, shape).expect("schema")
}

async fn create_dense(
    name: &str,
    shape: Shape,
    block_shape: Shape,
) -> (PathBuf, Tensor<FsEntry, f32>, TensorSchema) {
    let (root, dir) = new_dir(name).await;
    let schema = dense_schema_f32(shape);
    let max_capacity = block_shape.iter().product::<usize>().max(1);
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema.clone(), Layout::Dense, max_capacity)
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
    let schema = sparse_schema_f32(shape);
    let max_capacity = block_shape.iter().product::<usize>().max(1);
    let tensor =
        Tensor::<FsEntry, f32>::create(dir, schema.clone(), Layout::Sparse { axis }, max_capacity)
            .await
            .expect("create sparse");
    (root, tensor, schema)
}

fn encode_value(coord: &[u64]) -> f32 {
    (coord[0] as f32) * 100.0 + (coord[1] as f32) * 10.0 + (coord[2] as f32)
}

async fn seed_values<T>(tensor: &T)
where
    T: TensorWrite<DType = f32>,
{
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
        let (root, tensor, _schema) =
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

        // TODO: block materialization checks require re-deriving via directory-listing
        // since block_id_for_coord was removed

        for (coord, value) in &writes {
            tensor.write_value(coord, *value).await.expect("write");
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
        let sliced = tensor.view().slice(r).expect("slice");
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
        let transposed = tensor.view().transpose(Some(perm.clone())).expect("tx");
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

        let reshaped = tensor.view().reshape(shape![6, 4]).expect("reshape");
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
        let transposed = tensor.view().transpose(Some(perm.clone())).expect("tx");
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
        let sliced = tensor.view().slice(r).expect("slice");

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
            .view()
            .slice(r.clone())
            .expect("slice")
            .transpose(Some(perm.clone()))
            .expect("tx");

        let remapped = transpose_range(&r, &perm);
        let right = tensor
            .view()
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
            .view()
            .slice(r.clone())
            .expect("slice")
            .transpose(Some(perm.clone()))
            .expect("tx");

        let remapped = transpose_range(&r, &perm);
        let right = tensor
            .view()
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
            .view()
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
            .view()
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
            .view()
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
            .view()
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
            .view()
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
            .view()
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
            .view()
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
            .view()
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
        let result = tensor.view().slice(range![
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
            .view()
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

    #[tokio::test]
    async fn two_simultaneous_views_from_same_tensor() {
        let (root, tensor, _) = create_dense("b_two_views", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let r: Range = range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(1, 3, 1),
            AxisRange::In(0, 4, 2)
        ];
        let perm = axes![2, 0, 1];

        // Two independent borrows of `tensor` held live at the same time.
        let left = tensor.view().slice(r).expect("slice");
        let right = tensor.view().transpose(Some(perm.clone())).expect("tx");

        assert_eq!(left.shape(), &[2, 2, 2]);
        assert_eq!(right.shape(), &[4, 2, 3]);

        let mut inverse = vec![0usize; perm.len()];
        for (i, axis) in perm.iter().enumerate() {
            inverse[*axis] = i;
        }

        for s_coord in iter_coords(left.shape()) {
            let src = vec![s_coord[0], s_coord[1] + 1, s_coord[2] * 2];
            let expected = tensor.read_value(&src).await.expect("read original");
            let actual = left.read_value(&s_coord).await.expect("read sliced");
            assert_eq!(actual, expected, "left coord {:?}", s_coord);
        }

        for t_coord in iter_coords(right.shape()) {
            let mut src = vec![0u64; t_coord.len()];
            for old_axis in 0..t_coord.len() {
                src[old_axis] = t_coord[inverse[old_axis]];
            }
            let expected = tensor.read_value(&src).await.expect("read original");
            let actual = right.read_value(&t_coord).await.expect("read transposed");
            assert_eq!(actual, expected, "right coord {:?}", t_coord);
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
            .view()
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

        let flipped = tensor.view().flip(2).expect("flip must be supported");

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
            .view()
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
            .view()
            .unsqueeze(axes![0, 1])
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
            .view()
            .squeeze(axes![1])
            .err()
            .expect("must reject squeeze on size>1 axis");
        assert!(
            matches!(err, Error::InvalidLayout(_) | Error::Unsupported(_)),
            "got {err:?}"
        );
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_removes_size_one_axes_sparse() {
        let (root, tensor, _) = create_sparse(
            "c_squeeze_sparse",
            shape![1, 3, 1, 4],
            shape![1, 1, 1],
            Some(0),
        )
        .await;
        seed_values(&tensor).await;

        let squeezed = tensor
            .view()
            .squeeze(axes![0, 2])
            .expect("squeeze must be supported");
        assert_eq!(squeezed.shape(), &[3, 4]);

        for b in 0..3u64 {
            for c in 0..4u64 {
                let v = squeezed.read_value(&[b, c]).await.expect("squeezed read");
                assert_eq!(v, encode_value(&[0, b, 0, c]), "coord [{},{c}]", b);
            }
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unsqueeze_inserts_size_one_axes_sparse() {
        let (root, tensor, _) = create_sparse(
            "c_unsqueeze_sparse",
            shape![2, 3, 4],
            shape![1, 1, 4],
            Some(0),
        )
        .await;
        seed_values(&tensor).await;

        let unsqueezed = tensor
            .view()
            .unsqueeze(axes![0, 2])
            .expect("unsqueeze must be supported");
        assert_eq!(unsqueezed.shape(), &[1, 2, 3, 1, 4]);

        for a in 0..2u64 {
            for b in 0..3u64 {
                for c in 0..4u64 {
                    let v = unsqueezed
                        .read_value(&[0, a, b, 0, c])
                        .await
                        .expect("unsqueezed read");
                    assert_eq!(v, encode_value(&[a, b, c]), "coord [0,{},{},0,{}]", a, b, c);
                }
            }
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_duplicate_axis_rejected() {
        let (root, tensor, _) =
            create_dense("c_squeeze_dup", shape![2, 3, 4], shape![1, 1, 1]).await;
        let err = tensor
            .view()
            .squeeze(axes![1, 1])
            .err()
            .expect("must reject duplicate axis");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_axis_out_of_bounds_rejected() {
        let (root, tensor, _) =
            create_dense("c_squeeze_oob", shape![2, 3, 4], shape![1, 1, 1]).await;
        let err = tensor
            .view()
            .squeeze(axes![5])
            .err()
            .expect("must reject out of bounds");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unsqueeze_duplicate_axis_rejected() {
        let (root, tensor, _) = create_dense("c_unsqueeze_dup", shape![3, 4], shape![1, 1]).await;
        let err = tensor
            .view()
            .unsqueeze(axes![0, 0])
            .err()
            .expect("must reject duplicate axis");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unsqueeze_axis_out_of_bounds_rejected() {
        let (root, tensor, _) = create_dense("c_unsqueeze_oob", shape![3, 4], shape![1, 1]).await;
        // For rank-2 tensor, valid axes are 0,1 only
        let err2 = tensor
            .view()
            .unsqueeze(axes![2])
            .err()
            .expect("must reject axis 2");
        assert!(matches!(err2, Error::InvalidLayout(_)), "got {err2:?}");

        let err3 = tensor
            .view()
            .unsqueeze(axes![3])
            .err()
            .expect("must reject axis 3");
        assert!(matches!(err3, Error::InvalidLayout(_)), "got {err3:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_empty_axes_rejected() {
        let (root, tensor, _) =
            create_dense("c_squeeze_empty", shape![3, 1, 4], shape![1, 1, 1]).await;
        let err = tensor
            .view()
            .squeeze(axes![])
            .err()
            .expect("must reject empty axes");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unsqueeze_empty_axes_rejected() {
        let (root, tensor, _) = create_dense("c_unsqueeze_empty", shape![3, 4], shape![1, 1]).await;
        let err = tensor
            .view()
            .unsqueeze(axes![])
            .err()
            .expect("must reject empty axes");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_all_axes_rejected() {
        let (root, tensor, _) = create_dense("c_squeeze_all", shape![1, 1], shape![1, 1]).await;
        let err = tensor
            .view()
            .squeeze(axes![0, 1])
            .err()
            .expect("must reject squeezing all axes");
        assert!(matches!(err, Error::InvalidLayout(_)), "got {err:?}");
        cleanup(&root).await;
    }
}

// ====================================================================
// Section C2 — Squeeze/Unsqueeze chaining with other transforms
// ====================================================================

mod section_c2_squeeze_unsqueeze_chains {
    use super::*;

    #[tokio::test]
    async fn chained_slice_then_squeeze_vs_squeeze_then_slice() {
        // shape [2,1,3,4] (size-1 axis at original position 1)
        let (root, tensor, _) =
            create_dense("c2_slice_squeeze", shape![2, 1, 3, 4], shape![1, 1, 1]).await;
        seed_values(&tensor).await;

        // Path (a): slice then squeeze
        let sliced_a = tensor
            .view()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 1, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");
        let squeezed_a = sliced_a.squeeze(axes![1]).expect("squeeze");

        // Path (b): squeeze then slice
        let squeezed_b = tensor.view().squeeze(axes![1]).expect("squeeze");
        let sliced_b = squeezed_b
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");

        assert_eq!(squeezed_a.shape(), sliced_b.shape());
        for coord in iter_coords(squeezed_a.shape()) {
            let v_a = squeezed_a.read_value(&coord).await.expect("read a");
            let v_b = sliced_b.read_value(&coord).await.expect("read b");
            assert_eq!(v_a, v_b, "coord {:?}", coord);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_unsqueeze_then_transpose_dense() {
        // shape [2,3,4], .unsqueeze(axes![1]) -> [2,1,3,4]
        // then .transpose(Some(axes![0,3,1,2])) -> [2,4,1,3]
        let (root, tensor, _) =
            create_dense("c2_unsqueeze_transpose", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let unsqueezed = tensor.view().unsqueeze(axes![1]).expect("unsqueeze");
        assert_eq!(unsqueezed.shape(), &[2, 1, 3, 4]);

        let transposed = unsqueezed
            .transpose(Some(axes![0, 3, 1, 2]))
            .expect("transpose");
        assert_eq!(transposed.shape(), &[2, 4, 1, 3]);

        // Verify reads: transposed[a,c,0,b] should equal original [a,b,c]
        for a in 0..2u64 {
            for b in 0..3u64 {
                for c in 0..4u64 {
                    let v = transposed.read_value(&[a, c, 0, b]).await.expect("read");
                    let expected = encode_value(&[a, b, c]);
                    assert_eq!(v, expected, "coord [{},{},0,{}]", a, c, b);
                }
            }
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_unsqueeze_then_flip_is_noop_on_new_axis() {
        // shape [2,3,4], .unsqueeze(axes![1]) -> [2,1,3,4]
        // then .flip(1) (flip the new size-1 axis, which is a no-op)
        let (root, tensor, _) =
            create_dense("c2_unsqueeze_flip", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let unsqueezed = tensor.view().unsqueeze(axes![1]).expect("unsqueeze");
        let flipped = unsqueezed.clone().flip(1).expect("flip");

        // Flipping a dim-1 axis is a no-op, so reads should be identical
        for coord in iter_coords(unsqueezed.shape()) {
            let v_unflipped = unsqueezed.read_value(&coord).await.expect("read unflipped");
            let v_flipped = flipped.read_value(&coord).await.expect("read flipped");
            assert_eq!(v_flipped, v_unflipped, "coord {:?}", coord);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_unsqueeze_then_broadcast() {
        // shape [2,3,4], .unsqueeze(axes![0]) -> [1,2,3,4]
        // then .broadcast([5,2,3,4]) (broadcast new axis from 1 to 5)
        let (root, tensor, _) =
            create_dense("c2_unsqueeze_broadcast", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let unsqueezed = tensor.view().unsqueeze(axes![0]).expect("unsqueeze");
        let broadcasted = unsqueezed.broadcast(shape![5, 2, 3, 4]).expect("broadcast");
        assert_eq!(broadcasted.shape(), &[5, 2, 3, 4]);

        // Each k in 0..5 should read the same value as the original tensor at [a,b,c]
        for k in 0..5u64 {
            for a in 0..2u64 {
                for b in 0..3u64 {
                    for c in 0..4u64 {
                        let v = broadcasted.read_value(&[k, a, b, c]).await.expect("read");
                        let expected = encode_value(&[a, b, c]);
                        assert_eq!(v, expected, "coord [{},{},{},{}]", k, a, b, c);
                    }
                }
            }
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_squeeze_then_reshape_dense() {
        // shape [1,3,4], .squeeze(axes![0]) -> [3,4]
        // then .reshape([12])
        let (root, tensor, _) =
            create_dense("c2_squeeze_reshape", shape![1, 3, 4], shape![1, 1]).await;
        seed_values(&tensor).await;

        let squeezed = tensor.view().squeeze(axes![0]).expect("squeeze");
        assert_eq!(squeezed.shape(), &[3, 4]);

        let reshaped = squeezed.reshape(shape![12]).expect("reshape");
        assert_eq!(reshaped.shape(), &[12]);

        // Verify values match row-major order
        for idx in 0..12u64 {
            let b = idx / 4;
            let c = idx % 4;
            let v = reshaped.read_value(&[idx]).await.expect("read");
            let expected = encode_value(&[0, b, c]);
            assert_eq!(v, expected, "idx={}", idx);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn chained_unsqueeze_then_reshape_dense() {
        // shape [2,3,4], .unsqueeze(axes![0]) -> [1,2,3,4]
        // then .reshape([24])
        let (root, tensor, _) =
            create_dense("c2_unsqueeze_reshape", shape![2, 3, 4], shape![1, 1, 4]).await;
        seed_values(&tensor).await;

        let unsqueezed = tensor.view().unsqueeze(axes![0]).expect("unsqueeze");
        assert_eq!(unsqueezed.shape(), &[1, 2, 3, 4]);

        let reshaped = unsqueezed.reshape(shape![24]).expect("reshape");
        assert_eq!(reshaped.shape(), &[24]);

        // Verify values match row-major order
        for idx in 0..24u64 {
            let a = (idx / 12) % 2;
            let b = (idx / 4) % 3;
            let c = idx % 4;
            let v = reshaped.read_value(&[idx]).await.expect("read");
            let expected = encode_value(&[a, b, c]);
            assert_eq!(v, expected, "idx={}", idx);
        }
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn squeeze_then_unsqueeze_round_trip_sparse() {
        // sparse shape [3,1,4], squeeze axis 1 then unsqueeze back
        let (root, tensor, _) = create_sparse(
            "c2_squeeze_unsqueeze_sparse",
            shape![3, 1, 4],
            shape![1, 1],
            Some(0),
        )
        .await;
        seed_values(&tensor).await;

        let squeezed = tensor.view().squeeze(axes![1]).expect("squeeze");
        assert_eq!(squeezed.shape(), &[3, 4]);

        let restored = squeezed.unsqueeze(axes![1]).expect("unsqueeze");
        assert_eq!(restored.shape(), &[3, 1, 4]);

        // Verify every coordinate reads the original tensor's value
        for coord in iter_coords(tensor.shape()) {
            let v_orig = tensor.read_value(&coord).await.expect("read orig");
            let v_restored = restored.read_value(&coord).await.expect("read restored");
            assert_eq!(v_restored, v_orig, "coord {:?}", coord);
        }
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
        let (root, tensor, _) =
            create_sparse("d_fill", shape![2, 3, 4], shape![1, 1, 4], Some(1)).await;

        tensor.fill(0.0).await.expect("fill default supported");

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

// ====================================================================
// Section F — View semantics / writeability
// ====================================================================

mod section_f_view_semantics {
    use super::*;

    #[tokio::test]
    async fn is_base_tensor_true_only_for_identity_view() {
        let (root, tensor, _) = create_dense("f_is_base", shape![2, 3, 4], shape![1, 1, 4]).await;

        assert!(tensor.view().is_base_tensor(), "fresh tensor is the base");

        let sliced = tensor
            .view()
            .slice(range![
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 2, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");
        assert!(!sliced.is_base_tensor(), "sliced view is not the base");

        let transposed = tensor.view().transpose(Some(axes![2, 0, 1])).expect("tx");
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
            .view()
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
        let broadcasted = match tensor.view().broadcast(shape![2, 3, 4]) {
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
    async fn write_through_rejected_for_gather_slice() {
        let (root, tensor, _) =
            create_dense("f_no_wt_gather", shape![4, 3, 4], shape![1, 1, 4]).await;
        let gathered = tensor
            .view()
            .slice(range![
                AxisRange::Of(shape![0, 2]),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 1)
            ])
            .expect("slice");

        assert!(
            !gathered.supports_write_through(),
            "gather-sliced view must NOT be write-through"
        );

        let err = gathered
            .write_value(&[0, 0, 0], 1.0)
            .await
            .expect_err("write must be rejected on gather-sliced view");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn reshape_view_writeability_documented() {
        let (root, tensor, _) =
            create_dense("f_reshape_view", shape![2, 3, 4], shape![1, 1, 4]).await;

        // Reshape over identity is fine.
        let reshaped = tensor.view().reshape(shape![6, 4]).expect("reshape ok");
        assert!(reshaped.supports_write_through());

        // Reshape over a transformed view is rejected (current contract).
        let sliced = tensor
            .view()
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

        let rows: Vec<(Vec<u64>, f32)> = tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![0, 1, 2],
            )
            .await
            .expect("compatible order must succeed")
            .try_collect()
            .await
            .expect("stream must not error");

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

        let rows: Vec<(Vec<u64>, f32)> = tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(1, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![0, 1, 2],
            )
            .await
            .expect("partial range supported")
            .try_collect()
            .await
            .expect("stream must not error");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, vec![1u64, 2, 3]);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn incompatible_order_returns_structured_error() {
        let (root, tensor, _) =
            create_sparse("g_bad_order", shape![2, 3, 4], shape![1, 1, 4], Some(0)).await;

        let err = match tensor
            .read_sparse_elements_in_order(
                range![
                    AxisRange::In(0, 2, 1),
                    AxisRange::In(0, 3, 1),
                    AxisRange::In(0, 4, 1)
                ],
                axes![2, 0, 1],
            )
            .await
        {
            Ok(_) => panic!("incompatible order must fail"),
            Err(e) => e,
        };

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
            .view()
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

        match result {
            Ok(stream) => {
                let rows: Vec<(Vec<u64>, f32)> =
                    stream.try_collect().await.expect("stream must not error");
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
        let schema = dense_schema_f32(shape![2, 3, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let tensor =
                Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone(), Layout::Dense, 4)
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
        let loaded = Tensor::<FsEntry, f32>::load(dir2).await.expect("reload");
        assert_eq!(schema, *loaded.schema());
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
        let schema = sparse_schema_f32(shape![2, 3, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let tensor = Tensor::<FsEntry, f32>::create(
                dir.clone(),
                schema.clone(),
                Layout::Sparse { axis: Some(1) },
                4,
            )
            .await
            .expect("create");
            tensor.write_value(&[0, 0, 0], 1.0).await.expect("w");
            tensor.write_value(&[1, 2, 3], 9.0).await.expect("w");
            dir.sync().await.expect("sync");
        }

        let dir2 = open_dir(&root).expect("reopen");
        let loaded = Tensor::<FsEntry, f32>::load(dir2).await.expect("reload");
        assert_eq!(schema, *loaded.schema());
        assert_eq!(loaded.read_value(&[0, 0, 0]).await.expect("read"), 1.0);
        assert_eq!(loaded.read_value(&[1, 2, 3]).await.expect("read"), 9.0);
        assert_eq!(loaded.read_value(&[1, 0, 0]).await.expect("read"), 0.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn metadata_file_missing_fails_closed() {
        let root = common::unique_tmp_dir("h_meta_missing");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone(), Layout::Dense, 4)
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
        let schema = dense_schema_f32(shape![2, 3, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone(), Layout::Dense, 4)
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
        let (root, dir) = new_dir("h_index_orphan").await;
        let schema = sparse_schema_f32(shape![2, 3, 4]);

        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            schema.clone(),
            Layout::Sparse { axis: Some(1) },
            4,
        )
        .await
        .expect("create");
        tensor.write_value(&[0, 1, 2], 5.0).await.expect("write");

        let blocks_dir = {
            let guard = dir.read().await;
            guard.get_dir("blocks").cloned().expect("blocks dir")
        };
        let block_id = {
            let blocks_guard = blocks_dir.read().await;
            blocks_guard
                .names()
                .find(|name| name.as_str() != "metadata")
                .cloned()
                .expect("the single written row's block file")
        };
        {
            let mut blocks_guard = blocks_dir.write().await;
            blocks_guard.delete(&block_id).await;
        }
        blocks_dir.sync().await.expect("sync deleted block");

        let err = tensor
            .read_value(&[0, 1, 2])
            .await
            .expect_err("orphan index row must fail closed");
        assert!(matches!(err, Error::Io(_)), "got {err:?}");

        cleanup(&root).await;
    }

    // Dense blocks are eagerly materialized at `create` (see
    // `dense_create_materializes_all_blocks_on_disk`), so a missing block
    // file for a valid coord is corruption, not a legitimate lazy-block
    // state — `read_value` must fail closed rather than fall back to
    // `T::default()`.
    #[tokio::test]
    async fn dense_block_missing_on_read_fails_closed() {
        let (root, dir) = new_dir("h_dense_block_missing").await;
        let schema = dense_schema_f32(shape![2, 3, 4]);

        let tensor = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone(), Layout::Dense, 4)
            .await
            .expect("create dense");
        tensor.write_value(&[0, 1, 2], 5.0).await.expect("write");

        let blocks_dir = {
            let guard = dir.read().await;
            guard.get_dir("blocks").cloned().expect("blocks dir")
        };
        let block_id = {
            let blocks_guard = blocks_dir.read().await;
            blocks_guard
                .names()
                .find(|name| name.as_str() != "metadata")
                .cloned()
                .expect("at least one block file")
        };
        {
            let mut blocks_guard = blocks_dir.write().await;
            blocks_guard.delete(&block_id).await;
        }
        blocks_dir.sync().await.expect("sync deleted block");

        let mut saw_io_error = false;
        for coord in iter_coords(&[2, 3, 4]) {
            if let Err(err) = tensor.read_value(&coord).await {
                assert!(matches!(err, Error::Io(_)), "got {err:?}");
                saw_io_error = true;
            }
        }
        assert!(
            saw_io_error,
            "deleting a block file must make at least one coordinate fail closed"
        );

        cleanup(&root).await;
    }

    // On a freshly created (unwritten) dense tensor, every block position
    // implied by shape/block_shape must already have a block file on disk —
    // dense storage is fully materialized at `create`, not populated lazily
    // on first write.
    #[tokio::test]
    async fn dense_create_materializes_all_blocks_on_disk() {
        let root = common::unique_tmp_dir("h_dense_all_blocks");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let shape = shape![2, 3, 4];
        let max_capacity: usize = 4; // matches the original block_shape![1, 1, 4]'s product
        let schema = dense_schema_f32(shape.clone());
        let expected_blocks = (shape.iter().product::<usize>() as u64) / (max_capacity as u64);

        let dir = open_dir(&root).expect("open");
        let _tensor =
            Tensor::<FsEntry, f32>::create(dir.clone(), schema, Layout::Dense, max_capacity)
                .await
                .expect("create dense");
        dir.sync().await.expect("sync");

        let mut block_files = Vec::new();
        let mut rd = tokio::fs::read_dir(root.join("blocks"))
            .await
            .expect("read blocks dir");
        while let Some(entry) = rd.next_entry().await.expect("next entry") {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name != "metadata" {
                block_files.push(name);
            }
        }

        assert_eq!(
            block_files.len() as u64,
            expected_blocks,
            "expected {expected_blocks} block files on disk for a freshly created dense tensor, found {:?}",
            block_files
        );

        cleanup(&root).await;
    }

    // A freshly created (unwritten) sparse tensor must not materialize any
    // block files — only the schema `metadata` file should exist under
    // `blocks/` until a nonzero write forces a block into existence.
    #[tokio::test]
    async fn sparse_create_has_no_blocks_only_metadata_on_disk() {
        let root = common::unique_tmp_dir("h_sparse_no_blocks");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = sparse_schema_f32(shape![2, 3, 4]);

        let dir = open_dir(&root).expect("open");
        let _tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            schema,
            Layout::Sparse { axis: Some(1) },
            4,
        )
        .await
        .expect("create sparse");
        dir.sync().await.expect("sync");

        let mut entries = Vec::new();
        let mut rd = tokio::fs::read_dir(root.join("blocks"))
            .await
            .expect("read blocks dir");
        while let Some(entry) = rd.next_entry().await.expect("next entry") {
            entries.push(entry.file_name().to_string_lossy().into_owned());
        }

        assert_eq!(
            entries,
            vec!["metadata".to_string()],
            "a freshly created sparse tensor must have no block files, only the metadata file"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn unknown_metadata_version_rejected_on_reload() {
        let root = common::unique_tmp_dir("h_meta_version");
        tokio::fs::create_dir(&root).await.expect("mkdir");
        let schema = dense_schema_f32(shape![2, 3, 4]);

        {
            let dir = open_dir(&root).expect("open");
            let _ = Tensor::<FsEntry, f32>::create(dir.clone(), schema.clone(), Layout::Dense, 4)
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
