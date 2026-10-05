//! Regression fixtures which deliberately inspect or corrupt private native storage.

mod matrix_unary {
    use crate::test_support::cleanup;
    use crate::test_support::{FsEntry, new_dir};
    use crate::{
        AxisRange, Error, Layout, Tensor, TensorMatrixUnary, TensorRead, TensorSchema, TensorWrite,
    };
    use futures::TryStreamExt;
    use number_general::DType;
    use smallvec::smallvec;
    #[tokio::test]
    async fn huge_selected_diagonal_and_corrupt_blocks() {
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let n = 1_000_000_000u64;
            let (root, dir) = new_dir("huge_diag").await;
            let tensor = Tensor::<FsEntry, u8>::create(
                dir.clone(),
                TensorSchema::new(u8::dtype(), smallvec![n, n]).unwrap(),
                Layout::Sparse { axis: None },
                1,
            )
            .await
            .unwrap();
            tensor.write_value(&[n - 1, n - 1], 255).await.unwrap();
            tensor.write_value(&[0, 1], 127).await.unwrap();
            tensor.corrupt_sparse_payload(1).await;
            let diag = tensor.view().diag().await.unwrap();
            assert_eq!(diag.read_value(&[0]).await.unwrap(), 0);
            let entries: Vec<_> = diag
                .read_sparse_elements_in_order(smallvec![AxisRange::In(n - 2, n, 1)], smallvec![0])
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(entries, vec![(vec![n - 1], 255)]);
            assert!(diag.read_value(&[n]).await.is_err());
            assert!(
                diag.read_sparse_elements_in_order(smallvec![AxisRange::At(n)], smallvec![0])
                    .await
                    .is_err()
            );
            assert!(
                diag.read_sparse_elements_in_order(smallvec![AxisRange::At(0)], smallvec![1])
                    .await
                    .is_err()
            );
            tensor.corrupt_sparse_payload(n * n - 1).await;
            assert!(matches!(
                diag.read_value(&[n - 1]).await,
                Err(Error::InvalidLayout(_))
            ));

            assert!(diag.read_value(&[n - 1]).await.is_err());
            cleanup(&root).await;
        })
        .await
        .unwrap();
    }
}

mod math_unary {
    use crate::test_support::{self as common, FsEntry, new_dir};
    use crate::{AxisRange, Layout, Tensor, TensorRead, TensorSchema, TensorUnary, TensorWrite};
    use futures::TryStreamExt;
    use ha_ndarray::{axes, range, shape};
    use number_general::{FloatType, NumberType};
    #[tokio::test]
    async fn sparse_range_reads_only_selected_storage() {
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
        tensor.corrupt_sparse_payload(7).await;
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
}

mod storage_source {
    use crate::test_support::cleanup;
    use crate::test_support::{FsEntry, new_dir};
    use crate::{Layout, NumberType, Tensor, TensorRead, TensorSchema, TensorWrite};
    use number_general::FloatType;
    #[tokio::test]
    async fn replacement_preserves_shared_sparse_pages_and_zero_deletions() {
        let (root, dir) = new_dir("logical_sparse").await;
        let schema =
            TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 2].into()).unwrap();
        let tensor =
            Tensor::<FsEntry, f64>::create(dir.clone(), schema, Layout::Sparse { axis: None }, 2)
                .await
                .unwrap();
        tensor.write_value(&[0, 0], 2.0).await.unwrap();
        assert_eq!(
            tensor
                .storage
                .sparse()
                .unwrap()
                .occupied_marker(&[0, 0])
                .await
                .unwrap(),
            Some(0)
        );
        tensor.write_value(&[1, 0], 2.0).await.unwrap();
        tensor
            .replace_logical_block(0, vec![9.0, 0.0])
            .await
            .unwrap();
        assert_eq!(tensor.read_value(&[0, 0]).await.unwrap(), 9.0);
        assert_eq!(tensor.read_value(&[1, 0]).await.unwrap(), 2.0);
        tensor.validate().await.unwrap();
        tensor.replace_logical_block(0, vec![0.0; 2]).await.unwrap();
        assert!(
            tensor
                .storage
                .sparse()
                .unwrap()
                .occupied_marker(&[0, 0])
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(tensor.read_value(&[1, 0]).await.unwrap(), 2.0);
        tensor.sync().await.unwrap();
        let reopened = Tensor::<FsEntry, f64>::load(dir).await.unwrap();
        assert_eq!(reopened.read_value(&[1, 0]).await.unwrap(), 2.0);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn strict_load_rejects_out_of_bounds_and_duplicate_sparse_keys() {
        for invalid_key in [true, false] {
            let (root, dir) = new_dir("strict_sparse_rows").await;
            let schema =
                TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 2].into()).unwrap();
            let tensor = Tensor::<FsEntry, f64>::create(
                dir.clone(),
                schema,
                Layout::Sparse { axis: None },
                2,
            )
            .await
            .unwrap();
            tensor.write_value(&[0, 0], 1.0).await.unwrap();
            tensor.sync().await.unwrap();
            let index_dir = dir.read().await.get_dir("index").cloned().unwrap();
            let index = b_table::TableLock::load(
                crate::schema::SparseTableSchema::default(),
                b_table::collate::Collator::<u64>::default(),
                index_dir,
            )
            .unwrap();
            {
                let mut index = index.write().await;
                let key = if invalid_key { vec![10, 0] } else { vec![0, 0] };
                index.upsert(key, vec![12]).await.unwrap();
            }
            index.sync().await.unwrap();
            assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
            cleanup(&root).await;
        }
    }
}

mod matmul {
    use crate::test_support::{self as common, FsEntry, new_dir};
    use crate::{
        AxisRange, Layout, Tensor, TensorMatMul, TensorRead, TensorSchema, TensorTransform,
        TensorWrite,
    };
    use futures::TryStreamExt;
    use ha_ndarray::{axes, range, shape};
    use number_general::DType;
    #[tokio::test]
    async fn huge_selected_outputs_and_corruption_boundaries() {
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let (_dir_root, dir) = new_dir("huge_matmul").await;
            let a = Tensor::<FsEntry, f32>::create(
                dir.clone(),
                TensorSchema::new(f32::dtype(), shape![1_000_000_000, 2]).unwrap(),
                Layout::Sparse { axis: None },
                2,
            )
            .await
            .unwrap();
            a.write_value(&[999_999_999, 1], 2.).await.unwrap();
            a.write_value(&[0, 0], 1.).await.unwrap();
            a.corrupt_sparse_payload(0).await;
            let (_b_root, b) = common::fixture::source(
                "right_operand",
                shape![2, 2],
                Layout::Sparse { axis: None },
                2,
                1_000_000,
                [1f32, 2., 3., 4.],
            )
            .await;
            let view = a.view().matmul(&b.view()).await.unwrap();
            let entries: Vec<_> = view
                .read_sparse_elements_in_order(
                    range![AxisRange::At(999_999_999), AxisRange::In(0, 2, 1)],
                    axes![0, 1],
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(
                entries,
                vec![(vec![999_999_999, 0], 6.), (vec![999_999_999, 1], 8.)]
            );
            assert!(view.read_value(&[0, 0]).await.is_err());
            assert!(
                view.read_sparse_elements_in_order(
                    range![AxisRange::At(1_000_000_000), AxisRange::At(0)],
                    axes![0, 1]
                )
                .await
                .is_err()
            );
            // The same corrupt source on the right is only read for selected columns.
            let right = a.view().transpose(None).unwrap();
            let left = b.view().transpose(None).unwrap();
            let reversed = left.matmul(&right).await.unwrap();
            assert_eq!(reversed.read_value(&[0, 999_999_999]).await.unwrap(), 6.);
            assert!(reversed.read_value(&[0, 0]).await.is_err());
        })
        .await
        .expect("selected product must not traverse unrelated output tiles");
    }
}

mod math_binary {
    use crate::test_support::{self as common, FsEntry, new_dir};
    use crate::{
        AxisRange, Layout, Tensor, TensorBooleanScalar, TensorMath, TensorRead, TensorSchema,
        TensorUnary, TensorWhere, TensorWrite,
    };
    use futures::TryStreamExt;
    use ha_ndarray::{axes, range, shape};
    use number_general::DType;
    #[tokio::test]
    async fn selected_range_propagates_corruption_only_when_read() {
        let (_dir_root, dir) = new_dir("binary_corruption").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(f32::dtype(), shape![8]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&[0], 0.2).await.unwrap();
        tensor.write_value(&[7], 0.2).await.unwrap();
        tensor.corrupt_sparse_payload(7).await;
        let (_empty_root, empty) = common::fixture::source(
            "empty_operand",
            shape![8],
            Layout::Sparse { axis: None },
            3,
            1_000_000,
            [0f32; 8],
        )
        .await;

        for reversed in [false, true] {
            let (left, right) = if reversed {
                (&empty, &tensor)
            } else {
                (&tensor, &empty)
            };

            let expression = left
                .view()
                .add(&right.view())
                .await
                .unwrap()
                .exp()
                .await
                .unwrap();
            let values: Vec<_> = expression
                .read_sparse_elements_in_order(range![AxisRange::At(0)], axes![0])
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();

            assert_eq!(values, vec![(vec![0], 0.2f32.exp())]);
            let mut stream = expression
                .read_sparse_elements_in_order(range![AxisRange::At(7)], axes![0])
                .await
                .unwrap();

            assert!(stream.try_next().await.is_err());
            assert!(expression.read_value(&[7]).await.is_err());
            assert!(
                expression
                    .read_blocks()
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn conditional_selected_range_propagates_each_operand_error() {
        let (_dir_root, dir) = new_dir("conditional_corruption").await;
        let corrupt = Tensor::<FsEntry, u8>::create(
            dir.clone(),
            TensorSchema::new(u8::dtype(), shape![8]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();
        corrupt.write_value(&[0], 1).await.unwrap();
        corrupt.write_value(&[7], 1).await.unwrap();
        corrupt.corrupt_sparse_payload(7).await;
        let (_empty_root, empty) = common::fixture::source(
            "empty_operand",
            shape![8],
            Layout::Sparse { axis: None },
            3,
            1_000_000,
            [0u8; 8],
        )
        .await;

        for position in 0..3 {
            let operands =
                std::array::from_fn::<_, 3, _>(|i| if i == position { &corrupt } else { &empty });
            let expression = operands[0]
                .view()
                .cond(&operands[1].view(), &operands[2].view())
                .await
                .unwrap()
                .or_scalar(1)
                .await
                .unwrap();
            let values: Vec<_> = expression
                .read_sparse_elements_in_order(range![AxisRange::At(0)], axes![0])
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(values, vec![(vec![0], 1)]);
            assert!(expression.read_value(&[7]).await.is_err());
            let mut stream = expression
                .read_sparse_elements_in_order(range![AxisRange::At(7)], axes![0])
                .await
                .unwrap();
            assert!(stream.try_next().await.is_err());
            assert!(
                expression
                    .read_blocks()
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .is_err()
            );
        }
    }
}

mod reduce {
    use crate::test_support::{FsEntry, new_dir};
    use crate::{
        AxisRange, Layout, Tensor, TensorRead, TensorReduce, TensorReduceAll, TensorReduceBoolean,
        TensorSchema, TensorUnary, TensorWrite,
    };
    use futures::TryStreamExt;
    use ha_ndarray::{axes, range, shape};
    use number_general::DType;
    #[tokio::test]
    async fn selected_output_range_and_corruption_boundaries() {
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let (_dir_root, dir) = new_dir("reduce_far_range").await;
            let tensor = Tensor::<FsEntry, f32>::create(
                dir.clone(),
                TensorSchema::new(f32::dtype(), shape![500_000_000, 2]).unwrap(),
                Layout::Sparse { axis: None },
                2,
            )
            .await
            .unwrap();
            tensor.write_value(&[499_999_999, 0], 2.).await.unwrap();
            tensor.write_value(&[0, 0], 1.).await.unwrap();
            tensor.corrupt_sparse_payload(0).await;
            let view = tensor.view().sum(axes![1], false).await.unwrap();
            let entries: Vec<_> = view
                .read_sparse_elements_in_order(
                    range![AxisRange::Of(vec![499_999_999, 499_999_998, 499_999_999])],
                    axes![0],
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(entries, vec![(vec![499_999_999], 2.)]);
            assert!(view.read_value(&[0]).await.is_err());
            assert!(
                view.read_sparse_elements_in_order(range![AxisRange::At(500_000_000)], axes![0])
                    .await
                    .is_err()
            );
            assert!(
                view.read_sparse_elements_in_order(range![AxisRange::At(1)], axes![1])
                    .await
                    .is_err()
            );
        })
        .await
        .expect("selected reduction must not enumerate unrelated outputs");
    }

    #[tokio::test]
    async fn boolean_short_circuit_and_numeric_error_propagation() {
        let (_dir_root, dir) = new_dir("reduce_corrupt_boolean").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(f32::dtype(), shape![8193]).unwrap(),
            Layout::Sparse { axis: None },
            1,
        )
        .await
        .unwrap();
        tensor.write_value(&[0], 0.2).await.unwrap();
        tensor.write_value(&[8192], 1.).await.unwrap();
        tensor.corrupt_sparse_payload(8192).await;
        assert!(tensor.any().await.unwrap());
        assert!(!tensor.view().round().await.unwrap().all().await.unwrap());
        assert!(tensor.all().await.is_err());
        assert!(tensor.view().round().await.unwrap().any().await.is_err());
        assert!(tensor.sum_all().await.is_err());
    }
}

mod bulk_mutation {
    use super::super::copy_metrics::CURRENT;
    use crate::test_support::{FsEntry, cleanup, new_dir};
    use crate::{
        AxisRange, Layout, StorageGeometry, Tensor, TensorRead, TensorSchema, TensorSource,
        TensorWrite, TensorWriteBulk,
    };

    #[tokio::test]
    async fn bulk_writes_publish_blocks_and_preserve_padding_and_duplicates() {
        for layout in [Layout::Dense, Layout::Sparse { axis: Some(1) }] {
            let (root, dir) = new_dir("bulk_blocks").await;
            let geometry = StorageGeometry::new(
                TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![9, 129].into())
                    .unwrap(),
                layout,
                vec![4, 64].into(),
            )
            .unwrap();
            let tensor = Tensor::<FsEntry, f64>::create_with_geometry(dir, geometry.clone())
                .await
                .unwrap();
            CURRENT
                .scope(Default::default(), async {
                    for value in [2., 0., f64::NAN] {
                        let before = CURRENT.with(|m| m.borrow().block_updates);
                        tensor.fill(value).await.unwrap();
                        assert_eq!(
                            CURRENT.with(|m| m.borrow().block_updates) - before,
                            geometry.block_count() as usize
                        );
                        for id in 0..geometry.block_count() {
                            let block = tensor.read_logical_block(id).await.unwrap();
                            let mut valid = vec![false; geometry.block_len()];
                            let expected: Vec<_> =
                                crate::test_support::block_entries(&geometry, id).collect();
                            let offsets: Vec<_> = geometry.block_offsets(id).unwrap().collect();
                            assert_eq!(
                                offsets,
                                expected.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                            );
                            for offset in offsets {
                                valid[offset] = true;
                                assert!(
                                    block[offset] == value
                                        || block[offset].is_nan() && value.is_nan()
                                );
                            }
                            assert!(
                                block
                                    .iter()
                                    .zip(valid)
                                    .all(|(value, valid)| valid || *value == 0.)
                            );
                        }
                    }
                    let range = vec![
                        AxisRange::Of(vec![8, 0, 8]),
                        AxisRange::Of(vec![128, 1, 128]),
                    ]
                    .into();
                    let before = CURRENT.with(|m| m.borrow().block_updates);
                    tensor
                        .write_values(range, (1..=9).map(f64::from).collect())
                        .await
                        .unwrap();
                    assert_eq!(CURRENT.with(|m| m.borrow().block_updates) - before, 4);
                    assert_eq!(tensor.read_value(&[8, 128]).await.unwrap(), 9.);
                    assert_eq!(tensor.read_value(&[8, 1]).await.unwrap(), 8.);
                    assert_eq!(tensor.read_value(&[0, 128]).await.unwrap(), 6.);
                    assert!(
                        tensor
                            .write_values(vec![AxisRange::At(0), AxisRange::At(0)].into(), vec![])
                            .await
                            .is_err()
                    );
                })
                .await;
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn dense_mutation_never_recreates_missing_blocks() {
        let (root, dir) = new_dir("missing_dense_mutation").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![6].into()).unwrap(),
            Layout::Dense,
            2,
        )
        .await
        .unwrap();
        tensor.storage.blocks().write().await.delete("0").await;
        assert!(tensor.write_value(&[0], 1.).await.is_err());
        assert!(
            tensor
                .write_values(vec![AxisRange::At(0)].into(), vec![1.])
                .await
                .is_err()
        );
        assert!(tensor.fill(1.).await.is_err());
        assert!(tensor.storage.blocks().read().await.get_file("0").is_none());
        tensor
            .write_values(vec![AxisRange::In(4, 6, 1)].into(), vec![7., 8.])
            .await
            .unwrap();
        assert_eq!(tensor.read_value(&[5]).await.unwrap(), 8.);
        tensor
            .write_values(vec![AxisRange::Of(vec![])].into(), vec![])
            .await
            .unwrap();
        cleanup(&root).await;
    }
}
