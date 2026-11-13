use fensor::{
    AxisRange, Error, Layout, Tensor, TensorRead, TensorSchema, TensorWrite, TensorWriteBulk,
};
use ha_ndarray::{range, shape};
use number_general::DType;

mod common;
use common::{FsEntry, new_dir};

#[tokio::test]
async fn writes_validate_cardinality_before_mutation() {
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (_dir_root, dir) = new_dir("bounded_writes").await;
        let tensor = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![6]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();

        tensor
            .write_values(range![AxisRange::In(0, 6, 1)], vec![1, 2, 3, 4, 5, 6])
            .await
            .unwrap();

        for values in [vec![9], vec![9; 7]] {
            assert!(matches!(
                tensor
                    .write_values(range![AxisRange::In(0, 6, 1)], values)
                    .await,
                Err(Error::DataMismatch(_))
            ));

            for i in 0..6 {
                assert_eq!(tensor.read_value(&[i]).await.unwrap(), i as u8 + 1);
            }
        }

        tensor
            .write_values(range![AxisRange::Of(vec![5, 1, 5])], vec![3, 4, 7])
            .await
            .unwrap();

        assert_eq!(tensor.read_value(&[5]).await.unwrap(), 7);
        assert_eq!(tensor.read_value(&[1]).await.unwrap(), 4);
        tensor
            .write_values(range![AxisRange::In(0, 6, 2)], vec![8, 9, 10])
            .await
            .unwrap();

        assert_eq!(tensor.read_value(&[4]).await.unwrap(), 10);
    }

    for (layout, expected_blocks) in [
        (Layout::Dense, 4),
        (Layout::Sparse { axis: Some(0) }, 6),
        (Layout::Sparse { axis: None }, 15),
    ] {
        use fensor::TensorSource;
        use futures::TryStreamExt;
        let (_root, dir) = new_dir("compact_regions").await;
        let schema = TensorSchema::new(u8::dtype(), shape![3, 5]).unwrap();
        let tensor = Tensor::<FsEntry, u8>::create(dir.clone(), schema.clone(), layout, 4)
            .await
            .unwrap();

        tensor
            .write_values(
                range![AxisRange::In(0, 3, 1), AxisRange::In(0, 5, 1)],
                (1..=15).collect(),
            )
            .await
            .unwrap();

        let geometry = tensor.storage_geometry();
        assert_eq!(geometry.block_count(), expected_blocks);
        let mut stored = 0;
        for id in 0..geometry.block_count() {
            let block = tensor.read_logical_block(id).await.unwrap();
            assert_eq!(block.len(), geometry.block_len(id).unwrap());
            assert_eq!(
                block.len() as u64,
                geometry.block_range(id).unwrap().end - geometry.block_range(id).unwrap().start
            );

            stored += block.len();
            assert!(
                tensor
                    .replace_logical_block(id, vec![99; block.len() + 1])
                    .await
                    .is_err()
            );

            assert_eq!(tensor.read_logical_block(id).await.unwrap(), block);
        }

        assert_eq!(stored, 15);
        assert_eq!(
            tensor.read_blocks().unwrap().try_concat().await.unwrap(),
            (1..=15).collect::<Vec<_>>()
        );

        tensor.sync_all().await.unwrap();
        drop(tensor);
        let tensor = Tensor::<FsEntry, u8>::load(dir).await.unwrap();
        assert_eq!(tensor.read_value(&[2, 4]).await.unwrap(), 15);
        assert_eq!(tensor.storage_geometry(), geometry);

        let (_workspace, workspace) = new_dir("compact_delta_blocks").await;
        let blocks = fensor::SparseBlocks::<FsEntry, u8>::create(workspace, geometry.clone())
            .await
            .unwrap();

        let last = geometry.block_count() - 1;
        assert_eq!(
            blocks.read(last).await.unwrap(),
            vec![0; geometry.block_len(last).unwrap()]
        );

        blocks
            .replace(last, vec![7; geometry.block_len(last).unwrap()])
            .await
            .unwrap();

        assert_eq!(
            blocks
                .occupied_blocks(0..geometry.block_count())
                .try_collect::<Vec<_>>()
                .await
                .unwrap(),
            [last]
        );

        let retained = blocks.clone();
        drop(blocks);
        retained
            .replace(last, vec![0; geometry.block_len(last).unwrap()])
            .await
            .unwrap();

        assert!(
            retained
                .occupied_blocks(0..geometry.block_count())
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
    }

    let (_root, workspace) = new_dir("block_dtype_rejection").await;
    let wrong = fensor::StorageGeometry::new(
        TensorSchema::new(f64::dtype(), shape![3, 5]).unwrap(),
        Layout::Dense,
        4,
    )
    .unwrap();

    assert!(matches!(
        fensor::SparseBlocks::<FsEntry, u8>::create(workspace.clone(), wrong).await,
        Err(Error::InvalidSchema(_))
    ));

    assert!(workspace.read().await.is_empty());
}

#[tokio::test]
async fn huge_mismatched_write_does_not_expand_coordinates() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (_dir_root, dir) = new_dir("huge_bounded_write").await;
        let tensor = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![1_000_000_000]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();

        tensor.write_value(&[0], 127).await.unwrap();
        assert!(matches!(
            tensor
                .write_values(range![AxisRange::In(0, 1_000_000_000, 1)], vec![1])
                .await,
            Err(Error::DataMismatch(_))
        ));

        assert_eq!(tensor.read_value(&[0]).await.unwrap(), 127);
    })
    .await
    .expect("length validation must not traverse or allocate the full range");
}
