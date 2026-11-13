use futures::FutureExt;
use number_general::DType;

use crate::test_support::{FsEntry, cleanup, new_dir};
use crate::{Error, Layout, StorageGeometry, Tensor, TensorSchema, TensorSource};

#[tokio::test]
async fn replacement_validation_preserves_lock_precedence() {
    use b_tree::Schema;

    use crate::sparse::{PayloadSchema, SparseCell};

    let schema = PayloadSchema::<f32>::new(
        StorageGeometry::new(
            TensorSchema::new(f32::dtype(), vec![8].into()).unwrap(),
            Layout::Dense,
            2,
        )
        .unwrap(),
    );

    for prefix in [
        vec![SparseCell::Key(0)],
        vec![SparseCell::Key(0), SparseCell::Payload(vec![1., 2.])],
    ] {
        schema.validate_prefix(&prefix).unwrap();
    }

    for prefix in [
        vec![],
        vec![SparseCell::Payload(vec![1., 2.])],
        vec![SparseCell::Key(0), SparseCell::Key(1)],
        vec![SparseCell::Key(0), SparseCell::Payload(vec![1.])],
        vec![
            SparseCell::Key(0),
            SparseCell::Payload(vec![1., 2.]),
            SparseCell::Key(1),
        ],
    ] {
        assert!(schema.validate_prefix(&prefix).is_err());
    }

    assert!(schema.validate_key(vec![SparseCell::Key(0)]).is_err());

    let (root, dir) = new_dir("sparse_validation_order").await;
    let geometry = StorageGeometry::new(
        TensorSchema::new(f32::dtype(), vec![2, 5].into()).unwrap(),
        Layout::Sparse { axis: Some(0) },
        [1, 4].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let tensor = Tensor::<FsEntry, f32>::create_with_geometry(dir, geometry.clone())
        .await
        .unwrap();

    let storage = tensor.storage.sparse().unwrap();
    let guard = storage.inner.blocks.write().await;

    assert!(matches!(
        storage
            .replace(geometry.block_count(), Vec::new())
            .now_or_never(),
        Some(Err(Error::InvalidCoord(_)))
    ));

    assert!(matches!(
        storage.replace(1, Vec::new()).now_or_never(),
        Some(Err(Error::InvalidLayout(message)))
            if message == "invalid replacement block length"
    ));

    assert!(matches!(
        storage.read(geometry.block_count()).now_or_never(),
        Some(Err(Error::InvalidCoord(_)))
    ));

    assert!(matches!(
        storage.update(geometry.block_count(), &[]).now_or_never(),
        Some(Err(Error::InvalidCoord(_)))
    ));

    assert!(storage.replace(1, vec![1.; 1]).now_or_never().is_none());
    assert!(matches!(
        storage.replace(1, vec![1.; 4]).now_or_never(),
        Some(Err(Error::InvalidLayout(_)))
    ));

    let count = geometry.block_count();
    for direct in [false, true] {
        for (ids, oversized) in [
            (vec![count], false),
            (vec![0, count], false),
            (vec![1, 1], false),
            (vec![1, 0], false),
            (vec![count; crate::MAX_BATCH_ELEMENTS + 1], true),
        ] {
            let mut calls = 0;
            let callback = |_: u64, _: &[f32]| {
                calls += 1;
                Ok(())
            };

            let result = if direct {
                storage.visit_blocks(&ids, callback).now_or_never()
            } else {
                tensor.visit_blocks(&ids, callback).now_or_never()
            }
            .expect("invalid IDs must not wait for the native guard");

            assert!(if oversized {
                matches!(result, Err(Error::InvalidLayout(_)))
            } else {
                matches!(result, Err(Error::InvalidCoord(_)))
            });

            assert_eq!(calls, 0);
        }

        let mut calls = 0;
        let callback = |_: u64, _: &[f32]| {
            calls += 1;
            Ok(())
        };

        let result = if direct {
            storage.visit_blocks(&[], callback).now_or_never()
        } else {
            tensor.visit_blocks(&[], callback).now_or_never()
        };

        assert!(matches!(result, Some(Ok(()))));
        assert_eq!(calls, 0);
        assert!(
            storage
                .visit_blocks(&[0], |_, _| Ok(()))
                .now_or_never()
                .is_none()
        );
    }

    drop(guard);
    storage.replace(1, vec![1.; 1]).await.unwrap();

    for direct in [false, true] {
        let mut seen = Vec::new();
        let callback = |id, values: &[f32]| {
            seen.push((id, values.to_vec()));
            Ok(())
        };

        if direct {
            storage.visit_blocks(&[0, 1, 2, 3], callback).await.unwrap();
        } else {
            tensor.visit_blocks(&[0, 1, 2, 3], callback).await.unwrap();
        }

        assert_eq!(seen, [(1, vec![1.])]);

        let callback = |_, _: &[f32]| Err(Error::Io(std::io::Error::other("fixture callback")));
        let result = if direct {
            storage.visit_blocks(&[0, 1, 2], callback).await
        } else {
            tensor.visit_blocks(&[0, 1, 2], callback).await
        };

        assert!(matches!(result, Err(Error::Io(error))
            if error.to_string() == "fixture callback" && !freqfs::IntegrityError::is_marked(&error)));

        assert!(storage.inner.blocks.try_write().is_ok());
    }

    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let request = crate::request::BatchRequest::linear(0, 1).unwrap();
            assert_eq!(tensor.read_batch(&request, None).await.unwrap(), vec![0.]);
            crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
        })
        .await;

    assert!(storage.inner.blocks.try_write().is_ok());
    assert_eq!(tensor.read_logical_block(1).await.unwrap(), vec![1.; 1]);
    cleanup(&root).await;
}

#[tokio::test]
async fn index_guards_preserve_batch_reads_and_partial_updates() {
    use crate::request::BatchRequest;
    use crate::sparse::SparseCell;

    let (root, dir) = new_dir("sparse_table_guards").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(f32::dtype(), vec![4, 2].into()).unwrap(),
        Layout::Sparse { axis: Some(0) },
        2,
    )
    .await
    .unwrap();

    let storage = tensor.storage.sparse().unwrap();
    for id in 0..4 {
        storage.replace(id, vec![1.; 2]).await.unwrap();
    }

    // Stall the first range at its payload, after the batch acquired its tree guard.
    let values = dir.read().await.get_dir("values").cloned().unwrap();
    let primary = values.read().await.get_dir("primary").cloned().unwrap();
    let file = primary.read().await.files().next().unwrap().clone();
    let held = file.write::<crate::SparseNode<f32>>().await.unwrap();
    let request = BatchRequest::explicit(vec![vec![0, 0], vec![1, 0], vec![3, 0]]).unwrap();
    {
        let mut read = std::pin::pin!(tensor.read_batch(&request, None));
        assert!(futures::poll!(read.as_mut()).is_pending());
        let mut write = std::pin::pin!(storage.update(3, &[(0, 2.)]));
        assert!(futures::poll!(write.as_mut()).is_pending());
        drop(held);

        // A queued writer must not interrupt the batch between its two ID groups.
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), read)
                .await
                .unwrap()
                .unwrap(),
            vec![1.; 3]
        );

        write.await.unwrap();
    }

    let held = file.write::<crate::SparseNode<f32>>().await.unwrap();
    {
        let mut left = std::pin::pin!(storage.update(0, &[(0, 3.)]));
        let mut right = std::pin::pin!(storage.update(0, &[(1, 4.)]));
        assert!(futures::poll!(left.as_mut()).is_pending());
        assert!(storage.inner.blocks.try_read().is_err());
        assert!(futures::poll!(right.as_mut()).is_pending());
        drop(held);

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (left, right, read, sync) =
                tokio::join!(left, right, storage.read(3), tensor.sync_all(),);

            left.unwrap();
            right.unwrap();
            assert_eq!(read.unwrap(), vec![2., 1.]);
            sync.unwrap();
            assert_eq!(storage.read(0).await.unwrap(), vec![3., 4.]);
        })
        .await
        .unwrap();
    }

    // Invalid replacement is nonmutating, and guards release after failures.
    assert!(storage.replace(0, vec![5.]).await.is_err());
    assert_eq!(storage.read(0).await.unwrap(), vec![3., 4.]);
    assert_eq!(
        storage
            .inner
            .blocks
            .read()
            .await
            .get(&[SparseCell::Key(3)])
            .await
            .unwrap()
            .unwrap()[1],
        SparseCell::Payload(vec![2., 1.])
    );

    tensor.sync_all().await.unwrap();
    drop(tensor);
    let reopened = Tensor::<FsEntry, f32>::load(dir).await.unwrap();
    assert_eq!(reopened.read_logical_block(0).await.unwrap(), vec![3., 4.]);
    drop(reopened);
    cleanup(&root).await;
}
