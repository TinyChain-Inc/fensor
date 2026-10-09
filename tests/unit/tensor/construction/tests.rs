use futures::{FutureExt, StreamExt};
use number_general::DType;

use super::*;
use crate::tensor::copy_metrics;
use crate::test_support::{FsEntry, cleanup, new_dir};
use crate::{Layout, TensorSchema};

#[derive(Debug)]
enum InputError {
    Native(Error),
    Input,
}

impl From<Error> for InputError {
    fn from(error: Error) -> Self {
        Self::Native(error)
    }
}

#[tokio::test]
async fn invalid_or_interrupted_inputs_withhold_metadata() {
    // Sparse bounds, ordering, uniqueness, empty support, and stream errors
    // exercise the same constructor, with the original error type retained.
    for entries in [
        vec![Ok((vec![4], 1.))],
        vec![Ok((vec![0, 0], 1.))],
        vec![Ok((vec![1], 1.)), Ok((vec![0], 1.))],
        vec![Ok((vec![1], 0.)), Ok((vec![1], 1.))],
        vec![Ok((vec![0], 1.)), Err(InputError::Input)],
    ] {
        let (root, dir) = new_dir("construction_invalid").await;
        let result = Tensor::<FsEntry, f64>::from_sparse_elements(
            dir.clone(),
            TensorSchema::new(f64::dtype(), vec![4].into()).unwrap(),
            Layout::Sparse { axis: None },
            2,
            futures::stream::iter(entries),
        )
        .await;
        match result.err().unwrap() {
            InputError::Native(error) => assert!(matches!(error, Error::InvalidCoord(_))),
            InputError::Input => {}
        }
        assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
        cleanup(&root).await;
    }

    for values in [
        vec![],
        vec![Ok(1.)],
        vec![Ok(1.), Ok(2.), Ok(3.)],
        vec![Ok(1.), Err(InputError::Input)],
    ] {
        let (root, dir) = new_dir("construction_cardinality").await;
        let result = Tensor::<FsEntry, f64>::from_values(
            dir.clone(),
            TensorSchema::new(f64::dtype(), vec![2].into()).unwrap(),
            1,
            futures::stream::iter(values),
        )
        .await;
        assert!(result.is_err());
        assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
        cleanup(&root).await;
    }

    for sparse in [false, true] {
        let (root, dir) = new_dir("construction_cancel_input").await;
        let schema = TensorSchema::new(f64::dtype(), vec![8192].into()).unwrap();
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        struct DropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let flag = DropFlag(dropped.clone());
        let entries = futures::stream::iter(0..4096)
            .map(|i| Ok::<_, Error>((vec![i], 1.)))
            .chain(futures::stream::once(async move {
                let _flag = flag;
                futures::future::pending().await
            }));
        let mut future: BoxFuture<'_, Result<Tensor<FsEntry, f64>>> = if sparse {
            Box::pin(Tensor::from_sparse_elements(
                dir.clone(),
                schema,
                Layout::Sparse { axis: None },
                4096,
                entries,
            ))
        } else {
            Box::pin(Tensor::from_values(
                dir.clone(),
                schema,
                4096,
                entries.map_ok(|(_, value)| value),
            ))
        };
        assert!(future.as_mut().now_or_never().is_none());
        drop(future);
        assert!(dropped.load(std::sync::atomic::Ordering::Relaxed));
        assert!(Tensor::<FsEntry, f64>::load(dir.clone()).await.is_err());
        assert!(dir.try_write().is_ok());
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn sparse_stream_uses_bounded_single_pass_construction() {
    for (shape, axis, full) in [
        (vec![17, 257], Some(1), true),
        (vec![17, 257], Some(1), false),
        (vec![2, 127], None, true),
        (vec![2, 128], None, true),
    ] {
        let (root, dir) = new_dir("constructor_encodings").await;
        let count = std::sync::atomic::AtomicU64::new(0);
        let entries =
            futures::stream::iter(crate::row_major_coords(&shape).unwrap().map(|coord| {
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let value = if full { 1. } else { 0. };
                Ok::<_, Error>((coord, value))
            }));
        let tensor = crate::read_metrics::CURRENT
            .scope(
                Default::default(),
                copy_metrics::CURRENT.scope(Default::default(), async {
                    let tensor = Tensor::<FsEntry, f64>::from_sparse_elements(
                        dir,
                        TensorSchema::new(f64::dtype(), shape.clone().into()).unwrap(),
                        Layout::Sparse { axis },
                        4096,
                        entries,
                    )
                    .await
                    .unwrap();
                    copy_metrics::CURRENT.with(|m| {
                        let m = m.borrow();
                        assert_eq!(
                            m.block_updates, 0,
                            "construction never invokes ordinary replacement"
                        );
                        assert_eq!(m.replaced_blocks, 0);
                        assert_eq!(m.payload_writes, m.constructed_blocks);
                        assert!(m.max_staging_batch <= crate::expression::MAX_BATCH_ELEMENTS);
                        assert_eq!(
                            m.constructed_blocks as u64,
                            if full {
                                tensor.storage_geometry().block_count()
                            } else {
                                0
                            }
                        );
                    });
                    tensor
                }),
            )
            .await;
        assert_eq!(
            count.load(std::sync::atomic::Ordering::Relaxed),
            shape.iter().product::<u64>()
        );
        tensor.sync().await.unwrap();
        tensor.validate().await.unwrap();
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn dense_replacement_validates_required_payload_without_repair() {
    for malformed in [0, 1, 2] {
        let (root, dir) = new_dir("replace_required").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), vec![2].into()).unwrap(),
            Layout::Dense,
            2,
        )
        .await
        .unwrap();
        let blocks = tensor.storage.blocks();
        {
            let held = blocks.write().await;
            assert!(matches!(
                tensor.replace_logical_block(1, vec![]).now_or_never(),
                Some(Err(Error::InvalidCoord(_)))
            ));
            assert!(matches!(
                tensor.replace_logical_block(0, vec![]).now_or_never(),
                Some(Err(Error::InvalidLayout(_)))
            ));
            drop(held);
            assert_eq!(tensor.read_logical_block(0).await.unwrap(), vec![0.; 2]);
        }
        blocks.write().await.delete("0").await;
        match malformed {
            1 => {
                blocks
                    .write()
                    .await
                    .create_file(
                        "0".into(),
                        vec![1u8, 2],
                        std::mem::size_of::<FsEntry>() + std::mem::size_of::<Vec<u64>>() + 2,
                    )
                    .await
                    .unwrap();
            }
            2 => {
                blocks
                    .write()
                    .await
                    .create_file(
                        "0".into(),
                        vec![1f64],
                        std::mem::size_of::<FsEntry>() + std::mem::size_of::<Vec<u64>>() + 8,
                    )
                    .await
                    .unwrap();
            }
            _ => {}
        }
        assert!(tensor.replace_logical_block(0, vec![3.; 2]).await.is_err());
        use crate::TensorWrite;
        let point = tensor.write_value(&[0], 3.).await;
        assert!(point.is_err());
        if malformed == 0 {
            assert!(matches!(point, Err(Error::InvalidLayout(_))));
            assert!(matches!(
                tensor.read_logical_block(0).await,
                Err(Error::InvalidLayout(_))
            ));
        }

        let file = blocks.read().await.get_file("0").cloned();
        match malformed {
            1 => assert_eq!(*file.unwrap().read::<Vec<u8>>().await.unwrap(), vec![1, 2]),
            2 => assert_eq!(*file.unwrap().read::<Vec<f64>>().await.unwrap(), vec![1.]),
            _ => assert!(file.is_none()),
        }
        cleanup(&root).await;
    }
}
