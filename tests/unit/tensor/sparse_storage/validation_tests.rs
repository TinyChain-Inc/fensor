use futures::FutureExt;
use number_general::DType;

use crate::test_support::{FsEntry, cleanup, new_dir};
use crate::{Error, Layout, StorageGeometry, Tensor, TensorSchema, TensorSource};

#[tokio::test]
async fn replacement_validation_preserves_lock_precedence() {
    let (root, dir) = new_dir("sparse_validation_order").await;
    let geometry = StorageGeometry::new(
        TensorSchema::new(f32::dtype(), vec![2, 5].into()).unwrap(),
        Layout::Sparse { axis: Some(0) },
        vec![1, 4].into(),
    )
    .unwrap();
    let tensor = Tensor::<FsEntry, f32>::create_with_geometry(dir, geometry.clone())
        .await
        .unwrap();
    let storage = tensor.storage.sparse().unwrap();
    let guard = storage.gate.write().await;

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
    assert!(storage.replace(1, vec![1.; 4]).now_or_never().is_none());
    drop(guard);
    assert!(matches!(
        storage.replace(1, vec![1.; 4]).await,
        Err(Error::InvalidLayout(message)) if message == "nonzero logical block padding"
    ));

    assert!(storage.gate.try_write().is_ok());
    assert_eq!(tensor.read_logical_block(1).await.unwrap(), vec![0.; 4]);
    cleanup(&root).await;
}
