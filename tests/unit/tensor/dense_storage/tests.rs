use std::time::Duration;

use futures::FutureExt;
use ha_ndarray::shape;
use number_general::DType;

use crate::request::BatchRequest;
use crate::test_support::{FsEntry, new_dir};
use crate::{Layout, Tensor, TensorSchema};

#[tokio::test]
async fn dense_reads_delegate_capacity_and_release_failed_work() {
    let (root, dir) = new_dir("dense_read_capacity").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![2048]).unwrap(),
        Layout::Dense,
        512,
    )
    .await
    .unwrap();

    let request = BatchRequest::linear(0, 2048).unwrap();
    tensor.sync().await.unwrap();
    drop(tensor);

    // Successful buffering has room for all four required payloads.
    for handles in [None, Some(1)] {
        let cache = freqfs::Cache::<FsEntry>::new(32768, handles, 0, Duration::from_secs(1));
        let dir = cache.load(root.to_path_buf()).unwrap();
        let tensor = Tensor::<FsEntry, f32>::load(dir).await.unwrap();
        let values =
            tokio::time::timeout(Duration::from_secs(5), tensor.read_batch(&request, None))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(values, vec![0.; 2048]);
    }

    // Pin one payload in a cache that cannot admit a second. Failure must release
    // the attempted read; releasing the pin permits a subsequent demand read.
    let cache = freqfs::Cache::<FsEntry>::new(4096, None, 0, Duration::from_millis(50));
    let tensor = Tensor::<FsEntry, f32>::load(cache.load(root.to_path_buf()).unwrap())
        .await
        .unwrap();
    let crate::tensor::Storage::Dense(owner) = tensor.storage.as_ref() else {
        unreachable!()
    };
    let held = owner.read::<f32>(0).await.unwrap();
    let request = BatchRequest::linear(512, 512).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(5), tensor.read_batch(&request, None))
        .await
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::Io(error) if error.kind() == std::io::ErrorKind::ResourceBusy)
    );
    drop(held);

    assert_eq!(
        tensor.read_batch(&request, None).await.unwrap(),
        vec![0.; 512]
    );
}

#[tokio::test]
async fn buffered_reads_preserve_order_and_release_guards() {
    let (_root, dir) = new_dir("buffered_read_order").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![8]).unwrap(),
        Layout::Dense,
        1,
    )
    .await
    .unwrap();

    for id in 0..8 {
        tensor
            .replace_logical_block(id, vec![id as f32])
            .await
            .unwrap();
    }

    let crate::tensor::Storage::Dense(owner) = tensor.storage.as_ref() else {
        unreachable!()
    };
    let first = super::required_file(&*owner.blocks.read().await, 0).unwrap();
    let second = super::required_file(&*owner.blocks.read().await, 1).unwrap();
    let request =
        BatchRequest::explicit(vec![vec![0], vec![2], vec![2], vec![1], vec![7]]).unwrap();

    for cancel in [false, true] {
        let held = first.write::<Vec<f32>>().await.unwrap();
        let mut read = Box::pin(tensor.read_batch(&request, None));
        assert!(read.as_mut().now_or_never().is_none());

        if num_cpus::get() > 1 {
            // The second completed read is buffered behind the blocked first.
            assert!(second.try_write::<Vec<f32>>().is_err());
        }

        if cancel {
            drop(read);
            drop(held);
            drop(second.try_write::<Vec<f32>>().unwrap());
            drop(first.try_write::<Vec<f32>>().unwrap());
        } else {
            drop(held);
            assert_eq!(read.await.unwrap(), vec![0., 2., 2., 1., 7.]);
        }
    }

    // Later lookup failures stay behind the earlier payload validation.
    let mut held = first.write::<Vec<f32>>().await.unwrap();
    held.clear();
    owner.blocks.write().await.delete("1").await;
    let mut read = Box::pin(tensor.read_batch(&request, None));
    assert!(read.as_mut().now_or_never().is_none());
    drop(held);
    assert!(read.await.unwrap_err().to_string().contains("length"));
    drop(first.try_write::<Vec<f32>>().unwrap());
}
