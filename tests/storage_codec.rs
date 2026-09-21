//! The file-entry adapter chooses the byte codec and preserves payload types.
use std::io;
use std::path::Path;

use fensor::{
    Layout, Tensor, TensorCast, TensorGeometry, TensorMetadata, TensorRead, TensorSchema,
    TensorWrite,
};
use freqfs::{Cache, FileLoad, FileSave};
use futures::TryStreamExt;
use ha_ndarray::shape;
use safecast::AsType;
use tokio::io::AsyncWriteExt;

mod common;
use common::FsEntry;

// This adapter chooses JSON explicitly. Its envelope belongs to the application.
#[derive(Clone)]
struct JsonEntry(FsEntry);

impl<F> From<F> for JsonEntry
where
    FsEntry: From<F>,
{
    fn from(value: F) -> Self {
        Self(FsEntry::from(value))
    }
}
impl<F> AsType<F> for JsonEntry
where
    FsEntry: AsType<F>,
{
    fn into_type(self) -> Option<F> {
        self.0.into_type()
    }
    fn as_type(&self) -> Option<&F> {
        self.0.as_type()
    }
    fn as_type_mut(&mut self) -> Option<&mut F> {
        self.0.as_type_mut()
    }
}
impl FileLoad for JsonEntry {
    async fn load(_: &Path, file: tokio::fs::File, _: std::fs::Metadata) -> io::Result<Self> {
        destream_json::de::read_from((), file)
            .await
            .map(Self)
            .map_err(io::Error::other)
    }
}
impl FileSave for JsonEntry {
    async fn save(&self, file: &mut tokio::fs::File) -> io::Result<u64> {
        let mut stream = destream_json::en::encode(&self.0).map_err(io::Error::other)?;
        let mut size = 0;
        while let Some(chunk) = stream.try_next().await.map_err(io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }
        Ok(size)
    }
}

#[tokio::test]
async fn json_storage_supports_dense_sparse_copy_and_reload() {
    for layout in [
        Layout::Dense,
        Layout::Sparse { axis: None },
        Layout::Sparse { axis: Some(0) },
    ] {
        let root = common::unique_tmp_dir("json_storage");
        tokio::fs::create_dir(&root).await.unwrap();
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.clone().load(root.clone()).unwrap();
        let tensor = Tensor::<JsonEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&[4], 7.5).await.unwrap();
        dir.sync().await.unwrap();
        drop(tensor);
        drop(dir);
        drop(cache);
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.clone().load(root.clone()).unwrap();
        assert!(Tensor::<JsonEntry, u8>::load(dir.clone()).await.is_err());
        let tensor = Tensor::<JsonEntry, f32>::load(dir).await.unwrap();
        assert_eq!(tensor.read_value(&[4]).await.unwrap(), 7.5);
        assert_eq!(tensor.read_value(&[0]).await.unwrap(), 0.0);
        assert_eq!(tensor.layout(), layout);
        let (copy_root, copy_dir) = common::new_dir("json_to_tbon").await;
        let cast = tensor.view().cast().await.unwrap();
        let copy = Tensor::<FsEntry, f64>::copy_from(copy_dir.clone(), &cast, 2)
            .await
            .unwrap();
        assert_eq!(copy.read_value(&[4]).await.unwrap(), 7.5);
        copy_dir.sync().await.unwrap();
        drop(copy);
        drop(copy_dir);
        let copy = Tensor::<FsEntry, f64>::load(common::open_dir(&copy_root).unwrap())
            .await
            .unwrap();
        assert_eq!(copy.read_value(&[4]).await.unwrap(), 7.5);
        common::cleanup(&root).await;
        common::cleanup(&copy_root).await;
    }
}

#[tokio::test]
async fn metadata_is_codec_independent_and_rejects_invalid_geometry() {
    for layout in [
        Layout::Dense,
        Layout::Sparse { axis: None },
        Layout::Sparse { axis: Some(1) },
    ] {
        let metadata = TensorMetadata::<f32>::new(shape![3, 4], layout, shape![1, 2]).unwrap();
        let json: TensorMetadata<f32> =
            destream_json::de::try_decode((), destream_json::en::encode(&metadata).unwrap())
                .await
                .unwrap();
        let tbon: TensorMetadata<f32> =
            tbon::de::try_decode((), tbon::en::encode(&metadata).unwrap())
                .await
                .unwrap();
        assert_eq!(json, metadata);
        assert_eq!(tbon, metadata);
    }
    let malformed = [
        (vec![3u64, 4], false, None, vec![2u64]),
        (vec![3, 4], false, None, vec![0, 2]),
        (vec![3, 4], true, Some(2u64), vec![1, 2]),
        (vec![3, 4], false, Some(0), vec![1, 2]),
        (vec![3, 4], false, None, vec![4096, 2]),
    ];
    for value in malformed {
        let decoded: Result<TensorMetadata<f32>, _> =
            destream_json::de::try_decode((), destream_json::en::encode(&value).unwrap()).await;
        assert!(decoded.is_err());
    }
}
