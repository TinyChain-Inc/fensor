//! The file-entry adapter chooses the byte codec and preserves payload types.

use std::io;
use std::path::Path;

use fensor::{
    Layout, Tensor, TensorBooleanScalar, TensorCast, TensorCompare, TensorGeometry, TensorMatMul,
    TensorMath, TensorMathScalar, TensorMetadata, TensorRead, TensorReduce, TensorReduceAll,
    TensorSchema, TensorTransform, TensorWhere, TensorWrite,
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
        tensor.sync().await.unwrap();
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
        copy.sync().await.unwrap();
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

    // The pre-migration representation already stored u64 dimensions in this tuple.
    let fixture = (vec![3u64, 4], true, Some(1u64), vec![1u64, 2]);
    let expected =
        TensorMetadata::<f32>::new(shape![3, 4], Layout::Sparse { axis: Some(1) }, shape![1, 2])
            .unwrap();
    let decoded: TensorMetadata<f32> =
        tbon::de::try_decode((), tbon::en::encode(&fixture).unwrap())
            .await
            .unwrap();
    assert_eq!(decoded, expected);
    let encoded: (Vec<u64>, bool, Option<u64>, Vec<u64>) =
        tbon::de::try_decode((), tbon::en::encode(&expected).unwrap())
            .await
            .unwrap();
    assert_eq!(encoded, fixture);

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

#[tokio::test]
async fn reload_rejects_out_of_bounds_sparse_axis() {
    let root = common::unique_tmp_dir("json_invalid_axis");
    tokio::fs::create_dir(&root).await.unwrap();
    {
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let tensor = Tensor::<JsonEntry, f32>::create(
            cache.load(root.clone()).unwrap(),
            TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![3, 4]).unwrap(),
            Layout::Sparse { axis: Some(1) },
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&[1, 2], 7.).await.unwrap();
        tensor.sync().await.unwrap();
    }

    let metadata = root.join("blocks").join("metadata");
    // Preserve the adapter envelope and valid geometry; corrupt only the axis.
    let original = tokio::fs::read(&metadata).await.unwrap();
    let malformed = (2u8, (vec![3u64, 4], true, Some(2u64), vec![1u64, 2]));
    let mut bytes = Vec::new();
    let mut encoded = destream_json::en::encode(&malformed).unwrap();

    while let Some(chunk) = encoded.try_next().await.unwrap() {
        bytes.extend_from_slice(&chunk);
    }
    tokio::fs::write(&metadata, &bytes).await.unwrap();
    {
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let error = Tensor::<JsonEntry, f32>::load(cache.load(root.clone()).unwrap())
            .await
            .err()
            .expect("invalid persisted axis must fail closed");
        // Decoding is owned by the adapter and reaches fensor as an I/O error.
        assert!(matches!(error, fensor::Error::Io(_)), "{error:?}");
        assert!(
            error.to_string().contains("sparse axis hint out of bounds"),
            "{error}"
        );
    }
    assert_eq!(tokio::fs::read(&metadata).await.unwrap(), bytes);
    tokio::fs::write(&metadata, original).await.unwrap();
    let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
    let tensor = Tensor::<JsonEntry, f32>::load(cache.load(root.clone()).unwrap())
        .await
        .unwrap();
    assert_eq!(tensor.read_value(&[1, 2]).await.unwrap(), 7.);
    common::cleanup(&root).await;
}

// A binary expression can borrow sources with different adapters and block shapes.
#[tokio::test]
async fn binary_sources_use_independent_codecs() {
    let a_root = common::unique_tmp_dir("binary_json");
    tokio::fs::create_dir(&a_root).await.unwrap();
    let cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let a_dir = cache.load(a_root).unwrap();
    let a = Tensor::<JsonEntry, f32>::create(
        a_dir.clone(),
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let (_, b_dir) = common::new_dir("binary_tbon").await;
    let b = Tensor::<FsEntry, f32>::create(
        b_dir,
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Dense,
        3,
    )
    .await
    .unwrap();
    a.write_value(&[4], 2.5).await.unwrap();
    b.write_value(&[4], 3.5).await.unwrap();
    a.sync().await.unwrap();
    let expression = a.view().add(&b.view()).await.unwrap().clone();
    let (_, output_dir) = common::new_dir("binary_codecs_copy").await;
    let output: Tensor<FsEntry, f32> = Tensor::copy_from(output_dir, &expression, 4).await.unwrap();

    assert_eq!(output.read_value(&[4]).await.unwrap(), 6.);
    assert_eq!(output.read_value(&[0]).await.unwrap(), 0.);
}

// Conditional expressions and predicates retain independent source/destination adapters.
#[tokio::test]
async fn conditional_sources_and_output_use_independent_codecs() {
    let a_root = common::unique_tmp_dir("conditional_json");
    tokio::fs::create_dir(&a_root).await.unwrap();
    let cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let a = Tensor::<JsonEntry, f32>::create(
        cache.load(a_root).unwrap(),
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Sparse { axis: None },
        2,
    )
    .await
    .unwrap();
    let (_, b_dir) = common::new_dir("conditional_tbon").await;
    let b = Tensor::<FsEntry, f32>::create(
        b_dir,
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Sparse { axis: None },
        3,
    )
    .await
    .unwrap();
    a.write_value(&[3], 2.).await.unwrap();
    b.write_value(&[4], 3.).await.unwrap();
    let condition = a
        .view()
        .gt(&b.view())
        .await
        .unwrap()
        .and_scalar(127)
        .await
        .unwrap();
    let expression = condition
        .cond(&a.view().add_scalar(1.).await.unwrap(), &b.view())
        .await
        .unwrap()
        .clone();
    let out_root = common::unique_tmp_dir("conditional_json_output");
    tokio::fs::create_dir(&out_root).await.unwrap();
    let out_cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let out_dir = out_cache.load(out_root.clone()).unwrap();
    let output: Tensor<JsonEntry, f32> = Tensor::copy_from(out_dir.clone(), &expression, 4)
        .await
        .unwrap();
    output.sync().await.unwrap();
    drop(output);
    drop(out_dir);
    let reload_cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let output = Tensor::<JsonEntry, f32>::load(reload_cache.load(out_root).unwrap())
        .await
        .unwrap();
    assert_eq!(output.read_value(&[0]).await.unwrap(), 0.);
    assert_eq!(output.read_value(&[3]).await.unwrap(), 3.);
    assert_eq!(output.read_value(&[4]).await.unwrap(), 3.);
    let reduced = expression.sum(ha_ndarray::axes![0], false).await.unwrap();
    assert_eq!(reduced.sum_all().await.unwrap(), 6.);
    let (_, dir) = common::new_dir("reduced_independent_codec").await;
    let reduced_copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &reduced, 1).await.unwrap();
    assert_eq!(reduced_copy.read_value(&[0]).await.unwrap(), 6.);
    let matrix = a
        .view()
        .reshape(ha_ndarray::shape![1, 5])
        .unwrap()
        .matmul(&b.view().reshape(ha_ndarray::shape![5, 1]).unwrap())
        .await
        .unwrap();
    let (_, dir) = common::new_dir("matmul_independent_codec").await;
    let product: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &matrix, 1).await.unwrap();
    assert_eq!(product.read_value(&[0, 0]).await.unwrap(), 0.);
}
