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
use get_size::GetSize;
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

impl GetSize for JsonEntry {
    fn get_heap_size(&self) -> usize {
        self.0.get_heap_size()
    }
}

impl FileLoad for JsonEntry {
    async fn load_size(
        _: &Path,
        file: &mut tokio::fs::File,
        _: &std::fs::Metadata,
    ) -> io::Result<usize> {
        destream_json::de::read_from::<_, common::file_size::Size>((), file)
            .await
            .map(|bound| bound.0)
            .map_err(io::Error::other)
    }

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
async fn file_preflight_bounds_retained_containers_without_decoding_payloads() {
    let rank = fensor::Shape::new().inline_size() + 1;
    let entries = [
        FsEntry::U8(vec![3; 7]),
        FsEntry::F64(vec![0.; 65]),
        FsEntry::MetadataF32(
            TensorMetadata::new(vec![1; rank].into(), Layout::Dense, vec![1; rank].into()).unwrap(),
        ),
        FsEntry::Node(b_table::Node::Leaf(vec![vec![1, 2, 3]; 17])),
        FsEntry::SparseF64(b_table::Node::Leaf(vec![
            vec![
                fensor::SparseCell::Key(1),
                fensor::SparseCell::Key(2),
                fensor::SparseCell::Value(3.),
            ];
            17
        ])),
    ];
    for entry in entries {
        let json: FsEntry =
            destream_json::de::try_decode((), destream_json::en::encode(&entry).unwrap())
                .await
                .unwrap();
        let bound: common::file_size::Size =
            destream_json::de::try_decode((), destream_json::en::encode(&entry).unwrap())
                .await
                .unwrap();
        assert!(bound.0 >= json.get_size());
        let tbon: FsEntry = tbon::de::try_decode((), tbon::en::encode(&entry).unwrap())
            .await
            .unwrap();
        let bound: common::file_size::Size =
            tbon::de::try_decode((), tbon::en::encode(&entry).unwrap())
                .await
                .unwrap();
        assert!(bound.0 >= tbon.get_size());
    }
    // This decoded page fits 16 KiB when decoding owns only its final rows;
    // retaining a second encoded-row Vec would exceed that admission bound.
    let tiny_page = FsEntry::SparseF64(b_table::Node::Leaf(vec![
        vec![
            fensor::SparseCell::Key(1),
            fensor::SparseCell::Key(2),
            fensor::SparseCell::Value(3.),
        ];
        160
    ]));
    let bound: common::file_size::Size =
        tbon::de::try_decode((), tbon::en::encode(&tiny_page).unwrap())
            .await
            .unwrap();
    let decoded: FsEntry = tbon::de::try_decode((), tbon::en::encode(&tiny_page).unwrap())
        .await
        .unwrap();
    assert!(bound.0 >= decoded.get_size());
    assert!(bound.0 <= 16 * 1024);

    let oversized = (1u8, vec![0f32; fensor::MAX_BLOCK_CAPACITY + 1]);
    assert!(
        tbon::de::try_decode::<_, _, common::file_size::Size>(
            (),
            tbon::en::encode(&oversized).unwrap()
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn json_storage_supports_dense_sparse_copy_and_reload() {
    for layout in [
        Layout::Dense,
        Layout::Sparse { axis: None },
        Layout::Sparse { axis: Some(0) },
    ] {
        let root = common::Directory::new("json_storage").await;
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.clone().load(root.to_path_buf()).unwrap();
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
        let dir = cache.clone().load(root.to_path_buf()).unwrap();
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
async fn metadata_decoding_uses_the_adapters_rank_limit_for_both_shapes() {
    for (rank, limit) in [(1, 0), (1, 1), (2, 1), (2, 2), (2, 3)] {
        let metadata =
            TensorMetadata::<f32>::new(vec![1; rank].into(), Layout::Dense, vec![1; rank].into())
                .unwrap();
        let json: Result<TensorMetadata<f32>, _> =
            destream_json::de::try_decode(limit, destream_json::en::encode(&metadata).unwrap())
                .await;
        let tbon: Result<TensorMetadata<f32>, _> =
            tbon::de::try_decode(limit, tbon::en::encode(&metadata).unwrap()).await;
        assert_eq!(json.is_ok(), rank <= limit);
        assert_eq!(tbon.is_ok(), rank <= limit);
    }
    // The block shape has its own bound, before rank/geometry validation.
    let oversized_block_shape = (vec![1u64], false, None::<u64>, vec![1u64, 1]);
    let error = tbon::de::try_decode::<_, _, TensorMetadata<f32>>(
        1,
        tbon::en::encode(&oversized_block_shape).unwrap(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("rank limit"));
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
            destream_json::de::try_decode(2, destream_json::en::encode(&metadata).unwrap())
                .await
                .unwrap();
        let tbon: TensorMetadata<f32> =
            tbon::de::try_decode(2, tbon::en::encode(&metadata).unwrap())
                .await
                .unwrap();
        assert_eq!(json, metadata);
        assert_eq!(tbon, metadata);
    }

    let extra = (vec![3u64, 4], true, Some(1u64), vec![1u64, 2], 0u64);
    assert!(
        tbon::de::try_decode::<_, _, TensorMetadata<f32>>(2, tbon::en::encode(&extra).unwrap())
            .await
            .is_err()
    );
    let fixture = (vec![3u64, 4], true, Some(1u64), vec![1u64, 2]);
    let expected =
        TensorMetadata::<f32>::new(shape![3, 4], Layout::Sparse { axis: Some(1) }, shape![1, 2])
            .unwrap();
    let decoded: TensorMetadata<f32> = tbon::de::try_decode(2, tbon::en::encode(&fixture).unwrap())
        .await
        .unwrap();
    assert_eq!(decoded, expected);

    let malformed = [
        (vec![3u64, 4], false, None, vec![2u64]),
        (vec![3, 4], false, None, vec![0, 2]),
        (vec![3, 4], true, Some(2u64), vec![1, 2]),
        (vec![3, 4], false, Some(0), vec![1, 2]),
        (vec![3, 4], false, None, vec![4096, 2]),
    ];

    for value in malformed {
        let decoded: Result<TensorMetadata<f32>, _> =
            destream_json::de::try_decode(2, destream_json::en::encode(&value).unwrap()).await;

        assert!(decoded.is_err());
    }
}

#[tokio::test]
async fn reload_rejects_out_of_bounds_sparse_axis() {
    let root = common::Directory::new("json_invalid_axis").await;
    {
        let cache = Cache::<JsonEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
        let tensor = Tensor::<JsonEntry, f32>::create(
            cache.load(root.to_path_buf()).unwrap(),
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
        let error = Tensor::<JsonEntry, f32>::load(cache.load(root.to_path_buf()).unwrap())
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
    let tensor = Tensor::<JsonEntry, f32>::load(cache.load(root.to_path_buf()).unwrap())
        .await
        .unwrap();
    assert_eq!(tensor.read_value(&[1, 2]).await.unwrap(), 7.);
    common::cleanup(&root).await;
}

// A binary expression can borrow sources with different adapters and block shapes.
#[tokio::test]
async fn binary_sources_use_independent_codecs() {
    let a_root = common::Directory::new("binary_json").await;
    let cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let a_dir = cache.load(a_root.to_path_buf()).unwrap();
    let a = Tensor::<JsonEntry, f32>::create(
        a_dir.clone(),
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let (_b_dir_root, b_dir) = common::new_dir("binary_tbon").await;
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
    let (_output_dir_root, output_dir) = common::new_dir("binary_codecs_copy").await;
    let output: Tensor<FsEntry, f32> = Tensor::copy_from(output_dir, &expression, 4).await.unwrap();

    assert_eq!(output.read_value(&[4]).await.unwrap(), 6.);
    assert_eq!(output.read_value(&[0]).await.unwrap(), 0.);
}

// Conditional expressions and predicates retain independent source/destination adapters.
#[tokio::test]
async fn conditional_sources_and_output_use_independent_codecs() {
    let a_root = common::Directory::new("conditional_json").await;
    let cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let a = Tensor::<JsonEntry, f32>::create(
        cache.load(a_root.to_path_buf()).unwrap(),
        TensorSchema::new(<f32 as number_general::DType>::dtype(), shape![5]).unwrap(),
        Layout::Sparse { axis: None },
        2,
    )
    .await
    .unwrap();
    let (_b_dir_root, b_dir) = common::new_dir("conditional_tbon").await;
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
    let out_root = common::Directory::new("conditional_json_output").await;
    let out_cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let out_dir = out_cache.load(out_root.to_path_buf()).unwrap();
    let output: Tensor<JsonEntry, f32> = Tensor::copy_from(out_dir.clone(), &expression, 4)
        .await
        .unwrap();
    output.sync().await.unwrap();
    drop(output);
    drop(out_dir);
    let reload_cache = Cache::<JsonEntry>::new(1024, None, 0, std::time::Duration::from_secs(1));
    let output = Tensor::<JsonEntry, f32>::load(reload_cache.load(out_root.to_path_buf()).unwrap())
        .await
        .unwrap();
    assert_eq!(output.read_value(&[0]).await.unwrap(), 0.);
    assert_eq!(output.read_value(&[3]).await.unwrap(), 3.);
    assert_eq!(output.read_value(&[4]).await.unwrap(), 3.);
    let reduced = expression.sum(ha_ndarray::axes![0], false).await.unwrap();
    assert_eq!(reduced.sum_all().await.unwrap(), 6.);
    let (_dir_root, dir) = common::new_dir("reduced_independent_codec").await;
    let reduced_copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &reduced, 1).await.unwrap();
    assert_eq!(reduced_copy.read_value(&[0]).await.unwrap(), 6.);
    let matrix = a
        .view()
        .reshape(ha_ndarray::shape![1, 5])
        .unwrap()
        .matmul(&b.view().reshape(ha_ndarray::shape![5, 1]).unwrap())
        .await
        .unwrap();
    let (_dir_root, dir) = common::new_dir("matmul_independent_codec").await;
    let product: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &matrix, 1).await.unwrap();
    assert_eq!(product.read_value(&[0, 0]).await.unwrap(), 0.);
}
