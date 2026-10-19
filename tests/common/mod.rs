//! Shared test scaffolding for `fensor` integration tests.
//!
//! Provides an `FsEntry` adapter for `freqfs::Cache`, a unique-tmpdir helper,
//! and a small `iter_coords` walker reused by the access-layer test matrix.
//! Payload tags and complex component pairs belong to this adapter only. Complex
//! encoding's temporary pair vector is bounded by the stored block's capacity.

#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

use destream::{de, en};
use fensor::{Layout, Tensor, TensorElement, TensorFileEntry, TensorSchema};
use freqfs::{Cache, DirLock};
use get_size::GetSize;
use safecast::as_type;

pub mod counters;
pub mod file_size;
pub mod fixture;
pub mod numbers;
mod sparse_codec;

const MAX_METADATA_RANK: usize = 4096;

#[derive(Clone, Debug)]
pub enum FsEntry {
    SparseU8(fensor::SparseNode<u8>),
    SparseU16(fensor::SparseNode<u16>),
    SparseU32(fensor::SparseNode<u32>),
    SparseU64(fensor::SparseNode<u64>),
    SparseI8(fensor::SparseNode<i8>),
    SparseI16(fensor::SparseNode<i16>),
    SparseI32(fensor::SparseNode<i32>),
    SparseI64(fensor::SparseNode<i64>),
    SparseF32(fensor::SparseNode<f32>),
    SparseF64(fensor::SparseNode<f64>),
    #[cfg(feature = "complex")]
    SparseC32(fensor::SparseNode<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    SparseC64(fensor::SparseNode<fensor::complex::Complex64>),
    F32(Vec<f32>),
    MetadataF32(fensor::TensorMetadata<f32>),
    U8(Vec<u8>),
    F64(Vec<f64>),
    MetadataU8(fensor::TensorMetadata<u8>),
    MetadataF64(fensor::TensorMetadata<f64>),
    U16(Vec<u16>),
    MetadataU16(fensor::TensorMetadata<u16>),
    U32(Vec<u32>),
    MetadataU32(fensor::TensorMetadata<u32>),
    U64(Vec<u64>),
    MetadataU64(fensor::TensorMetadata<u64>),
    I8(Vec<i8>),
    MetadataI8(fensor::TensorMetadata<i8>),
    I16(Vec<i16>),
    MetadataI16(fensor::TensorMetadata<i16>),
    I32(Vec<i32>),
    MetadataI32(fensor::TensorMetadata<i32>),
    I64(Vec<i64>),
    MetadataI64(fensor::TensorMetadata<i64>),
    #[cfg(feature = "complex")]
    C32(Vec<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    MetadataC32(fensor::TensorMetadata<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    C64(Vec<fensor::complex::Complex64>),
    #[cfg(feature = "complex")]
    MetadataC64(fensor::TensorMetadata<fensor::complex::Complex64>),
}

impl GetSize for FsEntry {
    fn get_heap_size(&self) -> usize {
        match self {
            Self::U8(values) => values.capacity() * std::mem::size_of::<u8>(),
            Self::MetadataU8(metadata) => metadata.get_heap_size(),
            Self::SparseU8(node) => node.get_heap_size(),
            Self::U16(values) => values.capacity() * std::mem::size_of::<u16>(),
            Self::MetadataU16(metadata) => metadata.get_heap_size(),
            Self::SparseU16(node) => node.get_heap_size(),
            Self::U32(values) => values.capacity() * std::mem::size_of::<u32>(),
            Self::MetadataU32(metadata) => metadata.get_heap_size(),
            Self::SparseU32(node) => node.get_heap_size(),
            Self::U64(values) => values.capacity() * std::mem::size_of::<u64>(),
            Self::MetadataU64(metadata) => metadata.get_heap_size(),
            Self::SparseU64(node) => node.get_heap_size(),
            Self::I8(values) => values.capacity() * std::mem::size_of::<i8>(),
            Self::MetadataI8(metadata) => metadata.get_heap_size(),
            Self::SparseI8(node) => node.get_heap_size(),
            Self::I16(values) => values.capacity() * std::mem::size_of::<i16>(),
            Self::MetadataI16(metadata) => metadata.get_heap_size(),
            Self::SparseI16(node) => node.get_heap_size(),
            Self::I32(values) => values.capacity() * std::mem::size_of::<i32>(),
            Self::MetadataI32(metadata) => metadata.get_heap_size(),
            Self::SparseI32(node) => node.get_heap_size(),
            Self::I64(values) => values.capacity() * std::mem::size_of::<i64>(),
            Self::MetadataI64(metadata) => metadata.get_heap_size(),
            Self::SparseI64(node) => node.get_heap_size(),
            Self::F32(values) => values.capacity() * std::mem::size_of::<f32>(),
            Self::MetadataF32(metadata) => metadata.get_heap_size(),
            Self::SparseF32(node) => node.get_heap_size(),
            Self::F64(values) => values.capacity() * std::mem::size_of::<f64>(),
            Self::MetadataF64(metadata) => metadata.get_heap_size(),
            Self::SparseF64(node) => node.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C32(values) => {
                values.capacity() * std::mem::size_of::<fensor::complex::Complex32>()
            }
            #[cfg(feature = "complex")]
            Self::MetadataC32(metadata) => metadata.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::SparseC32(node) => node.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C64(values) => {
                values.capacity() * std::mem::size_of::<fensor::complex::Complex64>()
            }
            #[cfg(feature = "complex")]
            Self::MetadataC64(metadata) => metadata.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::SparseC64(node) => node.get_heap_size(),
        }
    }
}

impl<'en> en::ToStream<'en> for FsEntry {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        match self {
            Self::SparseU8(node) => {
                en::IntoStream::into_stream((30u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU16(node) => {
                en::IntoStream::into_stream((31u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU32(node) => {
                en::IntoStream::into_stream((32u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU64(node) => {
                en::IntoStream::into_stream((33u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI8(node) => {
                en::IntoStream::into_stream((34u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI16(node) => {
                en::IntoStream::into_stream((35u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI32(node) => {
                en::IntoStream::into_stream((36u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI64(node) => {
                en::IntoStream::into_stream((37u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseF32(node) => {
                en::IntoStream::into_stream((38u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseF64(node) => {
                en::IntoStream::into_stream((39u8, sparse_codec::Encoded(node)), encoder)
            }
            #[cfg(feature = "complex")]
            Self::SparseC32(node) => {
                en::IntoStream::into_stream((40u8, sparse_codec::Encoded(node)), encoder)
            }
            #[cfg(feature = "complex")]
            Self::SparseC64(node) => {
                en::IntoStream::into_stream((41u8, sparse_codec::Encoded(node)), encoder)
            }
            Self::U16(value) => en::IntoStream::into_stream((7u8, value), encoder),
            Self::MetadataU16(value) => en::IntoStream::into_stream((8u8, value), encoder),
            Self::U32(value) => en::IntoStream::into_stream((9u8, value), encoder),
            Self::MetadataU32(value) => en::IntoStream::into_stream((10u8, value), encoder),
            Self::U64(value) => en::IntoStream::into_stream((11u8, value), encoder),
            Self::MetadataU64(value) => en::IntoStream::into_stream((12u8, value), encoder),
            Self::I8(value) => en::IntoStream::into_stream((13u8, value), encoder),
            Self::MetadataI8(value) => en::IntoStream::into_stream((14u8, value), encoder),
            Self::I16(value) => en::IntoStream::into_stream((15u8, value), encoder),
            Self::MetadataI16(value) => en::IntoStream::into_stream((16u8, value), encoder),
            Self::I32(value) => en::IntoStream::into_stream((17u8, value), encoder),
            Self::MetadataI32(value) => en::IntoStream::into_stream((18u8, value), encoder),
            Self::I64(value) => en::IntoStream::into_stream((19u8, value), encoder),
            Self::MetadataI64(value) => en::IntoStream::into_stream((20u8, value), encoder),
            #[cfg(feature = "complex")]
            Self::C32(value) => en::IntoStream::into_stream(
                (21u8, value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>()),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC32(value) => en::IntoStream::into_stream((22u8, value), encoder),
            #[cfg(feature = "complex")]
            Self::C64(value) => en::IntoStream::into_stream(
                (23u8, value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>()),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC64(value) => en::IntoStream::into_stream((24u8, value), encoder),
            Self::F32(value) => en::IntoStream::into_stream((1u8, value), encoder),
            Self::MetadataF32(value) => en::IntoStream::into_stream((2u8, value), encoder),
            Self::U8(value) => en::IntoStream::into_stream((3u8, value), encoder),
            Self::F64(value) => en::IntoStream::into_stream((4u8, value), encoder),
            Self::MetadataU8(value) => en::IntoStream::into_stream((5u8, value), encoder),
            Self::MetadataF64(value) => en::IntoStream::into_stream((6u8, value), encoder),
        }
    }
}

struct FsEntryVisitor;

impl de::Visitor for FsEntryVisitor {
    type Value = FsEntry;

    fn expecting() -> &'static str {
        "a typed filesystem entry"
    }

    async fn visit_seq<A: de::SeqAccess>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let entry = match seq.expect_next::<u8>(()).await? {
            30 => FsEntry::SparseU8(seq.expect_next::<sparse_codec::Decoded<u8>>(()).await?.0),
            31 => FsEntry::SparseU16(seq.expect_next::<sparse_codec::Decoded<u16>>(()).await?.0),
            32 => FsEntry::SparseU32(seq.expect_next::<sparse_codec::Decoded<u32>>(()).await?.0),
            33 => FsEntry::SparseU64(seq.expect_next::<sparse_codec::Decoded<u64>>(()).await?.0),
            34 => FsEntry::SparseI8(seq.expect_next::<sparse_codec::Decoded<i8>>(()).await?.0),
            35 => FsEntry::SparseI16(seq.expect_next::<sparse_codec::Decoded<i16>>(()).await?.0),
            36 => FsEntry::SparseI32(seq.expect_next::<sparse_codec::Decoded<i32>>(()).await?.0),
            37 => FsEntry::SparseI64(seq.expect_next::<sparse_codec::Decoded<i64>>(()).await?.0),
            38 => FsEntry::SparseF32(seq.expect_next::<sparse_codec::Decoded<f32>>(()).await?.0),
            39 => FsEntry::SparseF64(seq.expect_next::<sparse_codec::Decoded<f64>>(()).await?.0),
            #[cfg(feature = "complex")]
            40 => FsEntry::SparseC32(
                seq.expect_next::<sparse_codec::Decoded<fensor::complex::Complex32>>(())
                    .await?
                    .0,
            ),
            #[cfg(feature = "complex")]
            41 => FsEntry::SparseC64(
                seq.expect_next::<sparse_codec::Decoded<fensor::complex::Complex64>>(())
                    .await?
                    .0,
            ),
            1 => FsEntry::F32(seq.expect_next(()).await?),
            2 => FsEntry::MetadataF32(seq.expect_next(MAX_METADATA_RANK).await?),
            3 => FsEntry::U8(seq.expect_next(()).await?),
            4 => FsEntry::F64(seq.expect_next(()).await?),
            5 => FsEntry::MetadataU8(seq.expect_next(MAX_METADATA_RANK).await?),
            6 => FsEntry::MetadataF64(seq.expect_next(MAX_METADATA_RANK).await?),
            7 => FsEntry::U16(seq.expect_next(()).await?),
            8 => FsEntry::MetadataU16(seq.expect_next(MAX_METADATA_RANK).await?),
            9 => FsEntry::U32(seq.expect_next(()).await?),
            10 => FsEntry::MetadataU32(seq.expect_next(MAX_METADATA_RANK).await?),
            11 => FsEntry::U64(seq.expect_next(()).await?),
            12 => FsEntry::MetadataU64(seq.expect_next(MAX_METADATA_RANK).await?),
            13 => FsEntry::I8(seq.expect_next(()).await?),
            14 => FsEntry::MetadataI8(seq.expect_next(MAX_METADATA_RANK).await?),
            15 => FsEntry::I16(seq.expect_next(()).await?),
            16 => FsEntry::MetadataI16(seq.expect_next(MAX_METADATA_RANK).await?),
            17 => FsEntry::I32(seq.expect_next(()).await?),
            18 => FsEntry::MetadataI32(seq.expect_next(MAX_METADATA_RANK).await?),
            19 => FsEntry::I64(seq.expect_next(()).await?),
            20 => FsEntry::MetadataI64(seq.expect_next(MAX_METADATA_RANK).await?),
            #[cfg(feature = "complex")]
            21 => FsEntry::C32(
                seq.expect_next::<Vec<(f32, f32)>>(())
                    .await?
                    .into_iter()
                    .map(|(re, im)| fensor::complex::Complex32::new(re, im))
                    .collect(),
            ),
            #[cfg(feature = "complex")]
            22 => FsEntry::MetadataC32(seq.expect_next(MAX_METADATA_RANK).await?),
            #[cfg(feature = "complex")]
            23 => FsEntry::C64(
                seq.expect_next::<Vec<(f64, f64)>>(())
                    .await?
                    .into_iter()
                    .map(|(re, im)| fensor::complex::Complex64::new(re, im))
                    .collect(),
            ),
            #[cfg(feature = "complex")]
            24 => FsEntry::MetadataC64(seq.expect_next(MAX_METADATA_RANK).await?),
            tag => return Err(de::Error::custom(format!("unknown entry tag {tag}"))),
        };

        if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
            return Err(de::Error::custom("unexpected entry field"));
        }

        Ok(entry)
    }
}

impl de::FromStream for FsEntry {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _: (),
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        decoder.decode_seq(FsEntryVisitor).await
    }
}

as_type!(FsEntry, F32, Vec<f32>);
as_type!(FsEntry, MetadataF32, fensor::TensorMetadata<f32>);
as_type!(FsEntry, U8, Vec<u8>);
as_type!(FsEntry, F64, Vec<f64>);
as_type!(FsEntry, MetadataU8, fensor::TensorMetadata<u8>);
as_type!(FsEntry, MetadataF64, fensor::TensorMetadata<f64>);

as_type!(FsEntry, U16, Vec<u16>);
as_type!(FsEntry, MetadataU16, fensor::TensorMetadata<u16>);

as_type!(FsEntry, U32, Vec<u32>);
as_type!(FsEntry, MetadataU32, fensor::TensorMetadata<u32>);

as_type!(FsEntry, U64, Vec<u64>);
as_type!(FsEntry, MetadataU64, fensor::TensorMetadata<u64>);

as_type!(FsEntry, I8, Vec<i8>);
as_type!(FsEntry, MetadataI8, fensor::TensorMetadata<i8>);

as_type!(FsEntry, I16, Vec<i16>);
as_type!(FsEntry, MetadataI16, fensor::TensorMetadata<i16>);

as_type!(FsEntry, I32, Vec<i32>);
as_type!(FsEntry, MetadataI32, fensor::TensorMetadata<i32>);

as_type!(FsEntry, I64, Vec<i64>);
as_type!(FsEntry, MetadataI64, fensor::TensorMetadata<i64>);

#[cfg(feature = "complex")]
as_type!(FsEntry, C32, Vec<fensor::complex::Complex32>);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    MetadataC32,
    fensor::TensorMetadata<fensor::complex::Complex32>
);

#[cfg(feature = "complex")]
as_type!(FsEntry, C64, Vec<fensor::complex::Complex64>);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    MetadataC64,
    fensor::TensorMetadata<fensor::complex::Complex64>
);

pub fn unique_tmp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    // Wall-clock timestamps alone can collide between concurrent tests.
    static NEXT_DIR: counters::Counter = counters::Counter::new();
    let sequence = NEXT_DIR.increment();
    let process = std::process::id();
    path.push(format!("fensor_test_{name}_{unique}_{process}_{sequence}"));
    path
}

/// Own the fixture directory until every source and view in its case is dropped.
/// Explicit cleanup remains useful for reopen/corruption fixtures; Drop also
/// releases ordinary numerical fixtures when an assertion unwinds.
pub struct Directory(PathBuf);

impl Directory {
    pub async fn new(name: &str) -> Self {
        let root = unique_tmp_dir(name);
        tokio::fs::create_dir(&root).await.expect("create tmp dir");
        Self(root)
    }
}

impl std::ops::Deref for Directory {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Directory {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub async fn new_dir(name: &str) -> (Directory, DirLock<FsEntry>) {
    let root = Directory::new(name).await;
    let dir = open_dir(&root).expect("load tmp dir");
    (root, dir)
}

pub fn open_dir(root: &Path) -> io::Result<DirLock<FsEntry>> {
    let cache = Cache::<FsEntry>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
    cache.load(root.to_path_buf())
}

pub async fn cleanup(root: &Path) {
    let _ = tokio::fs::remove_dir_all(root).await;
}

pub fn block_entries(
    geometry: &fensor::StorageGeometry,
    id: u64,
) -> impl Iterator<Item = (usize, Vec<u64>)> + '_ {
    let bounds = geometry.block_bounds(id).unwrap();
    let shape: Vec<_> = bounds.iter().map(|(start, end)| end - start).collect();
    geometry
        .block_offsets(id)
        .unwrap()
        .zip(iter_coords(&shape).map(move |mut coord| {
            for (position, (start, _)) in coord.iter_mut().zip(&bounds) {
                *position += start;
            }
            coord
        }))
}

pub fn iter_coords(shape: &[u64]) -> CoordIter {
    CoordIter::new(shape)
}

pub struct CoordIter {
    shape: Vec<u64>,
    coord: Vec<u64>,
    remaining: u64,
}

impl CoordIter {
    pub fn new(shape: &[u64]) -> Self {
        let remaining = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        Self {
            shape: shape.to_vec(),
            coord: vec![0u64; shape.len()],
            remaining,
        }
    }
}

impl Iterator for CoordIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let out = self.coord.clone();
        self.remaining -= 1;

        if self.remaining > 0 {
            let mut axis = self.shape.len() - 1;

            loop {
                self.coord[axis] += 1;
                if self.coord[axis] < self.shape[axis] {
                    break;
                }

                self.coord[axis] = 0;
                axis -= 1;
            }
        }

        Some(out)
    }
}

pub async fn create_dense_tensor<T>(
    dir: DirLock<FsEntry>,
    schema: TensorSchema,
) -> Tensor<FsEntry, T>
where
    T: TensorElement,
    FsEntry: TensorFileEntry<T>,
{
    Tensor::<FsEntry, T>::create(dir, schema, Layout::Dense, 1000)
        .await
        .expect("created")
}

pub async fn create_sparse_tensor<T>(
    dir: DirLock<FsEntry>,
    schema: TensorSchema,
    axis: Option<usize>,
) -> Tensor<FsEntry, T>
where
    T: TensorElement,
    FsEntry: TensorFileEntry<T>,
{
    Tensor::<FsEntry, T>::create(dir, schema, Layout::Sparse { axis }, 1000)
        .await
        .expect("created")
}

impl freqfs::FileLoad for FsEntry {
    async fn load_size(
        _: &Path,
        file: &mut tokio::fs::File,
        _: &std::fs::Metadata,
    ) -> io::Result<usize> {
        tbon::de::read_from::<_, file_size::Size>((), file)
            .await
            .map(|bound| bound.0)
            .map_err(io::Error::other)
    }

    async fn load(
        _: &std::path::Path,
        file: tokio::fs::File,
        _: std::fs::Metadata,
    ) -> std::io::Result<Self> {
        counters::record_load();
        tbon::de::read_from((), file)
            .await
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}

impl freqfs::FileSave for FsEntry {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        counters::record_save();
        use futures::TryStreamExt;
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::io::BufWriter::new(file);
        let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
        let mut size = 0;

        while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }

        file.flush().await?;
        Ok(size)
    }
}

safecast::as_type!(FsEntry, SparseU8, fensor::SparseNode<u8>);
safecast::as_type!(FsEntry, SparseU16, fensor::SparseNode<u16>);
safecast::as_type!(FsEntry, SparseU32, fensor::SparseNode<u32>);
safecast::as_type!(FsEntry, SparseU64, fensor::SparseNode<u64>);
safecast::as_type!(FsEntry, SparseI8, fensor::SparseNode<i8>);
safecast::as_type!(FsEntry, SparseI16, fensor::SparseNode<i16>);
safecast::as_type!(FsEntry, SparseI32, fensor::SparseNode<i32>);
safecast::as_type!(FsEntry, SparseI64, fensor::SparseNode<i64>);
safecast::as_type!(FsEntry, SparseF32, fensor::SparseNode<f32>);
safecast::as_type!(FsEntry, SparseF64, fensor::SparseNode<f64>);
#[cfg(feature = "complex")]
safecast::as_type!(
    FsEntry,
    SparseC32,
    fensor::SparseNode<fensor::complex::Complex32>
);
#[cfg(feature = "complex")]
safecast::as_type!(
    FsEntry,
    SparseC64,
    fensor::SparseNode<fensor::complex::Complex64>
);
