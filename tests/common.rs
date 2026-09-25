//! Shared test scaffolding for `fensor` integration tests.
//!
//! Provides an `FsEntry` adapter for `freqfs::Cache`, a unique-tmpdir helper,
//! and a small `iter_coords` walker reused by the access-layer test matrix.

#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

use b_table::Node;
use destream::{de, en};
use fensor::{Layout, Tensor, TensorElement, TensorFileEntry, TensorSchema};
use freqfs::{Cache, DirLock};
use safecast::as_type;

#[path = "common/counters.rs"]
pub mod counters;

#[derive(Clone, Debug)]
pub enum FsEntry {
    Node(Node<u64>),
    F32(Vec<f32>),
    MetadataF32(fensor::TensorMetadata<f32>),
    U8(Vec<u8>),
    F64(Vec<f64>),
    MetadataU8(fensor::TensorMetadata<u8>),
    MetadataF64(fensor::TensorMetadata<f64>),
}

impl<'en> en::ToStream<'en> for FsEntry {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        match self {
            Self::Node(value) => en::IntoStream::into_stream((0u8, value), encoder),
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
            0 => FsEntry::Node(seq.expect_next(()).await?),
            1 => FsEntry::F32(seq.expect_next(()).await?),
            2 => FsEntry::MetadataF32(seq.expect_next(()).await?),
            3 => FsEntry::U8(seq.expect_next(()).await?),
            4 => FsEntry::F64(seq.expect_next(()).await?),
            5 => FsEntry::MetadataU8(seq.expect_next(()).await?),
            6 => FsEntry::MetadataF64(seq.expect_next(()).await?),
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

as_type!(FsEntry, Node, Node<u64>);
as_type!(FsEntry, F32, Vec<f32>);
as_type!(FsEntry, MetadataF32, fensor::TensorMetadata<f32>);
as_type!(FsEntry, U8, Vec<u8>);
as_type!(FsEntry, F64, Vec<f64>);
as_type!(FsEntry, MetadataU8, fensor::TensorMetadata<u8>);
as_type!(FsEntry, MetadataF64, fensor::TensorMetadata<f64>);

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

pub async fn new_dir(name: &str) -> (PathBuf, DirLock<FsEntry>) {
    let root = unique_tmp_dir(name);
    tokio::fs::create_dir(&root).await.expect("create tmp dir");
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

pub fn iter_coords(shape: &[usize]) -> CoordIter {
    CoordIter::new(shape)
}

pub struct CoordIter {
    shape: Vec<usize>,
    coord: Vec<u64>,
    remaining: usize,
}

impl CoordIter {
    pub fn new(shape: &[usize]) -> Self {
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
                if (self.coord[axis] as usize) < self.shape[axis] {
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
        let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
        let mut size = 0;

        while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }

        Ok(size)
    }
}
