//! Shared test scaffolding for `fensor` integration tests.
//!
//! Provides an `FsEntry` adapter for `freqfs::Cache`, a unique-tmpdir helper,
//! and a small `iter_coords` walker reused by the access-layer test matrix.

#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

use b_table::Node;
use destream::{de, en};
use fensor::{Layout, TensorSchema, TensorSparseIndex};
use freqfs::{Cache, DirLock};
use safecast::as_type;

#[derive(Clone, Debug)]
pub enum FsEntry {
    Node(Node<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Text(String),
}

impl<'en> en::ToStream<'en> for FsEntry {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Node(node) => node.to_stream(encoder),
            Self::F32(values) => values.to_stream(encoder),
            Self::F64(values) => values.to_stream(encoder),
            Self::Text(text) => text.to_stream(encoder),
        }
    }
}

// `TensorFileEntry<T>: FileLoad` is only satisfiable via the blanket
// `impl<T: FromStream> FileLoad for T`, so `FsEntry` needs a `FromStream` impl to
// type-check. Every read in this codebase goes through a concrete `AsType` target
// (`String`/`Vec<f32>`/`Vec<f64>`/`Node<u64>`), never through `FsEntry` itself, so
// this is never actually invoked at runtime.
impl de::FromStream for FsEntry {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), _decoder: &mut D) -> Result<Self, D::Error> {
        Err(de::Error::custom(
            "FsEntry does not support generic decoding; read via a concrete AsType target",
        ))
    }
}

as_type!(FsEntry, Node, Node<u64>);
as_type!(FsEntry, F32, Vec<f32>);
as_type!(FsEntry, F64, Vec<f64>);
as_type!(FsEntry, Text, String);

pub fn unique_tmp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    path.push(format!("fensor_test_{name}_{unique}"));
    path
}

pub async fn new_dir(name: &str) -> (PathBuf, DirLock<FsEntry>) {
    let root = unique_tmp_dir(name);
    tokio::fs::create_dir(&root).await.expect("create tmp dir");
    let dir = open_dir(&root).expect("load tmp dir");
    (root, dir)
}

pub fn open_dir(root: &Path) -> io::Result<DirLock<FsEntry>> {
    let cache = Cache::<FsEntry>::new(1_000_000, None);
    cache.load(root.to_path_buf())
}

pub async fn cleanup(root: &Path) {
    let _ = tokio::fs::remove_dir_all(root).await;
}

pub fn iter_coords(shape: &[usize]) -> CoordIter {
    CoordIter::new(shape)
}

/// Test helper: derive the sparse-index key `[coord[sparse_axis], block_offset]`
/// from a coord, using only the public `TensorSchema` surface. Mirrors the
/// arithmetic in `Tensor::block_position_from_base_coord` + `Tensor::sparse_key`.
/// Pass the **base** schema (the one used at `create`), not a view-transformed
/// one — views can change strides relative to the on-disk layout.
pub fn block_key_for_coord(schema: &TensorSchema, coord: &[u64]) -> Vec<u64> {
    assert_eq!(
        coord.len(),
        schema.shape().len(),
        "coord rank must match schema rank"
    );

    let strides = schema.strides();
    let block_len = schema.block_len().max(1) as u64;
    let offset: u64 = coord
        .iter()
        .zip(strides.iter())
        .map(|(c, s)| c * (*s as u64))
        .sum();
    let block_offset = offset / block_len;

    let sparse_axis = match schema.layout() {
        Layout::Sparse { axis } => axis.unwrap_or(0),
        Layout::Dense => 0,
    };
    let axis = sparse_axis.min(coord.len().saturating_sub(1));

    vec![coord[axis], block_offset]
}

/// Test helper: probe the sparse index for the block backing `coord`.
/// Wraps `block_key_for_coord` + `TensorSparseIndex::lookup_block_id` so test
/// bodies can talk in coord-space instead of hand-spelling block keys.
pub async fn block_id_for_coord<T: TensorSparseIndex>(
    tensor: &T,
    schema: &TensorSchema,
    coord: &[u64],
) -> Option<u64> {
    let key = block_key_for_coord(schema, coord);
    tensor.lookup_block_id(&key).await.expect("lookup_block_id")
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
