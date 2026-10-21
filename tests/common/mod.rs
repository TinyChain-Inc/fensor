//! Shared test scaffolding for `fensor` integration tests.
//!
//! Provides an `FsEntry` adapter for `freqfs::Cache`, a unique-tmpdir helper,
//! and a small `iter_coords` walker reused by the access-layer test matrix.
//! The test-only file representation and codec live in `file`.

#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

use fensor::{Layout, Tensor, TensorElement, TensorFileEntry, TensorSchema};
use freqfs::{Cache, DirLock};

pub mod counters;
mod file;
pub mod fixture;
pub mod numbers;
mod sparse_codec;

pub use file::FsEntry;

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
