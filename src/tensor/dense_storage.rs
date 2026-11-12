//! Dense storage keeps each logical block in a required typed payload file.

use freqfs::{Dir, DirLock, FileLoad, FileLock, FileReadGuardOwned};
use futures::{StreamExt, TryStreamExt, stream};

use super::{Storage, Tensor, TensorElement, TensorFileEntry, apply_block_updates};
use crate::request::BatchRequest;
use crate::schema::StorageSchema;
use crate::{BoxFuture, Error, Result, StorageGeometry, TensorSource};

pub(super) struct DenseStorage<FE> {
    blocks: DirLock<FE>,
    geometry: Box<StorageGeometry>,
}

// Boxed storage futures bound stack usage in deeply composed consumers.
impl<FE> DenseStorage<FE> {
    pub(super) fn new(blocks: DirLock<FE>, geometry: StorageGeometry) -> Self {
        Self {
            blocks,
            geometry: Box::new(geometry),
        }
    }

    pub(super) fn storage_schema(&self) -> &StorageSchema {
        &self.geometry.storage
    }

    pub(super) async fn initialize<T>(&self) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        for id in 0..self.geometry.storage.shape.iter().product::<u64>() {
            let values = vec![T::default(); self.block_len()];
            self.blocks
                .write()
                .await
                .create_file(id.to_string(), values)
                .await?;
        }

        Ok(())
    }

    pub(super) async fn validate<T>(&self) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        for id in 0..self.geometry.storage.shape.iter().product::<u64>() {
            drop(self.read::<T>(id).await?);
        }

        Ok(())
    }

    pub(super) async fn publish_metadata<T>(&self, metadata: super::TensorMetadata<T>) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        super::create_metadata_file(&self.blocks, metadata).await
    }

    pub(super) async fn sync(&self) -> Result<()>
    where
        FE: freqfs::FileSave + Clone,
    {
        self.blocks.sync().await.map_err(Error::from)
    }

    pub(super) async fn sync_all(&self, directory: &DirLock<FE>) -> Result<()>
    where
        FE: freqfs::FileSave + Clone,
    {
        self.sync().await?;
        directory.sync_all().await?;
        Ok(())
    }

    fn block_len(&self) -> usize {
        self.geometry
            .storage
            .block_schema
            .shape
            .iter()
            .product::<u64>() as usize
    }

    fn validate_id(&self, id: u64) -> Result<()> {
        if id >= self.geometry.storage.shape.iter().product::<u64>() {
            return Err(Error::InvalidCoord("logical block out of bounds".into()));
        }

        Ok(())
    }

    pub(super) async fn replace<T>(&self, id: u64, block: Vec<T>) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        self.geometry.validate_block(id, &block)?;

        let file = required_file(&*self.blocks.read().await, id)?;
        // Acquiring this guard only decodes/checks the existing payload type;
        // replacement values are applied below, after stored-data validation.
        let mut stored = file
            .write::<Vec<T>>()
            .await
            .map_err(freqfs::IntegrityError::from_read)?;
        self.geometry.validate_block(id, &stored).map_err(|error| {
            Error::Io(freqfs::IntegrityError::mark(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                error,
            )))
        })?;
        *stored = block;
        Ok(())
    }

    pub(super) async fn update<T>(&self, id: u64, updates: &[(usize, T)]) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        self.geometry.validate_id(id)?;
        let file = required_file(&*self.blocks.read().await, id)?;
        // No update input reaches the file while its existing type is checked.
        let mut block = file
            .write::<Vec<T>>()
            .await
            .map_err(freqfs::IntegrityError::from_read)?;
        self.geometry.validate_block(id, &block).map_err(|error| {
            Error::Io(freqfs::IntegrityError::mark(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                error,
            )))
        })?;
        for &(offset, value) in updates {
            if offset >= self.block_len() {
                return Err(Error::InvalidLayout("invalid block update offset".into()));
            }
            if value != T::ZERO
                && (0..self.geometry.schema.shape().len()).any(|axis| {
                    let storage = &self.geometry.storage;
                    let extent = self.geometry.block_shape()[axis];
                    let grid = (id / storage.strides[axis]) % storage.shape[axis];
                    let local = (offset as u64 / storage.block_schema.strides[axis]) % extent;
                    local >= self.geometry.schema.shape()[axis] - grid * extent
                })
            {
                return Err(Error::InvalidLayout("nonzero logical block padding".into()));
            }
        }
        apply_block_updates(&mut block, self.block_len(), updates)
    }

    pub(super) fn read<T>(&self, id: u64) -> BoxFuture<'_, Result<FileReadGuardOwned<FE, Vec<T>>>>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        Box::pin(async move {
            self.validate_id(id)?;
            let file = required_file(&*self.blocks.read().await, id)?;
            let block = file
                .into_read::<Vec<T>>()
                .await
                .map_err(freqfs::IntegrityError::from_read)?;
            self.geometry.validate_block(id, &block).map_err(|error| {
                Error::Io(freqfs::IntegrityError::mark(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    error,
                )))
            })?;
            Ok(block)
        })
    }

    pub(super) async fn visit_blocks<T, C>(&self, ids: &[u64], consume: &mut C) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
        C: FnMut(u64, &[T]) -> Result<()> + Send,
    {
        // Ordinary reads retain their guards until consumed. Cache admission may
        // fail when these reads or independent consumers pin too much capacity.
        let reads = stream::iter(ids.iter().copied())
            .map(|id| async move { self.read::<T>(id).await.map(|block| (id, block)) })
            .buffered(num_cpus::get().max(1));
        let mut reads = std::pin::pin!(reads);

        while let Some((id, block)) = reads.try_next().await? {
            #[cfg(test)]
            crate::read_metrics::record(|m| {
                m.borrows += 1;
                m.borrowed += block.len();
            });

            consume(id, &block)?;
        }

        Ok(())
    }
}

pub(super) struct Construction<F, T>(pub Tensor<F, T>);

impl<F: TensorFileEntry<T>, T: TensorElement> Construction<F, T> {
    pub fn stage(&self, request: BatchRequest, values: Vec<T>) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let tensor = &self.0;
            let Storage::Dense(dense) = tensor.storage.as_ref() else {
                return Err(Error::InvalidLayout(
                    "dense construction requires dense storage".into(),
                ));
            };

            let updates =
                crate::storage::plan_updates(&tensor.storage_geometry(), None, &request, values)?;
            #[cfg(test)]
            super::copy_metrics::record(|m| m.groups += updates.len());

            for (id, updates) in updates {
                let file = dense.blocks.read().await.get_file(&id.to_string()).cloned();

                if let Some(file) = file {
                    let mut block = file.write::<Vec<T>>().await?;
                    super::apply_block_updates(&mut block, tensor.block_len(), &updates)?;
                } else {
                    let mut block = tensor.default_block();
                    super::apply_block_updates(&mut block, tensor.block_len(), &updates)?;
                    dense
                        .blocks
                        .write()
                        .await
                        .create_file(id.to_string(), block)
                        .await?;
                }

                #[cfg(test)]
                super::copy_metrics::record(|m| m.block_updates += 1);
            }

            Ok(())
        })
    }

    pub async fn finish(self) -> Result<Tensor<F, T>> {
        self.0.publish_metadata().await?;

        Ok(self.0)
    }
}

fn required_file<FE: FileLoad>(blocks: &Dir<FE>, id: u64) -> Result<FileLock<FE>> {
    blocks.get_file(&id.to_string()).cloned().ok_or_else(|| {
        Error::Io(freqfs::IntegrityError::invalid_data(
            "missing required tensor payload",
        ))
    })
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/dense_storage/test_support.rs"]
pub(super) mod test_support;

#[cfg(test)]
#[path = "../../tests/unit/tensor/dense_storage/tests.rs"]
mod tests;
