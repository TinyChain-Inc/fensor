//! Dense storage keeps each logical block in a required typed payload file.

use freqfs::{Dir, DirLock, FileLoad, FileLock, FileReadGuardOwned};
use futures::{StreamExt, TryStreamExt, stream};

use super::{
    Storage, Tensor, TensorElement, TensorFileEntry, apply_block_updates, validate_stored_length,
};
use crate::request::BatchRequest;
use crate::schema::StorageSchema;
use crate::{BoxFuture, Error, Result, TensorSource};

pub(super) struct DenseStorage<FE> {
    blocks: DirLock<FE>,
    storage_schema: StorageSchema,
}

// Boxed storage futures bound stack usage in deeply composed consumers.
impl<FE> DenseStorage<FE> {
    pub(super) fn new(blocks: DirLock<FE>, storage_schema: StorageSchema) -> Self {
        Self {
            blocks,
            storage_schema,
        }
    }

    pub(super) fn storage_schema(&self) -> &StorageSchema {
        &self.storage_schema
    }

    pub(super) async fn initialize<T>(&self) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        for id in 0..self.storage_schema.shape.iter().product::<u64>() {
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
        for id in 0..self.storage_schema.shape.iter().product::<u64>() {
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
        self.storage_schema
            .block_schema
            .shape
            .iter()
            .product::<u64>() as usize
    }

    fn validate_id(&self, id: u64) -> Result<()> {
        if id >= self.storage_schema.shape.iter().product::<u64>() {
            return Err(Error::InvalidCoord("logical block out of bounds".into()));
        }

        Ok(())
    }

    pub(super) async fn replace<T>(&self, id: u64, block: Vec<T>) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        self.validate_id(id)?;
        if block.len() != self.block_len() {
            return Err(Error::InvalidLayout(
                "invalid replacement block length".into(),
            ));
        }

        let file = required_file(&*self.blocks.read().await, id)?;
        let mut stored = file.write::<Vec<T>>().await?;
        validate_stored_length(&stored, self.block_len())?;
        *stored = block;
        Ok(())
    }

    pub(super) async fn update<T>(&self, id: u64, updates: &[(usize, T)]) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        let file = required_file(&*self.blocks.read().await, id)?;
        let mut block = file.write::<Vec<T>>().await?;
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
            let block = file.into_read::<Vec<T>>().await?;
            validate_stored_length(&block, self.block_len())?;
            Ok(block)
        })
    }

    pub(super) async fn read_groups<T>(
        &self,
        groups: crate::storage_read::LogicalGroups,
        values: &mut [T],
    ) -> Result<()>
    where
        FE: TensorFileEntry<T>,
        T: TensorElement,
    {
        // Ordinary reads retain their guards until consumed. Cache admission may
        // fail when these reads or independent consumers pin too much capacity.
        let reads = stream::iter(groups)
            .map(|(id, positions)| async move {
                self.read::<T>(id).await.map(|block| (positions, block))
            })
            .buffered(num_cpus::get().max(1));
        let mut reads = std::pin::pin!(reads);

        while let Some((positions, block)) = reads.try_next().await? {
            #[cfg(test)]
            crate::read_metrics::record(|m| {
                m.borrows += 1;
                m.borrowed += block.len();
            });

            for run in positions {
                run.scatter(&block, values)?;
            }
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
    blocks
        .get_file(&id.to_string())
        .cloned()
        .ok_or_else(|| Error::InvalidLayout("missing required tensor payload".into()))
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/dense_storage/test_support.rs"]
pub(super) mod test_support;

#[cfg(test)]
#[path = "../../tests/unit/tensor/dense_storage/tests.rs"]
mod tests;
