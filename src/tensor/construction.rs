//! Unpublished construction owns first-use allocation; published writes never do.

use futures::{StreamExt, TryStreamExt};

use super::{Tensor, TensorElement, TensorFileEntry, sparse_storage};
use crate::request::BatchRequest;
use crate::{BoxFuture, Error, Result, TensorSource};

pub(super) struct Dense<F, T>(pub Tensor<F, T>);

impl<F: TensorFileEntry<T>, T: TensorElement> Dense<F, T> {
    pub fn stage(&self, request: BatchRequest, values: Vec<T>) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let tensor = &self.0;
            let updates =
                crate::storage::plan_updates(&tensor.storage_geometry(), None, &request, values)?;
            #[cfg(test)]
            super::copy_metrics::record(|m| m.groups += updates.len());

            for (id, updates) in updates {
                let file = tensor
                    .storage
                    .blocks()
                    .read()
                    .await
                    .get_file(&id.to_string())
                    .cloned();
                if let Some(file) = file {
                    let mut block = file.write::<Vec<T>>(0).await?;
                    super::apply_block_updates(&mut block, tensor.block_len(), &updates)?;
                } else {
                    let mut block = tensor.default_block();
                    super::apply_block_updates(&mut block, tensor.block_len(), &updates)?;
                    let bound = super::block_allocation::<F, T>(&block);
                    tensor
                        .storage
                        .blocks()
                        .write()
                        .await
                        .create_file(id.to_string(), block, bound)
                        .await?;
                }

                #[cfg(test)]
                super::copy_metrics::record(|m| m.block_updates += 1);
            }

            Ok(())
        })
    }

    pub async fn finish(self) -> Result<Tensor<F, T>> {
        self.0.persist_metadata().await?;
        Ok(self.0)
    }
}

pub(super) async fn sparse<F, T, S, E>(
    tensor: Tensor<F, T>,
    entries: S,
) -> std::result::Result<Tensor<F, T>, E>
where
    F: TensorFileEntry<T>,
    T: TensorElement,
    S: futures::Stream<Item = std::result::Result<(Vec<u64>, T), E>> + Send,
    E: From<Error>,
{
    let mut output = sparse_storage::Construction::new(tensor);
    let entries = entries.fuse();
    futures::pin_mut!(entries);
    let mut previous = None;

    loop {
        let mut coords = Vec::new();
        let mut values = Vec::new();

        while values.len() < crate::expression::MAX_BATCH_ELEMENTS {
            let Some((coord, value)) = entries.try_next().await? else {
                break;
            };

            if previous.as_ref().is_some_and(|last| last >= &coord) {
                return Err(Error::InvalidCoord(
                    "sparse entries must be ordered and unique".into(),
                )
                .into());
            }
            previous = Some(coord.clone());
            coords.push(coord);
            values.push(value);
        }

        if values.is_empty() {
            break;
        }
        output.stage(&coords, values).await?;
    }

    Ok(output.finish().await?)
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/construction/tests.rs"]
mod tests;
