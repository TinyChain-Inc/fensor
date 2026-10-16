//! Unpublished construction owns first-use allocation; published writes never do.

use futures::{StreamExt, TryStreamExt};

use super::{Tensor, TensorElement, TensorFileEntry, sparse_storage};
use crate::Error;

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
