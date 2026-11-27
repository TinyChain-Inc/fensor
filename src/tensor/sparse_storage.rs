//! Sparse storage is one native BTree of dense logical blocks.

use std::borrow::Borrow;
use std::ops::Bound::{Excluded, Included};
use std::ops::{ControlFlow, Range};
use std::sync::Arc;

use b_tree::{BTreeLock, BTreeWriteGuard};
use freqfs::DirLock;
use futures::TryStreamExt;

use super::{METADATA, OCCUPIED_BATCH_SIZE};
use crate::sparse::{PayloadSchema, SparseCell, SparseCollator, has_payload};
use crate::{
    BoxFuture, Error, MAX_BLOCK_CAPACITY, Result, StorageGeometry, TensorElement, TensorFileEntry,
};

const VALUES: &str = "values";

const PRIMARY: &str = "primary";

type Blocks<F, T> = BTreeLock<PayloadSchema<T>, SparseCollator<T>, F>;

pub struct SparseBlocks<F, T> {
    inner: Arc<BlockData<F, T>>,
}

struct BlockData<F, T> {
    metadata_dir: DirLock<F>,
    values_dir: DirLock<F>,
    blocks: Blocks<F, T>,
}

impl<F, T> SparseBlocks<F, T> {
    /// The logical block geometry validated by this store.
    pub fn geometry(&self) -> &StorageGeometry {
        self.inner.blocks.schema().geometry()
    }
}

impl<F, T> Clone for SparseBlocks<F, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

type WriteBlocks<F, T> = BTreeWriteGuard<PayloadSchema<T>, SparseCollator<T>, F>;

impl<F: TensorFileEntry<T>, T: TensorElement> SparseBlocks<F, T> {
    /// Strictly load standalone logical blocks using their owner's geometry.
    pub async fn load(dir: DirLock<F>, geometry: StorageGeometry) -> Result<Self> {
        super::validate_tensor_dtype::<T>(geometry.schema.dtype())?;
        let metadata = dir
            .read()
            .await
            .get_dir(super::BLOCKS)
            .cloned()
            .ok_or_else(|| {
                freqfs::IntegrityError::invalid_data("missing sparse blocks directory")
            })?;
        let blocks = Self::load_parts(&dir, metadata, geometry)?;
        blocks.validate().await?;
        Ok(blocks)
    }

    /// Create sparse payload storage using the supplied canonical block geometry.
    /// Absence represents zero; callers separately track replacement membership.
    pub async fn create(dir: DirLock<F>, geometry: StorageGeometry) -> Result<Self> {
        super::validate_tensor_dtype::<T>(geometry.schema().dtype())?;
        let metadata = {
            let mut entries = dir.write().await;
            if !entries.is_empty() {
                return Err(Error::InvalidLayout(
                    "block storage requires empty directory".into(),
                ));
            }

            entries.create_dir(super::BLOCKS.into())?
        };

        Self::create_canonical(&dir, metadata, geometry).await
    }

    /// Stream occupied block IDs in ascending order, releasing index guards before delivery.
    pub fn occupied_blocks(
        &self,
        range: Range<u64>,
    ) -> futures::stream::BoxStream<'static, Result<u64>> {
        use futures::StreamExt;

        futures::stream::try_unfold((self.clone(), range), |(owner, range)| async move {
            let keys = owner.occupied_batch(range.clone()).await?;
            let Some(&last) = keys.last() else {
                return Ok::<_, Error>(None);
            };

            Ok(Some((
                futures::stream::iter(keys.into_iter().map(Ok)),
                (owner, last + 1..range.end),
            )))
        })
        .try_flatten()
        .boxed()
    }

    pub(super) async fn clear(&self) -> Result<()> {
        let mut occupied = self.occupied_blocks(0..self.geometry().block_count());
        while let Some(id) = occupied.try_next().await? {
            self.replace(id, None).await?;
        }

        Ok(())
    }

    pub(super) async fn create_canonical(
        dir: &DirLock<F>,
        metadata_dir: DirLock<F>,
        geometry: StorageGeometry,
    ) -> Result<Self> {
        let values = dir.try_write()?.create_dir(VALUES.into())?;
        let primary = values.try_write()?.create_dir(PRIMARY.into())?;

        Ok(Self {
            inner: Arc::new(BlockData {
                blocks: BTreeLock::create(
                    PayloadSchema::new(geometry),
                    SparseCollator::default(),
                    primary,
                )
                .await?,
                values_dir: values,
                metadata_dir,
            }),
        })
    }

    fn load_parts(
        dir: &DirLock<F>,
        metadata_dir: DirLock<F>,
        geometry: StorageGeometry,
    ) -> Result<Self> {
        let dir = dir.try_read()?;

        if dir
            .iter()
            .any(|(name, _)| name != super::BLOCKS && name != VALUES)
        {
            return Err(
                freqfs::IntegrityError::invalid_data("unexpected sparse storage entry").into(),
            );
        }

        let values = dir.get_dir(VALUES).cloned().ok_or_else(|| {
            freqfs::IntegrityError::invalid_data("missing sparse values directory")
        })?;

        let primary = {
            let entries = values.try_read()?;
            let primary = entries.get_dir(PRIMARY).cloned().ok_or_else(|| {
                freqfs::IntegrityError::invalid_data("missing sparse primary index")
            })?;

            if entries.len() != 1 {
                return Err(
                    freqfs::IntegrityError::invalid_data("unexpected sparse index entry").into(),
                );
            }

            primary
        };

        Ok(Self {
            inner: Arc::new(BlockData {
                blocks: BTreeLock::load(
                    PayloadSchema::new(geometry),
                    SparseCollator::default(),
                    primary,
                )?,
                values_dir: values,
                metadata_dir,
            }),
        })
    }

    fn validate_payload(&self, id: u64, values: &[T]) -> Result<bool> {
        self.geometry().validate_block(id, values)?;

        Ok(has_payload(self.geometry().layout(), values))
    }

    fn row_values<'a, V: Borrow<SparseCell<T>>>(&self, id: u64, row: &'a [V]) -> Result<&'a [T]> {
        let [key, payload] = row else {
            return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                "invalid sparse block key",
            )));
        };

        if key.borrow() != &SparseCell::Key(id) {
            return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                "invalid sparse block key",
            )));
        }

        let SparseCell::Payload(values) = payload.borrow() else {
            return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                "invalid sparse payload",
            )));
        };

        if !self
            .validate_payload(id, values)
            .map_err(|error| Error::Io(freqfs::IntegrityError::invalid_data(error)))?
        {
            return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                "empty sparse payload",
            )));
        }

        Ok(values)
    }

    fn row_payload(&self, id: u64, mut row: b_tree::Key<SparseCell<T>>) -> Result<Vec<T>> {
        self.row_values(id, &row)?;
        match row.pop() {
            Some(SparseCell::Payload(values)) => Ok(values),
            _ => Err(Error::Io(freqfs::IntegrityError::invalid_data(
                "invalid sparse payload",
            ))),
        }
    }

    // Read only consecutive requested IDs together. Gaps open separate native
    // ranges, so a corrupt payload outside the request is never consumed.
    // One read guard keeps all groups coherent; absent rows skip consumption.
    /// Visit a bounded, sorted, unique selection of in-range logical block IDs.
    /// Validate the complete selection before locking or invoking callbacks.
    /// Absent blocks skip the callback; borrowed payloads cannot escape it.
    pub async fn visit_blocks<C>(&self, ids: &[u64], mut consume: C) -> Result<()>
    where
        C: FnMut(u64, &[T]) -> Result<()> + Send,
    {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.logical_payload_reads += ids.len());

        self.visit_selected(ids, |id, row| {
            let block = self.row_values(id, row)?;
            #[cfg(test)]
            crate::read_metrics::record(|m| {
                m.borrows += 1;
                m.borrowed += block.len();
            });

            consume(id, block)
        })
        .await
    }

    /// Visit occupied IDs in a bounded, sorted, unique block selection.
    /// The synchronous callback runs under the native index read guard.
    pub async fn visit_occupied(
        &self,
        ids: &[u64],
        mut consume: impl FnMut(u64) + Send,
    ) -> Result<()> {
        self.visit_selected(ids, |id, _| {
            consume(id);
            Ok(())
        })
        .await
    }

    async fn visit_selected(
        &self,
        ids: &[u64],
        mut consume: impl FnMut(u64, &[SparseCell<T>]) -> Result<()> + Send,
    ) -> Result<()> {
        self.geometry().validate_block_ids(ids)?;
        if ids.is_empty() {
            return Ok(());
        }

        let table = self.inner.blocks.read().await;
        let mut ids = ids.iter().copied().peekable();

        while let Some(first) = ids.next() {
            let mut last = first;
            while let Some(&next) = ids.peek() {
                if last.checked_add(1) != Some(next) {
                    break;
                }

                last = ids.next().expect("peeked block ID");
            }

            if first == last {
                table
                    .get_with(&[SparseCell::Key(first)], |row| {
                        Ok::<_, std::io::Error>(consume(first, row))
                    })
                    .await
                    .map_err(freqfs::IntegrityError::from_read)?
                    .transpose()?;

                continue;
            }

            let selection = b_tree::Range::with_bounds(
                Vec::new(),
                (
                    Included(SparseCell::Key(first)),
                    Included(SparseCell::Key(last)),
                ),
            );

            let mut previous = None;
            table
                .visit(selection, false, |row| {
                    let Some(SparseCell::Key(id)) = row.first() else {
                        return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block key",
                        )));
                    };

                    let id = *id;
                    if id < first || id > last || previous.is_some_and(|prior| prior >= id) {
                        return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block order",
                        )));
                    }

                    consume(id, row)?;
                    previous = Some(id);
                    Ok(ControlFlow::Continue(()))
                })
                .await?;
        }

        Ok(())
    }

    /// Read the stored payload for a logical block, if present.
    pub fn read(&self, id: u64) -> BoxFuture<'_, Result<Option<Vec<T>>>> {
        Box::pin(async move {
            self.geometry().validate_id(id)?;
            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += 1);

            let row = self
                .inner
                .blocks
                .read()
                .await
                .get(&[SparseCell::Key(id)])
                .await
                .map_err(freqfs::IntegrityError::from_read)?;

            row.map(|row| self.row_payload(id, row)).transpose()
        })
    }

    /// Fill an empty, unpublished store. Canonical zero leaves it empty.
    pub async fn fill_empty(&self, value: T) -> Result<()> {
        let mut table = self.inner.blocks.write().await;
        if !table.is_empty(b_tree::Range::default()).await? {
            return Err(Error::InvalidLayout(
                "fill requires empty block storage".into(),
            ));
        }

        let geometry = self.geometry();
        if !has_payload(geometry.layout(), std::slice::from_ref(&value)) {
            return Ok(());
        }

        let rows = (0..geometry.block_count()).map(|id| {
            Ok(vec![
                SparseCell::Key(id),
                SparseCell::Payload(vec![
                    value;
                    geometry.block_len(id).map_err(std::io::Error::other)?
                ]),
            ])
        });

        self.append(&mut table, futures::stream::iter(rows)).await
    }

    /// Initialize empty storage from blocks with strictly increasing IDs.
    /// The caller exclusively owns the unpublished directory. Errors leave any
    /// already appended blocks in that directory for the caller to discard.
    /// Source errors are returned unchanged.
    pub async fn append_sorted<E: From<Error>>(
        &mut self,
        blocks: impl futures::Stream<Item = std::result::Result<(u64, Vec<T>), E>> + Send,
    ) -> std::result::Result<(), E> {
        if Arc::get_mut(&mut self.inner).is_none()
            || !self
                .inner
                .blocks
                .read()
                .await
                .is_empty(b_tree::Range::default())
                .await
                .map_err(Error::from)?
        {
            return Err(Error::InvalidLayout(
                "ordered initialization requires unique empty block storage".into(),
            )
            .into());
        }

        futures::pin_mut!(blocks);
        let mut previous = None;
        let mut rows = Vec::new();
        let mut elements = 0;
        while let Some((id, values)) = blocks.try_next().await? {
            tokio::task::coop::consume_budget().await;
            if previous.is_some_and(|prior| prior >= id) {
                return Err(Error::InvalidCoord(
                    "ordered initialization requires increasing block IDs".into(),
                )
                .into());
            }

            previous = Some(id);
            if !self.validate_payload(id, &values)? {
                continue;
            }

            // Bound the batch by one maximum-sized block's worth of values.
            // Poll the source only after releasing the destination tree guard.
            if elements + values.len() > MAX_BLOCK_CAPACITY {
                let mut table = self.inner.blocks.write().await;
                self.append(&mut table, futures::stream::iter(rows.drain(..).map(Ok)))
                    .await?;
                elements = 0;
            }

            elements += values.len();
            rows.push(vec![SparseCell::Key(id), SparseCell::Payload(values)]);
        }

        if !rows.is_empty() {
            let mut table = self.inner.blocks.write().await;
            self.append(&mut table, futures::stream::iter(rows.into_iter().map(Ok)))
                .await?;
        }

        Ok(())
    }

    // Only unpublished construction appends rows whose block IDs increase.
    async fn append<S>(&self, table: &mut WriteBlocks<F, T>, rows: S) -> Result<()>
    where
        S: futures::Stream<Item = std::io::Result<Vec<SparseCell<T>>>> + Send,
    {
        let _written = table.insert_sorted(rows).await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| {
            m.constructed_blocks += _written as usize;
            m.payload_writes += _written as usize;
        });

        Ok(())
    }

    async fn publish_block(
        table: &mut WriteBlocks<F, T>,
        id: u64,
        previous: Option<&[SparseCell<T>]>,
        replacement: Option<Vec<T>>,
    ) -> Result<()> {
        match (previous, replacement) {
            (Some(previous), Some(values)) => {
                if !table
                    .replace(
                        previous,
                        vec![SparseCell::Key(id), SparseCell::Payload(values)],
                    )
                    .await?
                {
                    return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                        "missing sparse block row",
                    )));
                }
            }
            (Some(previous), None) => {
                if !table.delete(previous).await? {
                    return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                        "missing sparse block row",
                    )));
                }
            }
            (None, Some(values)) => {
                table
                    .insert(vec![SparseCell::Key(id), SparseCell::Payload(values)])
                    .await?;
            }
            (None, None) => {}
        }

        Ok(())
    }

    async fn update_block(
        &self,
        table: &mut WriteBlocks<F, T>,
        id: u64,
        updates: &[(usize, T)],
    ) -> Result<bool> {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.logical_payload_reads += 1);

        let previous = table
            .get(&[SparseCell::Key(id)])
            .await
            .map_err(freqfs::IntegrityError::from_read)?;

        let mut values = match previous.as_deref() {
            Some(row) => self.row_values(id, row)?.to_vec(),
            None => vec![T::ZERO; self.geometry().block_len(id)?],
        };

        super::apply_block_updates(&mut values, self.geometry().block_len(id)?, updates)?;
        let nonzero = self.validate_payload(id, &values)?;
        let inserted = previous.is_none() && nonzero;
        Self::publish_block(table, id, previous.as_deref(), nonzero.then_some(values)).await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| m.payload_writes += usize::from(nonzero));

        Ok(inserted)
    }

    /// Replace one block and report whether its payload is stored. Passing `None` removes it.
    pub fn replace(&self, id: u64, values: Option<Vec<T>>) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async move {
            let replacement = match values {
                Some(values) => self.validate_payload(id, &values)?.then_some(values),
                None => {
                    self.geometry().validate_id(id)?;
                    None
                }
            };

            let mut table = self.inner.blocks.write().await;
            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += 1);

            let previous = table
                .get(&[SparseCell::Key(id)])
                .await
                .map_err(freqfs::IntegrityError::from_read)?;

            if let Some(row) = previous.as_deref() {
                self.row_values(id, row)?;
            }

            let has_payload = replacement.is_some();

            Self::publish_block(&mut table, id, previous.as_deref(), replacement).await?;

            #[cfg(test)]
            super::copy_metrics::record(|m| {
                m.replaced_blocks += 1;
                m.payload_writes += usize::from(has_payload);
            });

            Ok(has_payload)
        })
    }

    pub(super) fn update<'a>(
        &'a self,
        id: u64,
        updates: &'a [(usize, T)],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.geometry().validate_id(id)?;
            let mut table = self.inner.blocks.write().await;

            self.update_block(&mut table, id, updates).await?;
            #[cfg(test)]
            super::copy_metrics::record(|m| m.replaced_blocks += 1);

            Ok(())
        })
    }

    pub(super) async fn publish_metadata(&self, metadata: crate::TensorMetadata<T>) -> Result<()> {
        super::create_metadata_file(&self.inner.metadata_dir, metadata).await
    }

    pub(super) fn validate(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            {
                let entries = self.inner.values_dir.read().await;
                if entries.len() != 1 || entries.get_dir(PRIMARY).is_none() {
                    return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                        "invalid sparse index directory",
                    )));
                }
            }

            {
                let blocks = self.inner.metadata_dir.read().await;
                if blocks
                    .iter()
                    .any(|(name, entry)| name != METADATA || !entry.is_file())
                {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "unexpected sparse payload file",
                    )
                    .into());
                }
            }

            let mut previous = None;
            self.inner
                .blocks
                .validate_with(|row| {
                    let [SparseCell::Key(id), _] = row else {
                        return Err(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block row",
                        ));
                    };

                    if previous.is_some_and(|last| last >= *id) {
                        return Err(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block order",
                        ));
                    }

                    previous = Some(*id);
                    Ok(())
                })
                .await?;

            Ok(())
        })
    }

    // Release the index guard before delivering IDs to consumers that read payloads.
    async fn occupied_batch(&self, range: Range<u64>) -> Result<Vec<u64>> {
        if range.start > range.end || range.end > self.geometry().block_count() {
            return Err(Error::InvalidCoord(
                "occupied block range out of bounds".into(),
            ));
        }

        if range.is_empty() {
            return Ok(Vec::new());
        }

        let selection = b_tree::Range::with_bounds(
            Vec::new(),
            (
                Included(SparseCell::Key(range.start)),
                Excluded(SparseCell::Key(range.end)),
            ),
        );

        let table = self.inner.blocks.read().await;
        let mut keys = Vec::new();
        table
            .visit(selection, false, |row| {
                let [SparseCell::Key(id), _] = row else {
                    return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                        "invalid sparse block key",
                    )));
                };

                if !range.contains(id) || keys.last().is_some_and(|last| *last >= *id) {
                    return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                        "invalid occupied block order",
                    )));
                }

                #[cfg(test)]
                crate::read_metrics::record(|m| m.index_entries += 1);

                // Check the next key for duplicates before resuming above the last buffered ID.
                if keys.len() == OCCUPIED_BATCH_SIZE {
                    return Ok(ControlFlow::Break(()));
                }

                keys.push(*id);
                Ok(ControlFlow::Continue(()))
            })
            .await?;

        Ok(keys)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/sparse_storage/validation_tests.rs"]
mod validation_tests;

#[cfg(test)]
#[path = "../../tests/unit/tensor/sparse_storage/corruption.rs"]
pub(crate) mod corruption;
