//! Sparse storage is one native BTree of dense logical blocks.

use std::borrow::Borrow;
use std::ops::Bound::{Excluded, Included};
use std::ops::{ControlFlow, Range};
use std::sync::Arc;

use b_tree::{BTree, BTreeLock, BTreeWriteGuard};
use freqfs::DirLock;

use super::{METADATA, SPARSE_INDEX_PAGE_ENTRIES};
use crate::sparse::{PayloadSchema, SparseCell, SparseCollator};
use crate::{BoxFuture, Error, Result, StorageGeometry, Tensor, TensorElement, TensorFileEntry};

const VALUES: &str = "values";
const PRIMARY: &str = "primary";

type Blocks<F, T> = BTreeLock<PayloadSchema<T>, SparseCollator<T>, F>;

pub struct SparseBlocks<F, T> {
    inner: Arc<BlockData<F, T>>,
}

struct BlockData<F, T> {
    metadata_dir: DirLock<F>,
    values_dir: DirLock<F>,
    geometry: Box<StorageGeometry>,
    blocks: Blocks<F, T>,
}

impl<F, T> SparseBlocks<F, T> {
    pub(super) fn geometry(&self) -> &StorageGeometry {
        &self.inner.geometry
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

    pub fn occupied_blocks(
        &self,
        range: Range<u64>,
    ) -> futures::stream::BoxStream<'static, Result<u64>> {
        use futures::{StreamExt, TryStreamExt};
        futures::stream::try_unfold((self.clone(), range), |(owner, range)| async move {
            let keys = owner.occupied_page(range.clone()).await?;
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
                    PayloadSchema::new(geometry.clone()),
                    SparseCollator::default(),
                    primary,
                )
                .await?,
                values_dir: values,
                metadata_dir,
                geometry: Box::new(geometry),
            }),
        })
    }

    pub(super) fn load(
        dir: &DirLock<F>,
        metadata_dir: DirLock<F>,
        geometry: StorageGeometry,
    ) -> Result<Self> {
        let dir = dir.try_read()?;

        if dir
            .iter()
            .any(|(name, _)| name != super::BLOCKS && name != VALUES)
        {
            return Err(Error::InvalidLayout(
                "unexpected sparse storage entry".into(),
            ));
        }

        let values = dir
            .get_dir(VALUES)
            .cloned()
            .ok_or_else(|| Error::InvalidLayout("missing sparse values".into()))?;

        let primary = {
            let entries = values.try_read()?;
            if entries.len() != 1 {
                return Err(Error::InvalidLayout("unexpected sparse index entry".into()));
            }

            entries
                .get_dir(PRIMARY)
                .cloned()
                .ok_or_else(|| Error::InvalidLayout("missing sparse primary index".into()))?
        };

        Ok(Self {
            inner: Arc::new(BlockData {
                blocks: BTreeLock::load(
                    PayloadSchema::new(geometry.clone()),
                    SparseCollator::default(),
                    primary,
                )?,
                values_dir: values,
                metadata_dir,
                geometry: Box::new(geometry),
            }),
        })
    }

    fn validate_payload(&self, id: u64, values: &[T]) -> Result<bool> {
        self.inner.geometry.validate_block(id, values)?;

        Ok(values.iter().any(|v| *v != T::ZERO))
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

    // Reuse the acquired tree guard, including across a read-modify-write.
    async fn read_row<G>(
        &self,
        table: &BTree<PayloadSchema<T>, SparseCollator<T>, G>,
        id: u64,
    ) -> Result<Option<Vec<T>>>
    where
        G: freqfs::DirDeref<Entry = F> + 'static,
    {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.logical_payload_reads += 1);

        // Entry points validate IDs; grouped reads and construction use validated plans.
        let row = table
            .get(&[SparseCell::Key(id)])
            .await
            .map_err(freqfs::IntegrityError::from_read)?;

        row.map(|row| self.row_payload(id, row)).transpose()
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
        self.inner.geometry.validate_block_ids(ids)?;
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

            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += (last - first + 1) as usize);

            if first == last {
                table
                    .get_with(&[SparseCell::Key(first)], |row| {
                        Ok::<_, std::io::Error>((|| {
                            let block = self.row_values(first, row)?;
                            #[cfg(test)]
                            crate::read_metrics::record(|m| {
                                m.borrows += 1;
                                m.borrowed += block.len();
                            });

                            consume(first, block)
                        })())
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

                    let block = self.row_values(id, row)?;
                    #[cfg(test)]
                    crate::read_metrics::record(|m| {
                        m.borrows += 1;
                        m.borrowed += block.len();
                    });

                    consume(id, block)?;
                    previous = Some(id);
                    Ok(ControlFlow::Continue(()))
                })
                .await?;
        }

        Ok(())
    }

    pub fn read(&self, id: u64) -> BoxFuture<'_, Result<Vec<T>>> {
        Box::pin(async move {
            self.inner.geometry.validate_id(id)?;
            let table = self.inner.blocks.read().await;
            Ok(self.read_row(&table, id).await?.unwrap_or_else(|| {
                vec![T::ZERO; self.inner.geometry.block_len(id).expect("validated block")]
            }))
        })
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
            None => vec![T::ZERO; self.inner.geometry.block_len(id)?],
        };

        super::apply_block_updates(&mut values, self.inner.geometry.block_len(id)?, updates)?;
        let nonzero = self.validate_payload(id, &values)?;
        let inserted = previous.is_none() && nonzero;
        Self::publish_block(table, id, previous.as_deref(), nonzero.then_some(values)).await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| m.payload_writes += usize::from(nonzero));

        Ok(inserted)
    }

    pub fn replace(&self, id: u64, values: Vec<T>) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.inner.geometry.validate_id(id)?;
            if values.len() != self.inner.geometry.block_len(id)? {
                return Err(Error::InvalidLayout(
                    "invalid replacement block length".into(),
                ));
            }

            let mut table = self.inner.blocks.write().await;
            let nonzero = self.validate_payload(id, &values)?;
            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += 1);

            let previous = table
                .get(&[SparseCell::Key(id)])
                .await
                .map_err(freqfs::IntegrityError::from_read)?;

            if let Some(row) = previous.as_deref() {
                self.row_values(id, row)?;
            }

            Self::publish_block(
                &mut table,
                id,
                previous.as_deref(),
                nonzero.then_some(values),
            )
            .await?;

            #[cfg(test)]
            super::copy_metrics::record(|m| {
                m.replaced_blocks += 1;
                m.payload_writes += usize::from(nonzero);
            });

            Ok(())
        })
    }

    pub(super) fn update<'a>(
        &'a self,
        id: u64,
        updates: &'a [(usize, T)],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.inner.geometry.validate_id(id)?;
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

    pub(super) async fn sync(&self) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        self.inner.values_dir.sync().await?;
        self.inner.metadata_dir.sync().await?;
        Ok(())
    }

    pub(super) async fn sync_all(&self, directory: &DirLock<F>) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        self.sync().await?;
        directory.sync_all().await?;
        Ok(())
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

            self.inner.blocks.validate().await?;
            {
                let blocks = self.inner.metadata_dir.read().await;
                if blocks
                    .iter()
                    .any(|(name, entry)| name != METADATA || !entry.is_file())
                {
                    return Err(Error::InvalidLayout(
                        "unexpected sparse payload file".into(),
                    ));
                }
            }

            let table = self.inner.blocks.read().await;
            let mut previous = None;
            table
                .visit(b_tree::Range::<SparseCell<T>>::default(), false, |row| {
                    let [SparseCell::Key(id), SparseCell::Payload(values)] = row else {
                        return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block row",
                        )));
                    };

                    if previous.is_some_and(|last| last >= *id) {
                        return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                            "invalid sparse block order",
                        )));
                    }

                    if !self
                        .validate_payload(*id, values)
                        .map_err(|error| Error::Io(freqfs::IntegrityError::invalid_data(error)))?
                    {
                        return Err(Error::Io(freqfs::IntegrityError::invalid_data(
                            "empty sparse payload",
                        )));
                    }

                    previous = Some(*id);
                    Ok(ControlFlow::Continue(()))
                })
                .await?;

            Ok(())
        })
    }

    pub(super) async fn occupied_page(&self, range: Range<u64>) -> Result<Vec<u64>> {
        if range.start > range.end || range.end > self.inner.geometry.block_count() {
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

                // Inspect one extra key so a duplicate at a page boundary cannot be skipped
                // by exclusive continuation. The next page still owns that extra key.
                if keys.len() == SPARSE_INDEX_PAGE_ENTRIES {
                    return Ok(ControlFlow::Break(()));
                }

                keys.push(*id);
                Ok(ControlFlow::Continue(()))
            })
            .await?;

        Ok(keys)
    }
}

// Linear region blocks occur in the same order as canonical input coordinates.
pub(super) struct Construction<F, T> {
    tensor: Tensor<F, T>,
    current: Option<(u64, Vec<T>)>,
}

impl<F: TensorFileEntry<T>, T: TensorElement> Construction<F, T> {
    pub fn new(tensor: Tensor<F, T>) -> Self {
        Self {
            tensor,
            current: None,
        }
    }

    pub fn stage<'a>(
        &'a mut self,
        coords: &'a [Vec<u64>],
        values: Vec<T>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if coords.len() != values.len() || values.len() > crate::MAX_BATCH_ELEMENTS {
                return Err(Error::InvalidLayout(
                    "invalid sparse construction batch".into(),
                ));
            }

            #[cfg(test)]
            super::copy_metrics::record(|m| {
                m.staged_elements += values.len();
                m.max_staging_batch = m.max_staging_batch.max(values.len());
            });

            let owner = self.tensor.storage.sparse().expect("sparse construction");
            let mut previous = self.current.as_ref().map(|(id, _)| *id);
            for coord in coords {
                let (id, _) = owner.inner.geometry.block_position(coord)?;
                if previous.is_some_and(|prior| id < prior) {
                    return Err(Error::InvalidCoord(
                        "construction blocks out of order".into(),
                    ));
                }

                previous = Some(id);
            }

            let mut table = owner.inner.blocks.write().await;
            let current = &mut self.current;
            let mut input = coords.iter().zip(values);
            let rows = std::iter::from_fn(|| {
                for (coord, value) in input.by_ref() {
                    let (id, offset) = owner
                        .inner
                        .geometry
                        .block_position(coord)
                        .expect("validated construction coordinate");

                    if current
                        .as_ref()
                        .is_some_and(|(previous, _)| *previous != id)
                    {
                        let (previous, values) = current
                            .replace((
                                id,
                                vec![
                                    T::ZERO;
                                    owner.inner.geometry.block_len(id).expect("validated block")
                                ],
                            ))
                            .expect("current block");

                        current.as_mut().expect("current block").1[offset] = value;
                        if values.iter().any(|value| *value != T::ZERO) {
                            return Some(Ok(vec![
                                SparseCell::Key(previous),
                                SparseCell::Payload(values),
                            ]));
                        }

                        continue;
                    }

                    if current.is_none() {
                        *current = Some((
                            id,
                            vec![
                                T::ZERO;
                                owner.inner.geometry.block_len(id).expect("validated block")
                            ],
                        ));
                    }

                    current.as_mut().expect("current block").1[offset] = value;
                }

                None
            });

            owner
                .append(&mut table, futures::stream::iter(rows))
                .await?;

            Ok(())
        })
    }

    pub fn finish(mut self) -> BoxFuture<'static, Result<Tensor<F, T>>> {
        Box::pin(async move {
            if let Some((id, values)) = self.current.take() {
                let owner = self.tensor.storage.sparse().expect("sparse construction");
                if owner.validate_payload(id, &values)? {
                    let mut table = owner.inner.blocks.write().await;
                    owner
                        .append(
                            &mut table,
                            futures::stream::iter([Ok(vec![
                                SparseCell::Key(id),
                                SparseCell::Payload(values),
                            ])]),
                        )
                        .await?;
                }
            }

            self.tensor.publish_metadata().await?;
            Ok(self.tensor)
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/sparse_storage/validation_tests.rs"]
mod validation_tests;

#[cfg(test)]
#[path = "../../tests/unit/tensor/sparse_storage/corruption.rs"]
pub(crate) mod corruption;
