//! Sparse storage is one native table of dense logical blocks.

use std::ops::{
    Bound::{Excluded, Included},
    Range,
};

use b_table::{ColumnRange, TableLock, collate::Collator};
use freqfs::DirLock;
use futures::TryStreamExt;
use tokio::sync::RwLock;

use super::{METADATA, SPARSE_INDEX_PAGE_ENTRIES};
use crate::sparse::{PayloadSchema, SparseCell};
use crate::{BoxFuture, Error, Result, StorageGeometry, Tensor, TensorElement, TensorFileEntry};

type Blocks<F, T> = TableLock<PayloadSchema<T>, PayloadSchema<T>, Collator<SparseCell<T>>, F>;

pub(super) struct SparseStorage<F, T> {
    pub blocks: DirLock<F>,
    pub geometry: Box<StorageGeometry>,
    pub gate: RwLock<()>,
    pub(super) values: Blocks<F, T>,
}

impl<F: TensorFileEntry<T>, T: TensorElement> SparseStorage<F, T> {
    pub(super) async fn create(
        dir: &DirLock<F>,
        blocks: DirLock<F>,
        geometry: StorageGeometry,
    ) -> Result<Self> {
        let values = dir.try_write()?.create_dir("values".into())?;
        Ok(Self {
            values: TableLock::create(
                PayloadSchema::new(geometry.block_len()),
                Collator::default(),
                values,
            )
            .await?,
            blocks,
            geometry: Box::new(geometry),
            gate: RwLock::new(()),
        })
    }

    pub(super) fn load(
        dir: &DirLock<F>,
        blocks: DirLock<F>,
        geometry: StorageGeometry,
    ) -> Result<Self> {
        let dir = dir.try_read()?;
        if dir
            .iter()
            .any(|(name, _)| name != "blocks" && name != "values")
        {
            return Err(Error::InvalidLayout(
                "unexpected sparse storage entry".into(),
            ));
        }
        let values = dir
            .get_dir("values")
            .cloned()
            .ok_or_else(|| Error::InvalidLayout("missing sparse values".into()))?;
        Ok(Self {
            values: TableLock::load(
                PayloadSchema::new(geometry.block_len()),
                Collator::default(),
                values,
            )?,
            blocks,
            geometry: Box::new(geometry),
            gate: RwLock::new(()),
        })
    }

    fn validate_payload(&self, id: u64, values: &[T]) -> Result<bool> {
        self.geometry.validate_block(id, values)?;
        Ok(values.iter().any(|v| *v != T::ZERO))
    }

    fn row_payload(&self, id: u64, mut row: b_table::Row<SparseCell<T>>) -> Result<Vec<T>> {
        if row.len() != 2 || row[0] != SparseCell::Key(id) {
            return Err(Error::InvalidLayout("invalid sparse block key".into()));
        }
        let Some(SparseCell::Payload(values)) = row.pop() else {
            return Err(Error::InvalidLayout("invalid sparse payload".into()));
        };
        if !self.validate_payload(id, &values)? {
            return Err(Error::InvalidLayout("empty sparse payload".into()));
        }
        Ok(values)
    }

    // The caller retains the ownership guard across reading and any mutation.
    fn read_row(&self, id: u64) -> BoxFuture<'_, Result<Option<Vec<T>>>> {
        Box::pin(async move {
            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += 1);
            self.geometry.block_bounds(id)?;
            let row = self
                .values
                .read()
                .await
                .get_row(&[SparseCell::Key(id)])
                .await?;
            row.map(|row| self.row_payload(id, row)).transpose()
        })
    }

    // Read only consecutive requested IDs together. Gaps open separate native
    // ranges, so a corrupt payload outside the request is never consumed.
    // The caller owns the native guard and initializes output to zero.
    pub(super) fn read_groups<'a>(
        &'a self,
        groups: &'a crate::storage_read::LogicalGroups,
        output: &'a mut [T],
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut ids = groups.keys().copied().peekable();
            while let Some(first) = ids.next() {
                let mut last = first;
                while let Some(&next) = ids.peek() {
                    if last.checked_add(1) != Some(next) {
                        break;
                    }
                    last = ids.next().expect("peeked block ID");
                }

                if first == last {
                    let block = self.read(first).await?;
                    for run in &groups[&first] {
                        run.scatter(&block, output)?;
                    }
                    continue;
                }

                #[cfg(test)]
                crate::read_metrics::record(|m| {
                    m.logical_payload_reads += (last - first + 1) as usize
                });
                let selection = [(
                    "block".into(),
                    ColumnRange::In((
                        Included(SparseCell::Key(first)),
                        Included(SparseCell::Key(last)),
                    )),
                )]
                .into_iter()
                .collect();
                let table = self.values.read().await;
                let mut rows = table.rows(selection, &[], false, None).await?;
                let mut previous = None;
                while let Some(row) = rows.try_next().await? {
                    let Some(SparseCell::Key(id)) = row.first() else {
                        return Err(Error::InvalidLayout("invalid sparse block key".into()));
                    };
                    let id = *id;
                    if id < first || id > last || previous.is_some_and(|prior| prior >= id) {
                        return Err(Error::InvalidLayout("invalid sparse block order".into()));
                    }
                    let block = self.row_payload(id, row)?;
                    for run in &groups[&id] {
                        run.scatter(&block, output)?;
                    }
                    previous = Some(id);
                }
            }
            Ok(())
        })
    }

    pub fn read(&self, id: u64) -> BoxFuture<'_, Result<Vec<T>>> {
        Box::pin(async move {
            Ok(self
                .read_row(id)
                .await?
                .unwrap_or_else(|| vec![T::ZERO; self.geometry.block_len()]))
        })
    }

    async fn put(&self, id: u64, values: Vec<T>) -> Result<bool> {
        let inserted = self
            .values
            .write()
            .await
            .upsert(vec![SparseCell::Key(id)], vec![SparseCell::Payload(values)])
            .await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| m.payload_writes += 1);
        Ok(inserted)
    }

    // Only unpublished construction appends rows whose block IDs increase.
    async fn append<S>(&self, rows: S) -> Result<()>
    where
        S: futures::Stream<Item = std::io::Result<(Vec<SparseCell<T>>, Vec<SparseCell<T>>)>> + Send,
    {
        let _written = self.values.write().await.upsert_sorted(rows).await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| {
            m.constructed_blocks += _written as usize;
            m.payload_writes += _written as usize;
        });
        Ok(())
    }

    async fn remove(&self, id: u64) -> Result<()> {
        self.values
            .write()
            .await
            .delete_row(&[SparseCell::Key(id)])
            .await?;
        Ok(())
    }

    async fn replace_block(
        &self,
        id: u64,
        values: Vec<T>,
        nonzero: bool,
        present: bool,
    ) -> Result<()> {
        #[cfg(test)]
        super::copy_metrics::record(|m| m.replaced_blocks += 1);
        if !nonzero && !present {
            return Ok(());
        }

        if nonzero {
            self.put(id, values).await?;
        } else {
            self.remove(id).await?;
        }
        Ok(())
    }

    pub fn replace(&self, id: u64, values: Vec<T>) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            self.geometry.block_bounds(id)?;
            if values.len() != self.geometry.block_len() {
                return Err(Error::InvalidLayout(
                    "invalid replacement block length".into(),
                ));
            }
            let _guard = self.gate.write().await;

            let nonzero = self.validate_payload(id, &values)?;
            let present = self.read_row(id).await?.is_some();
            self.replace_block(id, values, nonzero, present).await
        })
    }

    pub fn update<'a>(&'a self, id: u64, updates: &'a [(usize, T)]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.geometry.block_bounds(id)?;
            let _guard = self.gate.write().await;

            let previous = self.read_row(id).await?;
            let present = previous.is_some();
            let mut values = previous.unwrap_or_else(|| vec![T::ZERO; self.geometry.block_len()]);
            super::apply_block_updates(&mut values, self.geometry.block_len(), updates)?;
            let nonzero = self.validate_payload(id, &values)?;
            self.replace_block(id, values, nonzero, present).await
        })
    }

    pub async fn sync(&self) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        let _guard = self.gate.write().await;

        self.sync_contents().await
    }

    pub async fn sync_all(&self, directory: &DirLock<F>) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        let _guard = self.gate.write().await;

        self.sync_contents().await?;
        directory.sync_all().await?;
        Ok(())
    }

    async fn sync_contents(&self) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        self.values.sync().await?;
        self.blocks.sync().await?;
        Ok(())
    }

    pub fn validate(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let _guard = self.gate.read().await;

            self.values.validate().await?;
            {
                let blocks = self.blocks.read().await;
                if blocks
                    .iter()
                    .any(|(name, entry)| name != METADATA || !entry.is_file())
                {
                    return Err(Error::InvalidLayout(
                        "unexpected sparse payload file".into(),
                    ));
                }
            }
            let table = self.values.read().await;
            let mut rows = table.rows(Default::default(), &[], false, None).await?;
            let mut previous = None;
            while let Some(row) = rows.try_next().await? {
                let [SparseCell::Key(id), SparseCell::Payload(values)] = row.as_slice() else {
                    return Err(Error::InvalidLayout("invalid sparse block row".into()));
                };
                if previous.is_some_and(|last| last >= *id) {
                    return Err(Error::InvalidLayout("invalid sparse block order".into()));
                }
                if !self.validate_payload(*id, values)? {
                    return Err(Error::InvalidLayout("empty sparse payload".into()));
                }
                previous = Some(*id);
            }
            Ok(())
        })
    }

    pub(super) async fn occupied_page(&self, range: Range<u64>) -> Result<Vec<u64>> {
        if range.start > range.end || range.end > self.geometry.block_count() {
            return Err(Error::InvalidCoord(
                "occupied block range out of bounds".into(),
            ));
        }
        let _guard = self.gate.read().await;

        if range.is_empty() {
            return Ok(Vec::new());
        }

        let selection = [(
            "block".into(),
            ColumnRange::In((
                Included(SparseCell::Key(range.start)),
                Excluded(SparseCell::Key(range.end)),
            )),
        )]
        .into_iter()
        .collect();
        let table = self.values.read().await;
        let mut rows = table
            .rows(selection, &[], false, Some(&["block".into()]))
            .await?;
        let mut keys = Vec::new();
        while let Some(row) = rows.try_next().await? {
            let [SparseCell::Key(id)] = row.as_slice() else {
                return Err(Error::InvalidLayout("invalid sparse block key".into()));
            };
            if !range.contains(id) || keys.last().is_some_and(|last| *last >= *id) {
                return Err(Error::InvalidLayout("invalid occupied block order".into()));
            }
            #[cfg(test)]
            crate::read_metrics::record(|m| m.index_entries += 1);
            // Inspect one extra key so a duplicate at a page boundary cannot be skipped
            // by exclusive continuation. The next page still owns that extra key.
            if keys.len() == SPARSE_INDEX_PAGE_ENTRIES {
                break;
            }
            keys.push(*id);
        }
        Ok(keys)
    }
}

// Sparse-axis chunks may interleave in row-major input. Contiguous construction
// needs one buffer; interleaved construction uses destination rows to avoid keeping
// every incomplete block in memory. Scalar-sparse storage alone needs no such split.
pub(super) struct Construction<F, T> {
    tensor: Tensor<F, T>,
    current: Option<(Option<u64>, Vec<T>)>,
}

impl<F: TensorFileEntry<T>, T: TensorElement> Construction<F, T> {
    pub fn new(tensor: Tensor<F, T>) -> Self {
        let geometry = &tensor
            .storage
            .sparse()
            .expect("sparse construction")
            .geometry;
        let shape = geometry.schema().shape();
        let block = geometry.block_shape();
        let contiguous = block
            .iter()
            .position(|&n| n > 1)
            .is_none_or(|first| block[first + 1..] == shape[first + 1..]);
        let current = contiguous.then(|| (None, vec![T::ZERO; geometry.block_len()]));
        Self { tensor, current }
    }

    pub fn stage<'a>(
        &'a mut self,
        coords: &'a [Vec<u64>],
        values: Vec<T>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if coords.len() != values.len() || values.len() > crate::expression::MAX_BATCH_ELEMENTS
            {
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
            let _guard = owner.gate.write().await;

            if let Some((current, _)) = &self.current {
                let mut previous = *current;
                // Validate the bounded source batch before native ingestion can mutate.
                // No block payloads or request-sized collection are constructed here.
                for coord in coords {
                    let (id, _) = owner.geometry.block_position(coord)?;
                    if previous.is_some_and(|last| id < last) {
                        return Err(Error::InvalidCoord(
                            "construction blocks out of order".into(),
                        ));
                    }
                    previous = Some(id);
                }
            }
            if let Some((current, block)) = &mut self.current {
                let rows = futures::stream::try_unfold(
                    (coords.iter().zip(values), current, block),
                    |(mut input, current, block)| async move {
                        for (coord, value) in input.by_ref() {
                            let (id, offset) = owner
                                .geometry
                                .block_position(coord)
                                .expect("validated construction coordinate");
                            if let Some(previous) = current.filter(|previous| id != *previous) {
                                let complete = std::mem::replace(
                                    block,
                                    vec![T::ZERO; owner.geometry.block_len()],
                                );
                                *current = Some(id);
                                block[offset] = value;
                                if complete.iter().any(|value| *value != T::ZERO) {
                                    let row = (
                                        vec![SparseCell::Key(previous)],
                                        vec![SparseCell::Payload(complete)],
                                    );
                                    return Ok::<_, std::io::Error>(Some((
                                        row,
                                        (input, current, block),
                                    )));
                                }
                                continue;
                            }
                            *current = Some(id);
                            block[offset] = value;
                        }
                        Ok(None)
                    },
                );
                owner.append(rows).await?;
            } else {
                let request = crate::request::BatchRequest::explicit(coords.to_vec())?;
                let groups = crate::storage::plan_updates(&owner.geometry, None, &request, values)?;
                for (id, updates) in groups {
                    let mut values = owner
                        .read_row(id)
                        .await?
                        .unwrap_or_else(|| vec![T::ZERO; owner.geometry.block_len()]);
                    super::apply_block_updates(&mut values, owner.geometry.block_len(), &updates)?;
                    let present = owner.validate_payload(id, &values)?;
                    if present {
                        let _inserted = owner.put(id, values).await?;
                        #[cfg(test)]
                        super::copy_metrics::record(|m| {
                            m.constructed_blocks += usize::from(_inserted)
                        });
                    } else {
                        owner.remove(id).await?;
                    }
                }
            }
            Ok(())
        })
    }

    pub fn finish(mut self) -> BoxFuture<'static, Result<Tensor<F, T>>> {
        Box::pin(async move {
            {
                let owner = self.tensor.storage.sparse().expect("sparse construction");
                let _guard = owner.gate.write().await;

                match self.current.take() {
                    Some((Some(id), values)) if owner.validate_payload(id, &values)? => {
                        let row = (vec![SparseCell::Key(id)], vec![SparseCell::Payload(values)]);
                        owner.append(futures::stream::iter([Ok(row)])).await?;
                    }
                    _ => {}
                }

                self.tensor.persist_metadata().await?;
            }
            Ok(self.tensor)
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tensor/sparse_storage/validation_tests.rs"]
mod validation_tests;
