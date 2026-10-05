//! Sparse blocks share typed table pages; dense payloads remain bounded files.

use std::sync::atomic::{AtomicBool, Ordering};

use b_table::{TableLock, collate::Collator};
use freqfs::DirLock;
use futures::TryStreamExt;
use tokio::sync::RwLock;

use super::{INDEX, METADATA, SPARSE_INDEX_PAGE_ENTRIES};
use crate::schema::{SparseIndexSchema, SparseTableSchema};
use crate::sparse::{PayloadSchema, SparseCell};
use crate::traits::BoxFuture;
use crate::{Error, Result, Tensor, TensorElement, TensorFileEntry};

const ROWS: u64 = 1;
const DENSE: u64 = 2;
const MIN_DENSE_BYTES: usize = 1024;
// Fixed packing cost estimate, independent of serialized row width.
const ROW_OVERHEAD_BYTES: usize = 24;

type SparseIndex<F> = TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, F>;
type Payloads<F, T> = TableLock<PayloadSchema<T>, PayloadSchema<T>, Collator<SparseCell<T>>, F>;

pub(super) struct SparseStorage<F, T> {
    pub blocks: DirLock<F>,
    pub index: SparseIndex<F>,
    pub geometry: Box<crate::StorageGeometry>,
    pub gate: RwLock<()>,
    invalid: AtomicBool,
    pub descriptors: SparseIndex<F>,
    pub values: Payloads<F, T>,
}

// One bounded analysis shared by validation, construction, and replacement.
struct BlockAnalysis {
    start: u64,
    occupied: Vec<bool>,
    nnz: usize,
}

struct Mutation<'a>(&'a AtomicBool, bool);

impl Drop for Mutation<'_> {
    fn drop(&mut self) {
        if !self.1 {
            self.0.store(true, Ordering::Release);
        }
    }
}

impl<F: TensorFileEntry<T>, T: TensorElement> SparseStorage<F, T> {
    pub(super) async fn create(
        dir: &DirLock<F>,
        blocks: DirLock<F>,
        geometry: crate::StorageGeometry,
    ) -> Result<Self> {
        let (index, descriptors, values) = {
            let mut dir = dir.try_write()?;
            (
                dir.create_dir(INDEX.into())?,
                dir.create_dir("descriptors".into())?,
                dir.create_dir("values".into())?,
            )
        };

        Ok(Self {
            blocks,
            index: TableLock::create(SparseTableSchema::default(), Collator::default(), index)
                .await?,
            geometry: Box::new(geometry),
            gate: RwLock::new(()),
            invalid: AtomicBool::new(false),
            descriptors: TableLock::create(
                SparseTableSchema::descriptors(),
                Collator::default(),
                descriptors,
            )
            .await?,
            values: TableLock::create(PayloadSchema::default(), Collator::default(), values)
                .await?,
        })
    }

    pub(super) fn load(
        dir: &DirLock<F>,
        blocks: DirLock<F>,
        geometry: crate::StorageGeometry,
    ) -> Result<Self> {
        let (index, descriptors, values) = {
            let dir = dir.try_read()?;
            (
                dir.get_dir(INDEX).cloned().ok_or_else(|| {
                    Error::InvalidSchema("sparse tensor missing index directory".into())
                })?,
                dir.get_dir("descriptors")
                    .cloned()
                    .ok_or_else(|| Error::InvalidLayout("missing sparse descriptors".into()))?,
                dir.get_dir("values")
                    .cloned()
                    .ok_or_else(|| Error::InvalidLayout("missing sparse values".into()))?,
            )
        };

        Ok(Self {
            blocks,
            index: TableLock::load(SparseTableSchema::default(), Collator::default(), index)?,
            geometry: Box::new(geometry),
            gate: RwLock::new(()),
            invalid: AtomicBool::new(false),
            descriptors: TableLock::load(
                SparseTableSchema::descriptors(),
                Collator::default(),
                descriptors,
            )?,
            values: TableLock::load(PayloadSchema::default(), Collator::default(), values)?,
        })
    }

    pub fn healthy(&self) -> Result<()> {
        if self.invalid.load(Ordering::Acquire) {
            Err(Error::InvalidLayout(
                "sparse owner invalidated by interrupted mutation; caller recovery required".into(),
            ))
        } else {
            Ok(())
        }
    }

    async fn descriptor(&self, id: u64) -> Result<Option<u64>> {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.descriptor_lookups += 1);
        let table = self.descriptors.read().await;
        match table.get_row(&[id]).await? {
            Some(row) if row.len() == 2 && matches!(row[1], ROWS | DENSE) => Ok(Some(row[1])),
            Some(_) => Err(Error::InvalidLayout("invalid sparse descriptor".into())),
            None => Ok(None),
        }
    }

    fn prefix(id: u64) -> b_table::Range<String, SparseCell<T>> {
        [(
            "block".into(),
            b_table::ColumnRange::Eq(SparseCell::Key(id)),
        )]
        .into_iter()
        .collect()
    }

    // The caller holds the ownership guard and has checked health.
    pub fn read<'a>(&'a self, id: u64) -> BoxFuture<'a, Result<(Option<u64>, Vec<T>)>> {
        Box::pin(async move {
            #[cfg(test)]
            crate::read_metrics::record(|m| m.logical_payload_reads += 1);
            let g = &self.geometry;
            let bounds = g.block_bounds(id)?;
            let Some(encoding) = self.descriptor(id).await? else {
                return Ok((None, vec![T::ZERO; self.geometry.block_len()]));
            };

            let values = self.read_payload(id, encoding).await?;
            if self.validate_payload(id, &values, &bounds, |_| {})? == 0 {
                return Err(Error::InvalidLayout("empty sparse payload".into()));
            }

            Ok((Some(encoding), values))
        })
    }

    // Caller validates the descriptor; consumers choose validation or occupancy analysis.
    async fn read_payload(&self, id: u64, encoding: u64) -> Result<Vec<T>> {
        let g = &self.geometry;
        let values = if encoding == DENSE {
            let file = super::required_file(&*self.blocks.read().await, id)?;
            let values = file.read::<Vec<T>>().await?;
            super::validate_stored_length(&values, g.block_len())?;
            values.clone()
        } else {
            self.read_rows(id).await?
        };

        Ok(values)
    }

    async fn read_rows(&self, id: u64) -> Result<Vec<T>> {
        let mut values = vec![T::ZERO; self.geometry.block_len()];
        let table = self.values.read().await;
        let mut rows = table.rows(Self::prefix(id), &[], false, None).await?;
        let mut previous = None;

        while let Some(row) = rows.try_next().await? {
            let [
                SparseCell::Key(block),
                SparseCell::Key(offset),
                SparseCell::Value(value),
            ] = row.as_slice()
            else {
                return Err(Error::InvalidLayout("invalid sparse payload cells".into()));
            };

            if *block != id
                || *offset >= values.len() as u64
                || previous.is_some_and(|p| p >= *offset)
                || *value == T::ZERO
            {
                return Err(Error::InvalidLayout(
                    "invalid sparse payload offset/value".into(),
                ));
            }
            values[*offset as usize] = *value;
            previous = Some(*offset);
        }

        if previous.is_none() {
            return Err(Error::InvalidLayout("empty sparse row payload".into()));
        }

        Ok(values)
    }

    pub fn replace(&self, id: u64, values: Vec<T>) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let bounds = self.replacement_bounds(id, &values)?;
            let _guard = self.gate.write().await;
            self.healthy()?;

            let analysis = self.analyze(id, &values, &bounds)?;
            let previous = self.descriptor(id).await?;
            self.replace_block(id, previous, values, analysis).await
        })
    }

    pub fn update<'a>(&'a self, id: u64, updates: &'a [(usize, T)]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let bounds = self.geometry.block_bounds(id)?;
            let _guard = self.gate.write().await;
            self.healthy()?;

            let (previous, mut values) = self.read(id).await?;
            super::apply_block_updates(&mut values, self.geometry.block_len(), updates)?;
            let analysis = self.analyze(id, &values, &bounds)?;
            self.replace_block(id, previous, values, analysis).await
        })
    }

    fn replacement_bounds(&self, id: u64, values: &[T]) -> Result<Vec<(u64, u64)>> {
        let bounds = self.geometry.block_bounds(id)?;
        if values.len() != self.geometry.block_len() {
            return Err(Error::InvalidLayout(
                "invalid replacement block length".into(),
            ));
        }

        Ok(bounds)
    }

    // Geometry is needed only for edge padding. The numerical pass visits
    // nonzero offsets directly, without constructing coordinates or occupancy.
    fn validate_payload(
        &self,
        id: u64,
        values: &[T],
        bounds: &[(u64, u64)],
        mut nonzero: impl FnMut(usize),
    ) -> Result<usize> {
        let g = &self.geometry;
        if bounds
            .iter()
            .zip(g.block_shape())
            .any(|((lo, hi), size)| hi - lo != *size)
        {
            #[cfg(test)]
            crate::read_metrics::record(|m| m.padding_walks += 1);
            // Increasing valid offsets delimit padding gaps; no validity mask.
            let mut next = 0;

            for offset in g.block_offsets(id)? {
                if next < offset && values[next..offset].iter().any(|v| *v != T::ZERO) {
                    return Err(Error::InvalidLayout("nonzero sparse padding".into()));
                }
                next = offset + 1;
            }

            if values[next..].iter().any(|v| *v != T::ZERO) {
                return Err(Error::InvalidLayout("nonzero sparse padding".into()));
            }
        }

        let mut nnz = 0;

        for (offset, value) in values.iter().enumerate() {
            if *value != T::ZERO {
                nnz += 1;
                nonzero(offset);
            }
        }

        Ok(nnz)
    }

    fn analyze(&self, id: u64, values: &[T], bounds: &[(u64, u64)]) -> Result<BlockAnalysis> {
        let g = &self.geometry;
        let axis = g.sparse_axis().expect("sparse storage");
        let mut occupied = vec![false; (bounds[axis].1 - bounds[axis].0) as usize];
        #[cfg(test)]
        crate::read_metrics::record(|m| m.occupancy_analyses += 1);
        let axis_stride = g.storage.block_schema.strides[axis] as usize;
        let axis_size = g.block_shape()[axis] as usize;
        let nnz = self.validate_payload(id, values, bounds, |offset| {
            if occupied.len() > 1 {
                occupied[(offset / axis_stride) % axis_size] = true;
            }
        })?;
        if occupied.len() == 1 {
            occupied[0] = nnz > 0;
        }

        Ok(BlockAnalysis {
            start: bounds[axis].0,
            occupied,
            nnz,
        })
    }

    // The entry point holds the ownership guard and has validated the new block.
    fn replace_block(
        &self,
        id: u64,
        previous: Option<u64>,
        values: Vec<T>,
        analysis: BlockAnalysis,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let encoding = Self::encoding(values.len(), analysis.nnz);
            let mut mutation = Mutation(&self.invalid, false);
            #[cfg(test)]
            super::copy_metrics::record(|m| m.replaced_blocks += 1);
            if previous == Some(ROWS) {
                self.clear_rows(id).await?;
            } else if previous == Some(DENSE) && (analysis.nnz == 0 || encoding != DENSE) {
                let file = super::required_file(&*self.blocks.read().await, id)?;
                {
                    let values = file.read::<Vec<T>>().await?;
                    super::validate_stored_length(&values, self.geometry.block_len())?;
                }

                self.blocks.write().await.delete(&id.to_string()).await;
            }

            if analysis.nnz != 0 {
                if encoding == DENSE {
                    if previous == Some(DENSE) {
                        super::replace_payload(&self.blocks, id, values, self.geometry.block_len())
                            .await?;
                    } else {
                        self.create_payload(id, values).await?;
                    }
                } else {
                    let mut table = self.values.write().await;

                    for (offset, value) in values
                        .into_iter()
                        .enumerate()
                        .filter(|(_, v)| *v != T::ZERO)
                    {
                        table
                            .upsert(
                                vec![SparseCell::Key(id), SparseCell::Key(offset as u64)],
                                vec![SparseCell::Value(value)],
                            )
                            .await?;
                    }
                }
            }

            {
                let mut index = self.index.write().await;

                for (offset, present) in analysis.occupied.iter().enumerate() {
                    if !present {
                        index
                            .delete_row(&[analysis.start + offset as u64, id])
                            .await?;
                    }
                }
            }

            self.insert_occupied(id, &analysis).await?;
            if analysis.nnz != 0 {
                self.write_descriptor(id, encoding).await?;
            } else {
                self.descriptors.write().await.delete_row(&[id]).await?;
            }

            mutation.1 = true;
            Ok(())
        })
    }

    async fn create_payload(&self, id: u64, values: Vec<T>) -> Result<()> {
        let bytes = super::block_allocation::<F, T>(&values);
        self.blocks
            .write()
            .await
            .create_file(id.to_string(), values, bytes)
            .await?;
        Ok(())
    }

    async fn insert_occupied(&self, id: u64, analysis: &BlockAnalysis) -> Result<()> {
        let mut index = self.index.write().await;

        for (offset, present) in analysis.occupied.iter().enumerate() {
            if *present {
                index
                    .upsert(vec![analysis.start + offset as u64, id], vec![0])
                    .await?;
            }
        }

        Ok(())
    }

    async fn write_descriptor(&self, id: u64, encoding: u64) -> Result<()> {
        self.descriptors
            .write()
            .await
            .upsert(vec![id], vec![encoding])
            .await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| m.descriptor_writes += 1);
        Ok(())
    }

    fn encoding(len: usize, nnz: usize) -> u64 {
        let bytes = len * std::mem::size_of::<T>();
        if bytes >= MIN_DENSE_BYTES
            && nnz * (ROW_OVERHEAD_BYTES + std::mem::size_of::<T>()) >= bytes
        {
            DENSE
        } else {
            ROWS
        }
    }

    // Construction completes blocks in ascending order. Batch only their fixed-size
    // descriptors, releasing this table guard before any payload or occupancy I/O.
    async fn append_descriptors(&self, descriptors: &mut Vec<(u64, u64)>) -> Result<()> {
        if descriptors.is_empty() {
            return Ok(());
        }

        #[cfg(test)]
        super::copy_metrics::record(|m| {
            m.max_descriptor_batch = m.max_descriptor_batch.max(descriptors.len())
        });
        let rows = descriptors
            .drain(..)
            .map(|(id, encoding)| Ok((vec![id], vec![encoding])));
        let _written = self
            .descriptors
            .write()
            .await
            .upsert_sorted(futures::stream::iter(rows))
            .await?;
        #[cfg(test)]
        super::copy_metrics::record(|m| m.descriptor_writes += _written as usize);
        Ok(())
    }

    // Construction alone knows that completed blocks arrive in increasing order.
    // Use native ordered ingestion without staging rows for dense encodings.
    fn construct(&self, id: u64, values: Vec<T>) -> BoxFuture<'_, Result<Option<u64>>> {
        Box::pin(async move {
            let bounds = self.replacement_bounds(id, &values)?;
            let analysis = self.analyze(id, &values, &bounds)?;
            if analysis.nnz == 0 {
                return Ok(None);
            }

            let encoding = Self::encoding(values.len(), analysis.nnz);
            if encoding == ROWS {
                let entries = values
                    .into_iter()
                    .enumerate()
                    .filter(|(_, v)| *v != T::ZERO)
                    .map(|(offset, value)| {
                        Ok((
                            vec![SparseCell::Key(id), SparseCell::Key(offset as u64)],
                            vec![SparseCell::Value(value)],
                        ))
                    });
                self.values
                    .write()
                    .await
                    .upsert_sorted(futures::stream::iter(entries))
                    .await?;
            } else {
                self.create_payload(id, values).await?;
            }

            self.insert_occupied(id, &analysis).await?;
            Ok(Some(encoding))
        })
    }

    // Staged rows already belong to the unpublished block; only dense conversion removes them.
    async fn complete_staged(&self, id: u64) -> Result<u64> {
        let values = self.read_rows(id).await?;
        let bounds = self.geometry.block_bounds(id)?;
        let analysis = self.analyze(id, &values, &bounds)?;
        let encoding = Self::encoding(values.len(), analysis.nnz);
        if encoding == DENSE {
            self.create_payload(id, values).await?;
            self.clear_rows(id).await?;
        }

        self.insert_occupied(id, &analysis).await?;
        Ok(encoding)
    }

    async fn clear_rows(&self, id: u64) -> Result<()> {
        let offsets = {
            let table = self.values.read().await;
            let mut rows = table.rows(Self::prefix(id), &[], false, None).await?;
            let mut offsets = Vec::new();

            while let Some(row) = rows.try_next().await? {
                if offsets.len() >= self.geometry.block_len() {
                    return Err(Error::InvalidLayout("oversized sparse payload".into()));
                }

                let Some(SparseCell::Key(offset)) = row.get(1) else {
                    return Err(Error::InvalidLayout("invalid sparse offset".into()));
                };
                offsets.push(*offset);
            }
            offsets
        };

        let mut table = self.values.write().await;

        for offset in offsets {
            table
                .delete_row(&[SparseCell::Key(id), SparseCell::Key(offset)])
                .await?;
        }

        Ok(())
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

    // The caller holds the ownership guard; each native sync releases its guards.
    async fn sync_contents(&self) -> Result<()>
    where
        F: freqfs::FileSave + Clone,
    {
        self.healthy()?;
        self.values.sync().await?;
        self.descriptors.sync().await?;
        self.blocks.sync().await?;
        self.index.sync().await?;
        Ok(())
    }
}

impl<F: TensorFileEntry<T>, T: TensorElement> SparseStorage<F, T> {
    pub fn validate<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.healthy()?;
            self.descriptors.validate().await?;
            self.values.validate().await?;
            self.index.validate().await?;
            let g = &self.geometry;
            let axis = g.sparse_axis().expect("sparse");
            let mut after = None;

            loop {
                let page = {
                    let table = self.descriptors.read().await;
                    let range = after.map_or_else(Default::default, |id| {
                        [(
                            "block".into(),
                            b_table::ColumnRange::In((
                                std::ops::Bound::Excluded(id),
                                std::ops::Bound::Unbounded,
                            )),
                        )]
                        .into_iter()
                        .collect()
                    });
                    let mut rows = table.rows(range, &[], false, None).await?;
                    let mut page = Vec::new();

                    while page.len() < SPARSE_INDEX_PAGE_ENTRIES {
                        let Some(row) = rows.try_next().await? else {
                            break;
                        };
                        page.push(row);
                    }
                    page
                };

                if page.is_empty() {
                    break;
                }

                for row in page {
                    if row.len() != 2 || after.is_some_and(|p| row[0] <= p) {
                        return Err(Error::InvalidLayout(
                            "invalid sparse descriptor order".into(),
                        ));
                    }

                    if !matches!(row[1], ROWS | DENSE) {
                        return Err(Error::InvalidLayout("invalid sparse descriptor".into()));
                    }

                    let bounds = g.block_bounds(row[0])?;
                    // Keep native payload I/O out of the enclosing validation future.
                    let block = Box::pin(self.read_payload(row[0], row[1])).await?;
                    let analysis = self.analyze(row[0], &block, &bounds)?;
                    if analysis.nnz == 0 {
                        return Err(Error::InvalidLayout("empty sparse payload".into()));
                    }

                    let occupied = analysis.occupied;

                    for (offset, present) in occupied.into_iter().enumerate() {
                        let value = self
                            .occupied_marker(&[analysis.start + offset as u64, row[0]])
                            .await?;
                        if value != present.then_some(0) {
                            return Err(Error::InvalidLayout(
                                "inconsistent sparse occupancy".into(),
                            ));
                        }
                    }
                    after = Some(row[0]);
                }
            }

            let mut after = None;

            loop {
                let rows = self.validation_page(after).await?;
                if rows.is_empty() {
                    break;
                }

                for row in rows {
                    let bounds = g.block_bounds(row[1])?;
                    if row[2] != 0
                        || !(bounds[axis].0..bounds[axis].1).contains(&row[0])
                        || self.descriptor(row[1]).await?.is_none()
                    {
                        return Err(Error::InvalidLayout(
                            "invalid sparse occupied region".into(),
                        ));
                    }
                    after = Some(row);
                }
            }
            // Page directory metadata, releasing its guard before accessing a table.
            let mut after_name: Option<String> = None;

            loop {
                let names = {
                    let blocks = self.blocks.read().await;
                    blocks
                        .iter()
                        .filter(|(name, _)| after_name.as_ref().is_none_or(|after| *name > after))
                        .take(SPARSE_INDEX_PAGE_ENTRIES)
                        .map(|(name, entry)| (name.clone(), entry.is_file()))
                        .collect::<Vec<_>>()
                };

                if names.is_empty() {
                    break;
                }

                for (name, is_file) in names {
                    if !is_file {
                        return Err(Error::InvalidLayout(
                            "directory inside sparse payloads".into(),
                        ));
                    }

                    if name != METADATA {
                        let id: u64 = name.parse().map_err(|_| {
                            Error::InvalidLayout("invalid sparse payload block".into())
                        })?;
                        if name != id.to_string() || self.descriptor(id).await? != Some(DENSE) {
                            return Err(Error::InvalidLayout(
                                "unreferenced sparse dense payload".into(),
                            ));
                        }
                    }
                    after_name = Some(name);
                }
            }
            // Every stored row must belong to a block with row encoding.
            let mut after: Option<[u64; 2]> = None;
            let mut checked_block = None;

            loop {
                let page = {
                    let table = self.values.read().await;
                    let mut ranges = Vec::new();
                    if let Some(key) = after {
                        for split in (0..2).rev() {
                            let mut range = std::collections::HashMap::new();

                            for i in 0..split {
                                range.insert(
                                    ["block", "offset"][i].into(),
                                    b_table::ColumnRange::Eq(SparseCell::Key(key[i])),
                                );
                            }
                            range.insert(
                                ["block", "offset"][split].into(),
                                b_table::ColumnRange::In((
                                    std::ops::Bound::Excluded(SparseCell::Key(key[split])),
                                    std::ops::Bound::Unbounded,
                                )),
                            );
                            ranges.push(range.into());
                        }
                    } else {
                        ranges.push(Default::default());
                    }

                    let mut page = Vec::new();

                    for range in ranges {
                        let mut rows = table.rows(range, &[], false, None).await?;

                        while page.len() < SPARSE_INDEX_PAGE_ENTRIES {
                            let Some(row) = rows.try_next().await? else {
                                break;
                            };
                            page.push(row);
                        }

                        if page.len() == SPARSE_INDEX_PAGE_ENTRIES {
                            break;
                        }
                    }
                    page
                };

                if page.is_empty() {
                    break;
                }

                for row in page {
                    let [
                        SparseCell::Key(id),
                        SparseCell::Key(offset),
                        SparseCell::Value(_),
                    ] = row.as_slice()
                    else {
                        return Err(Error::InvalidLayout("invalid typed sparse row".into()));
                    };
                    // Rows are ordered by block, including across pages.
                    if checked_block != Some(*id) {
                        if self.descriptor(*id).await? != Some(ROWS) {
                            return Err(Error::InvalidLayout(
                                "unreferenced sparse row payload".into(),
                            ));
                        }
                        checked_block = Some(*id);
                    }
                    after = Some([*id, *offset]);
                }
            }

            Ok(())
        })
    }
}

impl<F: TensorFileEntry<T>, T: TensorElement> SparseStorage<F, T> {
    /// Read at most one execution batch of index keys, releasing all index guards
    /// before returning. Two prefix ranges resume strictly after the previous key.
    pub(super) async fn slice_index_page(
        &self,
        last: Option<[u64; 2]>,
        lo: u64,
        hi: u64,
    ) -> Result<Vec<[u64; 2]>> {
        let _guard = self.gate.read().await;
        self.healthy()?;
        use std::ops::Bound::{Excluded, Included, Unbounded};

        let mut ranges = Vec::with_capacity(2);
        if let Some([coord, block]) = last {
            ranges.push(
                [
                    ("coord".to_string(), b_table::ColumnRange::Eq(coord)),
                    (
                        "block_offset".to_string(),
                        b_table::ColumnRange::In((Excluded(block), Unbounded)),
                    ),
                ]
                .into_iter()
                .collect(),
            );
        }

        if last.is_none_or(|[coord, _]| coord < hi) {
            let lower = last.map_or(Included(lo), |[coord, _]| Excluded(coord));
            ranges.push(
                [(
                    "coord".to_string(),
                    b_table::ColumnRange::In((lower, Included(hi))),
                )]
                .into_iter()
                .collect(),
            );
        }

        let columns = ["coord".to_string(), "block_offset".to_string()];
        let sparse_axis = self.geometry.sparse_axis().expect("sparse storage");
        // At most SPARSE_INDEX_PAGE_ENTRIES fixed-size keys, never an entire index.
        let mut keys = Vec::new();
        let table = self.index.read().await;

        for range in ranges {
            // Ordered row streaming descends to leaves between internal separators.
            let mut rows = table.rows(range, &columns, false, None).await?;

            while let Some(row) = rows.try_next().await? {
                if row.len() != 3
                    || row[2] != 0
                    || row[0] >= self.geometry.schema().shape()[sparse_axis]
                    || row[1] >= self.geometry.block_count()
                    || keys
                        .last()
                        .is_some_and(|previous| *previous >= [row[0], row[1]])
                {
                    return Err(Error::InvalidLayout(
                        "invalid sparse slice index key".into(),
                    ));
                }

                let bounds = self.geometry.block_bounds(row[1])?;
                if !(bounds[sparse_axis].0..bounds[sparse_axis].1).contains(&row[0]) {
                    return Err(Error::InvalidLayout(
                        "occupied region is outside its logical block".into(),
                    ));
                }
                keys.push([row[0], row[1]]);

                #[cfg(test)]
                crate::read_metrics::record(|m| m.index_entries += 1);
                if keys.len() == SPARSE_INDEX_PAGE_ENTRIES {
                    return Ok(keys);
                }
            }
        }

        Ok(keys)
    }

    // Page complete rows, including malformed coordinates and duplicate logical
    // keys. Validation must not restrict traversal to the expected coordinate range.
    async fn validation_page(&self, after: Option<[u64; 3]>) -> Result<Vec<[u64; 3]>> {
        use std::ops::Bound::{Excluded, Unbounded};
        let columns = ["coord", "block_offset", "block_id"];
        let mut ranges = Vec::with_capacity(3);
        if let Some(after) = after {
            for split in (0..columns.len()).rev() {
                let mut range = std::collections::HashMap::new();

                for axis in 0..split {
                    range.insert(columns[axis].into(), b_table::ColumnRange::Eq(after[axis]));
                }
                range.insert(
                    columns[split].into(),
                    b_table::ColumnRange::In((Excluded(after[split]), Unbounded)),
                );
                ranges.push(range);
            }
        } else {
            ranges.push(Default::default());
        }

        let index = &self.index.read().await;
        let mut page = Vec::new();

        for range in ranges {
            let mut rows = index.rows(range.into(), &[], false, None).await?;

            while let Some(row) = rows.try_next().await? {
                if row.len() != 3 {
                    return Err(Error::InvalidLayout("invalid sparse row arity".into()));
                }
                page.push([row[0], row[1], row[2]]);
                if page.len() == SPARSE_INDEX_PAGE_ENTRIES {
                    return Ok(page);
                }
            }
        }

        Ok(page)
    }

    pub(crate) fn occupied_marker<'a>(
        &'a self,
        key: &'a [u64],
    ) -> BoxFuture<'a, Result<Option<u64>>> {
        Box::pin(async move {
            let index = &self.index;

            let index_lock = index.read().await;
            let Some(row) = index_lock.get_row(key).await? else {
                return Ok(None);
            };

            if row.len() != 3 {
                return Err(Error::InvalidLayout(
                    "invalid occupied-region row width".into(),
                ));
            }

            Ok(Some(row[2]))
        })
    }
}

/// Owns an unpublished destination. Only finish exposes the completed Tensor.
pub(super) struct Construction<F, T> {
    tensor: Tensor<F, T>,
    // Present only when geometry proves blocks are contiguous in row-major order.
    current: Option<(Option<u64>, Vec<T>)>,
}

impl<F: TensorFileEntry<T>, T: TensorElement> Construction<F, T> {
    pub fn new(tensor: Tensor<F, T>) -> Self {
        let owner = tensor.storage.sparse().expect("sparse construction");
        let shape = owner.geometry.schema().shape();
        let block = owner.geometry.block_shape();
        let contiguous = block
            .iter()
            .position(|&n| n > 1)
            .is_none_or(|first| block[first + 1..] == shape[first + 1..]);
        let current = contiguous.then(|| (None, vec![T::ZERO; owner.geometry.block_len()]));
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
            owner.healthy()?;
            let mut mutation = Mutation(&owner.invalid, false);
            if let Some((current, block)) = &mut self.current {
                let mut descriptors = Vec::new();

                for (coord, value) in coords.iter().zip(values) {
                    let (id, offset) = owner.geometry.block_position(coord)?;
                    if value == T::ZERO {
                        continue;
                    }

                    if let Some(previous) = *current {
                        if id < previous {
                            return Err(Error::InvalidCoord(
                                "construction blocks out of order".into(),
                            ));
                        }

                        if id != previous {
                            if let Some(encoding) =
                                owner.construct(previous, std::mem::take(block)).await?
                            {
                                descriptors.push((previous, encoding));
                                if descriptors.len() == crate::expression::MAX_BATCH_ELEMENTS {
                                    owner.append_descriptors(&mut descriptors).await?;
                                }
                            }
                            block.resize(owner.geometry.block_len(), T::ZERO);
                            #[cfg(test)]
                            super::copy_metrics::record(|m| m.constructed_blocks += 1);
                        }
                    }
                    *current = Some(id);
                    block[offset] = value;
                }
                owner.append_descriptors(&mut descriptors).await?;

                mutation.1 = true;
                return Ok(());
            }

            let mut table = owner.values.write().await;

            for (coord, value) in coords.iter().zip(values) {
                let (id, offset) = owner.geometry.block_position(coord)?;
                if value != T::ZERO {
                    table
                        .upsert(
                            vec![SparseCell::Key(id), SparseCell::Key(offset as u64)],
                            vec![SparseCell::Value(value)],
                        )
                        .await?;
                }
            }

            mutation.1 = true;
            Ok(())
        })
    }

    pub fn finish(mut self) -> BoxFuture<'static, Result<Tensor<F, T>>> {
        Box::pin(async move {
            {
                let owner = self.tensor.storage.sparse().expect("sparse construction");
                let _guard = owner.gate.write().await;
                owner.healthy()?;
                let mut mutation = Mutation(&owner.invalid, false);
                let mut descriptors = Vec::new();
                match self.current.take() {
                    Some((current, values)) => {
                        if let Some(id) = current {
                            if let Some(encoding) = owner.construct(id, values).await? {
                                descriptors.push((id, encoding));
                            }

                            #[cfg(test)]
                            super::copy_metrics::record(|m| m.constructed_blocks += 1);
                        }
                    }
                    None => {
                        let mut after = None;

                        loop {
                            // Seek the next block without retaining a table guard across publication.
                            let id = {
                                let table = owner.values.read().await;
                                let range = after.map_or_else(Default::default, |id| {
                                    [(
                                        "block".into(),
                                        b_table::ColumnRange::In((
                                            std::ops::Bound::Excluded(SparseCell::Key(id)),
                                            std::ops::Bound::Unbounded,
                                        )),
                                    )]
                                    .into_iter()
                                    .collect()
                                });
                                let mut rows = table.rows(range, &[], false, None).await?;
                                match rows.try_next().await? {
                                    None => break,
                                    Some(row) => match row.first() {
                                        Some(SparseCell::Key(id)) => *id,
                                        _ => {
                                            return Err(Error::InvalidLayout(
                                                "invalid staged block".into(),
                                            ));
                                        }
                                    },
                                }
                            };

                            let encoding = owner.complete_staged(id).await?;
                            descriptors.push((id, encoding));
                            if descriptors.len() == crate::expression::MAX_BATCH_ELEMENTS {
                                owner.append_descriptors(&mut descriptors).await?;
                            }

                            #[cfg(test)]
                            super::copy_metrics::record(|m| m.constructed_blocks += 1);
                            after = Some(id);
                        }
                    }
                }
                owner.append_descriptors(&mut descriptors).await?;
                self.tensor.persist_metadata().await?;

                mutation.1 = true;
            }

            Ok(self.tensor)
        })
    }
}

#[cfg(test)]
impl<F: TensorFileEntry<T>, T: TensorElement> Tensor<F, T> {
    pub(crate) async fn corrupt_occupied_region(&self, old: [u64; 2], new: [u64; 2]) {
        let mut index = self.storage.sparse().unwrap().index.write().await;
        index.delete_row(&old).await.unwrap();
        index.upsert(new.to_vec(), vec![0]).await.unwrap();
    }

    /// Internal corruption fixture: remove a payload while retaining its descriptor.
    pub(crate) async fn corrupt_sparse_payload(&self, id: u64) {
        let a = self.storage.sparse().unwrap();
        let encoding = a.descriptor(id).await.unwrap().unwrap();
        if encoding == DENSE {
            self.storage
                .blocks()
                .write()
                .await
                .delete(&id.to_string())
                .await;
        } else {
            a.values
                .write()
                .await
                .delete_range(SparseStorage::<F, T>::prefix(id))
                .await
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, TensorSchema, TensorWrite};

    #[tokio::test]
    async fn successive_row_replacements_preserve_values() {
        let (root, dir) = crate::test_support::new_dir("adaptive_replacements").await;
        let t = Tensor::<crate::test_support::FsEntry, f32>::create(
            dir,
            TensorSchema::new(
                number_general::NumberType::Float(number_general::FloatType::F32),
                vec![2, 4].into(),
            )
            .unwrap(),
            Layout::Sparse { axis: None },
            4,
        )
        .await
        .unwrap();

        for i in 0..4 {
            t.write_value(&[0, i], (i + 1) as f32).await.unwrap();
            t.validate().await.unwrap();
        }
        t.validate().await.unwrap();
        crate::test_support::cleanup(&root).await;
    }
}

#[cfg(test)]
mod representation_tests {
    use super::*;
    use crate::test_support::{FsEntry, cleanup, new_dir, open_dir};
    use crate::{Layout, StorageGeometry, TensorSchema, TensorSource, TensorWrite};

    #[tokio::test]
    async fn occupied_pages_validate_markers_and_block_membership() {
        for (key, marker) in [([0, 0], 1), ([1, 0], 0)] {
            let (root, dir) = new_dir("occupied_page_validation").await;
            let geometry = StorageGeometry::new(
                TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![2, 4].into())
                    .unwrap(),
                Layout::Sparse { axis: Some(0) },
                vec![1, 4].into(),
            )
            .unwrap();
            let tensor = Tensor::<FsEntry, f64>::create_with_geometry(dir, geometry)
                .await
                .unwrap();
            tensor.replace_logical_block(0, vec![1.; 4]).await.unwrap();
            let owner = tensor.storage.sparse().unwrap();
            owner.index.write().await.delete_row(&[0, 0]).await.unwrap();
            owner
                .index
                .write()
                .await
                .upsert(key.to_vec(), vec![marker])
                .await
                .unwrap();
            // Strict validation resumes after the complete stored row, including
            // a malformed marker at the same logical key as the previous row.
            assert_eq!(
                owner.validation_page(Some([0, 0, 0])).await.unwrap(),
                vec![[key[0], key[1], marker]]
            );
            assert!(
                tensor
                    .occupied_regions(None, 0, 1)
                    .try_next()
                    .await
                    .is_err()
            );
            cleanup(&root).await;
        }
    }

    async fn transitions<T: TensorElement>()
    where
        FsEntry: TensorFileEntry<T>,
    {
        let threshold = 1024 / std::mem::size_of::<T>();

        for len in [threshold - 1, threshold, 4096] {
            let (root, dir) = new_dir("adaptive_transitions").await;
            let schema = TensorSchema::new(T::dtype(), vec![2, len as u64].into()).unwrap();
            let geometry = StorageGeometry::new(
                schema,
                Layout::Sparse { axis: Some(0) },
                vec![1, len as u64].into(),
            )
            .unwrap();
            let t = Tensor::<FsEntry, T>::create_with_geometry(dir, geometry)
                .await
                .unwrap();
            let bytes = len * std::mem::size_of::<T>();
            let tie = bytes / (24 + std::mem::size_of::<T>());
            t.replace_logical_block(1, vec![T::ONE; len]).await.unwrap();
            t.replace_logical_block(0, {
                let mut v = vec![T::ZERO; len];
                v[0] = T::ONE;
                v
            })
            .await
            .unwrap();
            // The threshold fixture contains both row and file payloads for every dtype.
            // Keep maximum-block copy coverage at the smallest and largest element widths.
            if len == threshold || (len == 4096 && matches!(std::mem::size_of::<T>(), 1 | 16)) {
                let (copy_root, copy_dir) = new_dir("representation_copy").await;
                let copied = Tensor::<FsEntry, T>::copy_from(copy_dir, &t, len)
                    .await
                    .unwrap();
                copied.validate().await.unwrap();
                assert_eq!(copied.read_logical_block(0).await.unwrap()[0], T::ONE);
                assert_eq!(
                    copied.read_logical_block(1).await.unwrap(),
                    vec![T::ONE; len]
                );
                cleanup(&copy_root).await;
            }

            for count in [1, tie.max(1), (tie + 1).min(len), len, 1, 0, len, 0] {
                let mut values = vec![T::ZERO; len];
                values[..count].fill(T::ONE);
                t.replace_logical_block(0, values.clone()).await.unwrap();
                crate::read_metrics::CURRENT
                    .scope(Default::default(), async {
                        t.write_value(&[0, 0], values[0]).await.unwrap();
                        crate::read_metrics::CURRENT.with(|m| {
                            assert_eq!(m.borrow().descriptor_lookups, 1);
                        });
                    })
                    .await;
                crate::read_metrics::CURRENT
                    .scope(Default::default(), async {
                        assert_eq!(t.read_logical_block(0).await.unwrap(), values);
                        assert_eq!(t.read_logical_block(1).await.unwrap(), vec![T::ONE; len]);
                        crate::read_metrics::CURRENT.with(|m| {
                            assert_eq!(m.borrow().occupancy_analyses, 0);
                            assert_eq!(m.borrow().padding_walks, 0);
                        });
                    })
                    .await;
                let descriptor = t.storage.sparse().unwrap().descriptor(0).await.unwrap();
                let expected = if count == 0 {
                    None
                } else {
                    Some(
                        if bytes < 1024 || count * (24 + std::mem::size_of::<T>()) < bytes {
                            1
                        } else {
                            2
                        },
                    )
                };
                assert_eq!(descriptor, expected);
                crate::read_metrics::CURRENT
                    .scope(Default::default(), async {
                        t.validate().await.unwrap();
                        crate::read_metrics::CURRENT.with(|m| {
                            let blocks = 1 + usize::from(count != 0);
                            assert_eq!(m.borrow().occupancy_analyses, blocks);
                            // One occupancy and one payload-reference probe per block.
                            assert_eq!(m.borrow().descriptor_lookups, 2 * blocks);
                        });
                    })
                    .await;
            }
            t.sync_all().await.unwrap();
            drop(t);
            let loaded = Tensor::<FsEntry, T>::load(open_dir(&root).unwrap())
                .await
                .unwrap();
            assert_eq!(
                loaded.read_logical_block(1).await.unwrap(),
                vec![T::ONE; len]
            );
            cleanup(&root).await;
        }
    }

    macro_rules! dtypes {
        ($($name:ident: $t:ty),+ $(,)?) => {
            $(
                #[tokio::test]
                async fn $name() {
                    transitions::<$t>().await;
                }
            )+
        };
    }

    dtypes!(
        u8_representations: u8,
        u16_representations: u16,
        u32_representations: u32,
        u64_representations: u64,
        i8_representations: i8,
        i16_representations: i16,
        i32_representations: i32,
        i64_representations: i64,
        f32_representations: f32,
        f64_representations: f64
    );

    #[cfg(feature = "complex")]
    dtypes!(
        c32_representations: crate::complex::Complex32,
        c64_representations: crate::complex::Complex64
    );

    #[tokio::test]
    async fn construction_publishes_interleaved_blocks_once() {
        for shape in [
            [257, 17],
            [2, crate::expression::MAX_BATCH_ELEMENTS as u64 + 1],
        ] {
            let (root, dir) = new_dir("construction_interleaved").await;
            let tensor = Tensor::<FsEntry, f64>::unpublished(
                dir,
                TensorSchema::new(
                    <f64 as number_general::DType>::dtype(),
                    shape.to_vec().into(),
                )
                .unwrap(),
                Layout::Sparse { axis: Some(1) },
                4096,
            )
            .await
            .unwrap();
            let tensor = crate::read_metrics::CURRENT
                .scope(
                    Default::default(),
                    super::super::copy_metrics::CURRENT.scope(Default::default(), async {
                        let mut builder = Construction::new(tensor);
                        let mut coords = Vec::new();

                        for i in 0..shape[0] {
                            for j in 0..shape[1] {
                                coords.push(vec![i, j]);
                                if coords.len() == crate::expression::MAX_BATCH_ELEMENTS {
                                    builder
                                        .stage(&coords, vec![1.; coords.len()])
                                        .await
                                        .unwrap();
                                    coords.clear();
                                }
                            }
                        }
                        builder
                            .stage(&coords, vec![1.; coords.len()])
                            .await
                            .unwrap();
                        let tensor = builder.finish().await.unwrap();
                        super::super::copy_metrics::CURRENT.with(|m| {
                            let m = m.borrow();
                            assert_eq!(m.constructed_blocks as u64, shape[1]);
                            assert_eq!(m.block_updates, 0);
                            assert_eq!(
                                m.max_descriptor_batch,
                                (shape[1] as usize).min(crate::expression::MAX_BATCH_ELEMENTS)
                            );
                            assert_eq!(m.replaced_blocks, 0);
                            assert_eq!(m.descriptor_writes as u64, shape[1]);
                            assert_eq!(m.staged_elements as u64, shape[0] * shape[1]);
                            assert_eq!(m.max_staging_batch, crate::expression::MAX_BATCH_ELEMENTS);
                        });
                        crate::read_metrics::CURRENT
                            .with(|m| assert_eq!(m.borrow().descriptor_lookups, 0));
                        tensor
                    }),
                )
                .await;
            for id in 0..shape[1] {
                assert_eq!(
                    tensor.read_logical_block(id).await.unwrap(),
                    vec![1.; shape[0] as usize]
                );
            }
            tensor.validate().await.unwrap();
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn interrupted_construction_never_reopens_and_releases_guards() {
        for phase in 0..3 {
            let (root, dir) = new_dir("construction_cancel").await;
            let tensor = Tensor::<FsEntry, f64>::unpublished(
                dir.clone(),
                TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![2, 512].into())
                    .unwrap(),
                Layout::Sparse { axis: Some(1) },
                512,
            )
            .await
            .unwrap();
            let mut builder = Construction::new(tensor);
            builder.stage(&[vec![0, 0]], vec![1.]).await.unwrap();
            let probe = builder.tensor.clone();
            let owner = probe.storage.sparse().unwrap();
            if phase == 1 {
                let held = owner.index.write().await;
                let mut future = Box::pin(builder.finish());
                assert!(
                    tokio::time::timeout(std::time::Duration::from_millis(10), &mut future)
                        .await
                        .is_err()
                );
                drop(future);
                drop(held);
            } else if phase == 2 {
                // Row encoding needs no payload file; completion first touches this
                // directory when publishing metadata after its native tables.
                let held = owner.blocks.write().await;
                let mut future = Box::pin(builder.finish());
                assert!(
                    tokio::time::timeout(std::time::Duration::from_millis(10), &mut future)
                        .await
                        .is_err()
                );
                drop(future);
                drop(held);
                assert_eq!(owner.descriptor(0).await.unwrap(), Some(ROWS));
            } else {
                let held = owner.values.write().await;
                let coords = [vec![1, 0]];
                let mut future = Box::pin(builder.stage(&coords, vec![2.]));
                assert!(
                    tokio::time::timeout(std::time::Duration::from_millis(10), &mut future)
                        .await
                        .is_err()
                );
                drop(future);
                drop(held);
                drop(builder);
            }
            assert!(owner.gate.try_write().is_ok());
            assert!(owner.healthy().is_err());
            assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn construction_rejects_invalid_staging_and_final_payloads() {
        for final_payload in [false, true] {
            let (root, dir) = new_dir("construction_error").await;
            let tensor = Tensor::<FsEntry, f64>::unpublished(
                dir.clone(),
                TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![2, 2].into())
                    .unwrap(),
                Layout::Sparse { axis: Some(1) },
                2,
            )
            .await
            .unwrap();
            let mut builder = Construction::new(tensor);
            if final_payload {
                builder.stage(&[vec![0, 0]], vec![1.]).await.unwrap();
                builder
                    .tensor
                    .storage
                    .sparse()
                    .unwrap()
                    .values
                    .write()
                    .await
                    .upsert(
                        vec![SparseCell::Key(0), SparseCell::Key(99)],
                        vec![SparseCell::Value(1.)],
                    )
                    .await
                    .unwrap();
                assert!(builder.finish().await.is_err());
            } else {
                assert!(
                    builder
                        .stage(&[vec![0, 0], vec![2, 0]], vec![1., 2.])
                        .await
                        .is_err()
                );
                assert!(builder.finish().await.is_err());
            }
            assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn strict_reopen_rejects_malformed_adaptive_storage() {
        for fault in 0..10 {
            for &nonzero in if fault >= 7 {
                &[0, 1, 512][..]
            } else {
                &[0][..]
            } {
                let (root, dir) = new_dir("adaptive_invalid").await;
                let t = Tensor::<FsEntry, f64>::create(
                    dir.clone(),
                    TensorSchema::new(
                        <f64 as number_general::DType>::dtype(),
                        (if fault >= 7 { vec![2, 512] } else { vec![4] }).into(),
                    )
                    .unwrap(),
                    Layout::Sparse { axis: None },
                    512,
                )
                .await
                .unwrap();
                t.replace_logical_block(0, vec![1.; t.block_len()])
                    .await
                    .unwrap();
                let a = t.storage.sparse().unwrap();
                match fault {
                    0 => {
                        a.values
                            .write()
                            .await
                            .upsert(
                                vec![SparseCell::Key(0), SparseCell::Key(99)],
                                vec![SparseCell::Value(2.)],
                            )
                            .await
                            .unwrap();
                    }
                    1 => {
                        a.values
                            .write()
                            .await
                            .upsert(
                                vec![SparseCell::Key(1), SparseCell::Key(0)],
                                vec![SparseCell::Value(2.)],
                            )
                            .await
                            .unwrap();
                    }
                    2 => {
                        a.values
                            .write()
                            .await
                            .upsert(
                                vec![SparseCell::Key(0), SparseCell::Key(0)],
                                vec![SparseCell::Value(0.)],
                            )
                            .await
                            .unwrap();
                    }
                    3 => {
                        a.descriptors.write().await.delete_row(&[0]).await.unwrap();
                    }
                    4 => {
                        t.storage
                            .blocks()
                            .write()
                            .await
                            .create_file(
                                "0.9".into(),
                                vec![1f64],
                                std::mem::size_of::<FsEntry>()
                                    + std::mem::size_of::<Vec<u64>>()
                                    + 8,
                            )
                            .await
                            .unwrap();
                    }
                    5 => {
                        a.index.write().await.delete_row(&[0, 0]).await.unwrap();
                    }
                    6 => {
                        t.storage
                            .blocks()
                            .write()
                            .await
                            .create_file(
                                "invalid_payload_name".into(),
                                vec![1f64],
                                std::mem::size_of::<FsEntry>()
                                    + std::mem::size_of::<Vec<u64>>()
                                    + 8,
                            )
                            .await
                            .unwrap();
                    }
                    7 => {
                        let file = t
                            .storage
                            .blocks()
                            .read()
                            .await
                            .get_file("0")
                            .cloned()
                            .unwrap();
                        file.write::<Vec<f64>>(0).await.unwrap().pop();
                    }
                    8 => {
                        let mut blocks = t.storage.blocks().write().await;
                        blocks.delete("0").await;
                        blocks
                            .create_file(
                                "0".into(),
                                vec![1u8; 512],
                                std::mem::size_of::<FsEntry>()
                                    + std::mem::size_of::<Vec<u64>>()
                                    + 512,
                            )
                            .await
                            .unwrap();
                    }
                    9 => {
                        t.storage.blocks().write().await.delete("0").await;
                    }
                    _ => unreachable!(),
                }
                t.sync().await.unwrap();
                if fault >= 7 {
                    // Replacing a dense sparse payload must not repair missing or malformed storage.
                    let mut replacement = vec![0.; t.block_len()];
                    replacement[..nonzero].fill(2.);
                    assert!(t.replace_logical_block(0, replacement).await.is_err());
                    assert!(a.healthy().is_err());
                }
                drop(t);
                drop(dir);
                assert!(
                    Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
                        .await
                        .is_err(),
                    "fault {fault}"
                );
                cleanup(&root).await;
            }
        }
    }

    #[tokio::test]
    async fn tiny_blocks_share_pages_and_preserve_scalar_bits() {
        let (root, _) = new_dir("adaptive_bits").await;
        // Splitting retains source and replacement pages, so this page-sharing
        // fixture needs several admitted native pages, not just one node target.
        let dir = freqfs::Cache::new(
            4 * crate::schema::SPARSE_NODE_MEMORY,
            None,
            0,
            std::time::Duration::from_secs(3),
        )
        .load(root.to_path_buf())
        .unwrap();
        let t = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(<f64 as number_general::DType>::dtype(), vec![257].into()).unwrap(),
            Layout::Sparse { axis: None },
            4096,
        )
        .await
        .unwrap();
        let values = [
            f64::from_bits(0x7ff8000000000123),
            f64::INFINITY,
            f64::NEG_INFINITY,
            3.,
        ];

        for id in 0..257 {
            t.replace_logical_block(id, vec![values[id as usize % 4]])
                .await
                .unwrap_or_else(|cause| panic!("block {id}: {cause}"));
        }
        assert_eq!(
            t.storage.blocks().read().await.len(),
            1,
            "tiny values have metadata only, no payload files"
        );
        let values_dir = t.directory.read().await.get_dir("values").cloned().unwrap();
        let primary = values_dir.read().await.get_dir("primary").cloned().unwrap();
        assert!(
            primary.read().await.len() < 257,
            "populated logical blocks must share native pages"
        );
        t.validate().await.expect("before deletion");
        t.replace_logical_block(63, vec![0.]).await.unwrap();
        t.validate().await.expect("after deletion");
        t.replace_logical_block(64, vec![7.]).await.unwrap();
        let a = t.storage.sparse().unwrap();
        a.descriptors
            .validate()
            .await
            .expect("descriptors after replacement");
        a.values.validate().await.expect("values after replacement");
        t.validate().await.expect("after replacement");
        t.sync_all().await.unwrap();
        drop(t);
        let t = Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
            .await
            .unwrap();
        for id in 0..257 {
            let expected = match id {
                63 => 0.,
                64 => 7.,
                _ => values[id as usize % 4],
            };
            assert_eq!(
                t.read_logical_block(id).await.unwrap()[0].to_bits(),
                expected.to_bits()
            );
        }
        cleanup(&root).await;
    }
}
