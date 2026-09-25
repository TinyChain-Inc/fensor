use std::collections::BTreeMap;
use std::io::{Error as IoError, ErrorKind};
use std::sync::Arc;

use b_table::{TableLock, collate::Collator};
use destream::{de, en};
use freqfs::{DirLock, FileLoad};
use futures::{StreamExt, TryStreamExt};

#[cfg(test)]
use number_general::FloatType;
use number_general::NumberType;
use safecast::AsType;

use crate::error::{Error, Result};
use crate::mapping::CoordinateMap;
use crate::request::BatchRequest;
use crate::schema::{
    BlockPosition, Layout, MAX_BLOCK_CAPACITY, SparseIndexSchema, SparseTableSchema, StorageSchema,
    TensorSchema,
};
use crate::storage_read::{LogicalGroups, Run};
use crate::traits::{
    BoxFuture, TensorArray, TensorBlockStore, TensorGeometry, TensorRead, TensorSparseIndex,
    TensorWrite, TensorWriteBulk,
};
use crate::view::TensorView;
use crate::{Range, TensorMetadata, validate};

const BLOCKS: &str = "blocks";

const INDEX: &str = "index";

const METADATA: &str = "metadata";

pub trait TensorElement:
    ha_ndarray::Number
    + number_general::DType
    + Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + de::FromStream<Context = ()>
    + for<'en> en::ToStream<'en>
{
}

impl TensorElement for u8 {}

impl TensorElement for f32 {}

impl TensorElement for f64 {}

pub trait TensorFileEntry<T: TensorElement>:
    FileLoad
    + AsType<b_table::Node<u64>>
    + AsType<Vec<T>>
    + AsType<TensorMetadata<T>>
    + Send
    + Sync
    + 'static
{
}

impl<FE, T> TensorFileEntry<T> for FE
where
    FE: FileLoad
        + AsType<b_table::Node<u64>>
        + AsType<Vec<T>>
        + AsType<TensorMetadata<T>>
        + Send
        + Sync
        + 'static,
    T: TensorElement,
{
}

// ---------------------------------------------------------------------------
// Private storage structs
// ---------------------------------------------------------------------------

struct DenseStorage<FE> {
    blocks: DirLock<FE>,
    storage_schema: StorageSchema,
}

type SparseIndex<FE> = TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>;

struct SparseStorage<FE> {
    blocks: DirLock<FE>,
    index: SparseIndex<FE>,
    storage_schema: StorageSchema,
}

enum Storage<FE> {
    Dense(DenseStorage<FE>),
    Sparse(SparseStorage<FE>),
}

impl<FE> Storage<FE> {
    pub(crate) fn blocks(&self) -> &DirLock<FE> {
        match self {
            Self::Dense(s) => &s.blocks,
            Self::Sparse(s) => &s.blocks,
        }
    }

    pub(crate) fn storage_schema(&self) -> &StorageSchema {
        match self {
            Self::Dense(s) => &s.storage_schema,
            Self::Sparse(s) => &s.storage_schema,
        }
    }

    pub(crate) fn index(&self) -> Option<&SparseIndex<FE>> {
        match self {
            Self::Dense(_) => None,
            Self::Sparse(s) => Some(&s.index),
        }
    }
}

enum SparseWriteAction {
    Write(u64),
    DeleteRow(u64),
    NoOp,
    CreateBlockAndWrite(u64),
}

#[derive(Clone)]
pub struct Tensor<FE, T> {
    storage: Arc<Storage<FE>>,
    schema: TensorSchema,
    _dtype: std::marker::PhantomData<T>,
}

impl<FE, T> Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    /// Write pending blocks and the current sparse index root to the filesystem buffer.
    ///
    /// Call this before dropping and reopening a tensor. Syncing the containing
    /// directory alone does not publish the index's in-memory root. The caller
    /// must exclude concurrent writes and remains responsible for durable directory
    /// synchronization and any transaction policy.
    pub async fn sync(&self) -> Result<()>
    where
        FE: freqfs::FileSave + Clone,
    {
        self.storage.blocks().sync().await?;

        if let Some(index) = self.storage.index() {
            index.sync().await?;
        }

        Ok(())
    }

    pub async fn create(
        dir: DirLock<FE>,
        schema: TensorSchema,
        layout: Layout,
        max_capacity: usize,
    ) -> Result<Self> {
        if max_capacity == 0 || max_capacity > MAX_BLOCK_CAPACITY {
            return Err(Error::InvalidSchema(format!(
                "max block capacity must be non-zero and at most {}, got {max_capacity}",
                MAX_BLOCK_CAPACITY
            )));
        }

        validate_tensor_dtype::<T>(schema.dtype())?;

        let storage_schema = StorageSchema::new(schema.shape().as_slice(), layout, max_capacity)?;

        let mut dir_guard = dir.try_write()?;
        let blocks_dir = dir_guard.create_dir(BLOCKS.to_string())?;

        let index: Option<SparseIndex<FE>> = if let Layout::Sparse { .. } = layout {
            let index_dir = dir_guard.create_dir(INDEX.to_string())?;
            Some(TableLock::create(
                SparseTableSchema::default(),
                Collator::default(),
                index_dir,
            )?)
        } else {
            None
        };

        Self::new_storage(blocks_dir, index, schema, storage_schema).await
    }

    /// Create an independent tensor by consuming a reader in bounded batches.
    ///
    /// Evaluation is driven by reads; no intermediate tensor is created. The
    /// destination uses the reader's dtype and shape. Sparse output resets the
    /// axis hint to `None` and omits zeros. Errors propagate, leaving cleanup of
    /// partial destination storage to the caller. Coordinate-bearing blocks may be
    /// unordered; bounds, lengths, and total count are checked. Exactly-once coverage
    /// is the source contract, without a whole-output duplicate detector.
    /// Updates are grouped by destination block within each bounded incoming batch;
    /// no block guard or computed result cache is retained across batches. Failures
    /// may leave partial output without a guaranteed update prefix or rollback.
    /// One destination update overlaps one source lookahead; neither spawns work.
    /// Dropping the copy cancels both futures. Concurrent failures have no promised
    /// precedence. The source retains its own bounded buffering in addition to the
    /// current block and at most one completed lookahead block.
    pub async fn copy_from<R>(dir: DirLock<FE>, source: &R, max_capacity: usize) -> Result<Self>
    where
        R: TensorRead<DType = T> + ?Sized,
    {
        let mut blocks = source.read_coordinate_blocks()?;
        let schema =
            TensorSchema::new(<T as number_general::DType>::dtype(), source.shape().into())?;
        let layout = match source.layout() {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: None },
        };

        #[cfg(test)]
        let initialized = std::time::Instant::now();
        let output = Self::create(dir, schema, layout, max_capacity).await?;

        #[cfg(test)]
        copy_metrics::record(|m| m.initialize += initialized.elapsed());
        let expected = crate::schema::checked_product(source.shape())?;
        let mut received = 0u64;

        #[cfg(test)]
        let consumed = std::time::Instant::now();
        let mut next = blocks.try_next().await?;

        #[cfg(test)]
        copy_metrics::record(|m| m.consume += consumed.elapsed());

        while let Some((coords, values)) = next {
            if coords.len() != values.len() || values.len() > crate::expression::MAX_BATCH_ELEMENTS
            {
                return Err(Error::InvalidLayout(format!(
                    "copy block: expected equal coordinate/value lengths at most {}, got {} and {}",
                    crate::expression::MAX_BATCH_ELEMENTS,
                    coords.len(),
                    values.len()
                )));
            }
            received = received
                .checked_add(values.len() as u64)
                .filter(|n| *n <= expected)
                .ok_or_else(|| {
                    Error::InvalidLayout("reader returned more values than its shape".into())
                })?;
            #[cfg(test)]
            let joined = std::time::Instant::now();
            #[cfg(test)]
            let (mut update_time, mut consume_time) = Default::default();

            // One writer, one lookahead, and no background task. Polling next lets
            // the source's existing ordered window progress while storage waits.
            let update = async {
                #[cfg(test)]
                let started = std::time::Instant::now();
                let result = output.write_copy_batch(&coords, values).await;
                #[cfg(test)]
                {
                    update_time = started.elapsed();
                }
                result
            };
            let consume = async {
                #[cfg(test)]
                let started = std::time::Instant::now();
                let result = blocks.try_next().await;
                #[cfg(test)]
                {
                    consume_time = started.elapsed();
                }
                result
            };
            let ((), following) = futures::try_join!(update, consume)?;
            next = following;

            #[cfg(test)]
            copy_metrics::record(|m| {
                m.update += update_time;
                m.consume += consume_time;
                m.overlap += (update_time + consume_time).saturating_sub(joined.elapsed());
            });
        }

        if received != expected {
            return Err(Error::InvalidLayout(
                "reader returned fewer values than its shape".into(),
            ));
        }

        Ok(output)
    }

    pub async fn load(dir: DirLock<FE>) -> Result<Self> {
        let mut dir_guard = dir.try_write()?;
        let blocks_dir = dir_guard.get_or_create_dir(BLOCKS.to_string())?;
        let (schema, storage_schema) = load_metadata_file::<FE, T>(&blocks_dir).await?;
        validate_tensor_dtype::<T>(schema.dtype())?;
        let index: Option<SparseIndex<FE>> = if let Layout::Sparse { .. } = storage_schema.layout {
            let index_dir = dir_guard.get_dir(INDEX).cloned().ok_or_else(|| {
                Error::InvalidSchema("sparse tensor missing index directory".to_string())
            })?;
            Some(TableLock::load(
                SparseTableSchema::default(),
                Collator::default(),
                index_dir,
            )?)
        } else {
            None
        };

        Self::new_storage(blocks_dir, index, schema, storage_schema).await
    }

    pub fn view(&self) -> TensorView<'_, FE, T> {
        TensorView::new_identity(self)
    }

    pub(crate) fn block_position_from_base_coord(&self, base_coord: &[u64]) -> BlockPosition {
        crate::storage_read::block_position(
            base_coord.iter().copied(),
            self.block_shape(),
            self.block_strides(),
            self.grid_strides(),
        )
    }

    pub(crate) fn block_shape(&self) -> &[u64] {
        &self.storage.storage_schema().block_schema.shape
    }

    pub(crate) fn block_strides(&self) -> &[u64] {
        &self.storage.storage_schema().block_schema.strides
    }

    pub(crate) fn grid_strides(&self) -> &[u64] {
        &self.storage.storage_schema().strides
    }

    pub(crate) fn num_blocks(&self) -> u64 {
        self.storage.storage_schema().shape.iter().product::<u64>()
    }

    pub(crate) fn block_len(&self) -> usize {
        self.block_shape().iter().product::<u64>() as usize
    }

    // One validated storage block, independent of tensor size.
    fn default_block(&self) -> Vec<T> {
        vec![T::default(); self.block_len()]
    }

    async fn initialize_dense_blocks(&self) -> Result<()> {
        if let Layout::Sparse { .. } = self.layout() {
            return Ok(());
        }

        let num_blocks = self.num_blocks();

        for block_id in 0..num_blocks {
            if !self.block_exists(block_id).await {
                self.write_block(block_id, self.default_block()).await?;
            }
        }

        Ok(())
    }

    async fn block_exists(&self, block_id: u64) -> bool {
        let blocks = self.storage.blocks().read().await;
        blocks.get_file(&block_id.to_string()).is_some()
    }

    // Updates retain input order and contain at most one execution batch.
    // Validate every offset before changing any value in this storage block.
    fn apply_block_updates(&self, block: &mut [T], updates: &[(usize, usize, T)]) -> Result<()> {
        if block.len() != self.block_len() {
            return Err(Error::InvalidLayout("invalid stored block length".into()));
        }

        for &(_, offset, _) in updates {
            validate::ensure_offset_in_bounds(offset, block.len())?;
        }

        for &(_, offset, value) in updates {
            block[offset] = value;
        }

        Ok(())
    }

    async fn write_block_updates(
        &self,
        block_id: u64,
        updates: &[(usize, usize, T)],
    ) -> Result<()> {
        let file = {
            let blocks = self.storage.blocks().read().await;
            blocks
                .get_file(&block_id.to_string())
                .cloned()
                .ok_or_else(|| Error::from(IoError::new(ErrorKind::NotFound, "Missing block")))?
        };
        let mut block = file.write::<Vec<T>>().await?;
        self.apply_block_updates(&mut block, updates)?;

        #[cfg(test)]
        copy_metrics::record(|m| m.block_updates += 1);
        Ok(())
    }

    /// Only copying uses this grouped path: sparse zeros are omitted, not deletions.
    /// Maps retain at most 4096 positions/values, never another coordinate list.
    async fn write_copy_batch(&self, coords: &[Vec<u64>], values: Vec<T>) -> Result<()> {
        if coords.len() != values.len() || values.len() > crate::expression::MAX_BATCH_ELEMENTS {
            return Err(Error::InvalidLayout("invalid copy batch lengths".into()));
        }

        for coord in coords {
            validate::validate_coord(self.shape(), coord)?;
        }

        let mut groups: BTreeMap<u64, Vec<(usize, usize, T)>> = BTreeMap::new();
        let mut sparse: BTreeMap<[u64; 2], Vec<(usize, usize, T)>> = BTreeMap::new();

        for (order, (coord, value)) in coords.iter().zip(values).enumerate() {
            let position = self.block_position_from_base_coord(coord);
            match self.layout() {
                Layout::Dense => groups.entry(position.block_id).or_default().push((
                    order,
                    position.offset_in_block,
                    value,
                )),
                Layout::Sparse { .. } if value != T::ZERO => {
                    let axis = sparse_axis_for_layout(self.layout());
                    sparse
                        .entry([coord[axis], position.block_id])
                        .or_default()
                        .push((order, position.offset_in_block, value));
                }
                Layout::Sparse { .. } => {}
            }
        }

        for (key, updates) in sparse {
            #[cfg(test)]
            copy_metrics::record(|m| m.sparse_lookups += 1);
            if let Some(id) = self.lookup_block_id(&key).await? {
                groups.entry(id).or_default().extend(updates);
            } else {
                let id = rand::random();
                let mut block = self.default_block();
                self.apply_block_updates(&mut block, &updates)?;
                self.write_block(id, block).await?;
                self.upsert_block_id(key.to_vec(), id).await?;

                #[cfg(test)]
                copy_metrics::record(|m| {
                    m.groups += 1;
                    m.block_updates += 1;
                });
            }
        }

        #[cfg(test)]
        copy_metrics::record(|m| m.groups += groups.len());

        for (id, mut updates) in groups {
            updates.sort_unstable_by_key(|(order, _, _)| *order);
            self.write_block_updates(id, &updates).await?;
        }

        Ok(())
    }

    async fn persist_metadata(&self) -> Result<()> {
        let metadata = TensorMetadata::<T>::new(
            self.schema.shape().clone(),
            self.layout(),
            self.storage.storage_schema().block_schema.shape.clone(),
        )?;
        write_metadata_file(self.storage.blocks(), metadata).await
    }

    async fn delete_block(&self, block_id: u64) {
        let mut blocks = self.storage.blocks().write().await;
        blocks.delete(&block_id.to_string()).await;
    }

    /// Resolve one bounded request into physical blocks and scatter positions.
    async fn resolve_read_groups(
        &self,
        request: &BatchRequest,
        mapping: Option<&CoordinateMap>,
    ) -> Result<impl Iterator<Item = (u64, Vec<Run>)>> {
        #[cfg(test)]
        let mapping_time = crate::read_metrics::Timer::new(|m| &mut m.mapping);
        let storage = crate::storage_read::StorageShape {
            shape: self.shape(),
            strides: self.strides(),
            block_shape: self.block_shape(),
            block_strides: self.block_strides(),
            grid_strides: self.grid_strides(),
            sparse_axis: matches!(self.layout(), Layout::Sparse { .. })
                .then(|| sparse_axis_for_layout(self.layout())),
        };
        let mut groups = storage.plan(request, mapping)?;

        #[cfg(test)]
        drop(mapping_time);

        #[cfg(test)]
        let _index_time = crate::read_metrics::Timer::new(|m| &mut m.index);

        // Dense groups already address physical blocks. Only sparse groups
        // need index resolution and alias merging into that same address space.
        if matches!(self.layout(), Layout::Sparse { .. }) {
            let mut resolved = LogicalGroups::new();
            for (key, runs) in groups {
                #[cfg(test)]
                crate::read_metrics::record(|m| m.lookups += 1);
                if let Some(id) = self.lookup_block_id(&key).await? {
                    resolved.entry([0, id]).or_default().extend(runs);
                }
            }
            groups = resolved;
        }

        Ok(groups.into_iter().map(|([_, id], runs)| (id, runs)))
    }

    /// Borrow each needed block once; all requests and results fit one execution batch.
    pub(crate) async fn read_batch(
        &self,
        request: &BatchRequest,
        mapping: Option<&CoordinateMap>,
    ) -> Result<Vec<T>> {
        let groups = self.resolve_read_groups(request, mapping).await?;

        let mut values = vec![T::ZERO; request.len()];

        for (id, positions) in groups {
            #[cfg(test)]
            let access_time = crate::read_metrics::Timer::new(|m| &mut m.access);
            let file = {
                let blocks = self.storage.blocks().read().await;
                blocks.get_file(&id.to_string()).cloned().ok_or_else(|| {
                    Error::from(IoError::new(ErrorKind::NotFound, "Block is missing"))
                })?
            };

            let block = file.read::<Vec<T>>().await?;
            if block.len() != self.block_len() {
                return Err(Error::InvalidLayout("invalid stored block length".into()));
            }

            #[cfg(test)]
            drop(access_time);

            #[cfg(test)]
            crate::read_metrics::record(|m| {
                m.borrows += 1;
                m.borrowed += block.len();
            });

            #[cfg(test)]
            let _scatter_time = crate::read_metrics::Timer::new(|m| &mut m.scatter);

            for run in positions {
                run.scatter(&block, &mut values)?;
            }

            // This block guard is released before the next block is awaited.
        }

        Ok(values)
    }

    pub(crate) fn sparse_key(&self, coords: &[u64], block_offset: u64) -> Vec<u64> {
        let axis = sparse_axis_for_layout(self.layout());
        vec![coords[axis], block_offset]
    }

    /// Read at most one execution batch of index keys, releasing all index guards
    /// before returning. Two prefix ranges resume strictly after the previous key.
    async fn slice_index_page(
        &self,
        last: Option<[u64; 2]>,
        lo: u64,
        hi: u64,
    ) -> Result<Vec<[u64; 2]>> {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        #[cfg(test)]
        let _index_time = crate::read_metrics::Timer::new(|m| &mut m.index);

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
        let sparse_axis = sparse_axis_for_layout(self.layout());
        let mut keys = Vec::new(); // At most 4096 fixed-size keys, never an entire index.
        let table = self.sparse_index()?.read().await;

        for range in ranges {
            // Ordered row streaming descends to leaves between internal separators.
            let mut rows = table.rows(range, &columns, false, Some(&columns)).await?;

            while let Some(row) = rows.try_next().await? {
                if row.len() != 2
                    || row[0] >= self.shape()[sparse_axis]
                    || row[1] >= self.num_blocks()
                {
                    return Err(Error::InvalidLayout(
                        "invalid sparse slice index key".into(),
                    ));
                }
                keys.push([row[0], row[1]]);

                #[cfg(test)]
                crate::read_metrics::record(|m| m.index_entries += 1);
                if keys.len() == crate::expression::MAX_BATCH_ELEMENTS {
                    return Ok(keys);
                }
            }
        }

        Ok(keys)
    }

    /// Numeric consumers visit occupied regions in storage order. One bounded
    /// key page and one request's rectangles are retained, with no index guards.
    pub(crate) fn storage_slice_requests(
        &self,
        slice: crate::slice::Slice,
        mapping: Option<crate::mapping::StorageSlice>,
    ) -> Result<crate::slice::Requests<'_>> {
        if slice.len() <= crate::expression::MAX_BATCH_ELEMENTS as u64
            || matches!(self.layout(), Layout::Dense)
        {
            return Ok(slice.stream());
        }

        let mapping =
            mapping.unwrap_or_else(|| crate::mapping::StorageSlice::identity(self.shape()));
        let sparse_axis = sparse_axis_for_layout(self.layout());
        let (lo, hi) = if let Some((axis, &(_, step))) = mapping
            .axes
            .iter()
            .enumerate()
            .find(|(_, (base, _))| *base == sparse_axis)
        {
            let selection = &slice.axes[axis];
            let (first, last) = match selection {
                crate::request::Axis::Span { .. } => {
                    (selection.at(0), selection.at(selection.len() - 1))
                }
                crate::request::Axis::Selected(values) => (
                    *values.iter().min().expect("nonempty slice"),
                    *values.iter().max().expect("nonempty slice"),
                ),
            };
            (
                mapping.origins[sparse_axis] + first * step,
                mapping.origins[sparse_axis] + last * step,
            )
        } else {
            (mapping.origins[sparse_axis], mapping.origins[sparse_axis])
        };

        struct Cursor {
            last: Option<[u64; 2]>,
            keys: std::vec::IntoIter<[u64; 2]>,
            pending: Option<crate::slice::SliceRequests>,
            // At most one bounded rectangle which did not fit the current batch.
            overflow: Option<crate::request::Cartesian>,
            done: bool,
        }

        let cursor = Cursor {
            last: None,
            keys: Vec::new().into_iter(),
            pending: None,
            overflow: None,
            done: false,
        };
        Ok(futures::stream::try_unfold(
            (cursor, slice, mapping),
            move |(mut cursor, slice, mapping)| async move {
                let mut rectangles = Vec::new();
                let mut len = 0;

                loop {
                    if let Some(rectangle) = cursor.overflow.take().or_else(|| {
                        cursor
                            .pending
                            .as_mut()
                            .and_then(crate::slice::SliceRequests::next_rectangle)
                    }) {
                        if rectangle.len() > crate::expression::MAX_BATCH_ELEMENTS - len {
                            cursor.overflow = Some(rectangle);
                            break;
                        }
                        len += rectangle.len();
                        rectangles.push(rectangle);
                        if len == crate::expression::MAX_BATCH_ELEMENTS {
                            break;
                        }
                        continue;
                    }

                    if cursor.keys.len() == 0 && !cursor.done {
                        let keys = self.slice_index_page(cursor.last, lo, hi).await?;
                        cursor.done = keys.len() < crate::expression::MAX_BATCH_ELEMENTS;
                        cursor.keys = keys.into_iter();
                    }

                    let Some(row) = cursor.keys.next() else {
                        break;
                    };
                    cursor.last = Some(row);
                    let mut bounds = Vec::with_capacity(self.ndim());

                    for axis in 0..self.ndim() {
                        let grid = (row[1] / self.grid_strides()[axis])
                            % self.storage.storage_schema().shape[axis];
                        let start = grid * self.block_shape()[axis];
                        let end = start + self.block_shape()[axis].min(self.shape()[axis] - start);
                        bounds.push((start, end));
                    }

                    let coord = row[0];
                    if !(bounds[sparse_axis].0..bounds[sparse_axis].1).contains(&coord) {
                        return Err(Error::InvalidLayout(
                            "sparse slice key disagrees with its grid block".into(),
                        ));
                    }
                    bounds[sparse_axis] = (coord, coord + 1);
                    if let Some(bounds) = mapping.bounds(&bounds) {
                        cursor.pending = Some(slice.intersect(&bounds)?.requests());
                    }
                }

                if rectangles.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some((
                        BatchRequest::rectangles(rectangles)?,
                        (cursor, slice, mapping),
                    )))
                }
            },
        )
        .boxed())
    }

    async fn lookup_sparse_block_for_coord(
        &self,
        base_coord: &[u64],
        block_offset: u64,
    ) -> Result<Option<u64>> {
        let key = self.sparse_key(base_coord, block_offset);
        self.lookup_block_id(&key).await
    }

    async fn plan_sparse_write(
        &self,
        base_coord: &[u64],
        block_offset: u64,
        value: T,
    ) -> Result<SparseWriteAction> {
        if let Some(block_id) = self
            .lookup_sparse_block_for_coord(base_coord, block_offset)
            .await?
        {
            if value == T::default() {
                return Ok(SparseWriteAction::DeleteRow(block_id));
            }
            return Ok(SparseWriteAction::Write(block_id));
        }

        if value == T::default() {
            return Ok(SparseWriteAction::NoOp);
        }

        let block_id: u64 = rand::random();
        Ok(SparseWriteAction::CreateBlockAndWrite(block_id))
    }

    async fn new_storage(
        blocks: DirLock<FE>,
        index: Option<TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>>,
        schema: TensorSchema,
        storage_schema: StorageSchema,
    ) -> Result<Self> {
        let storage = match index {
            Some(si) => Storage::Sparse(SparseStorage {
                blocks,
                index: si,
                storage_schema,
            }),
            None => Storage::Dense(DenseStorage {
                blocks,
                storage_schema,
            }),
        };

        let tensor = Self {
            storage: Arc::new(storage),
            schema,
            _dtype: std::marker::PhantomData,
        };

        tensor.initialize_dense_blocks().await?;
        tensor.persist_metadata().await?;

        Ok(tensor)
    }

    fn sparse_index(&self) -> Result<&SparseIndex<FE>> {
        self.storage.index().ok_or_else(|| {
            Error::SparseIndex(
                "Operations with index are not available for dense tensor".to_string(),
            )
        })
    }

    async fn is_empty_block(&self, block_id: u64) -> Result<bool> {
        match self.read_block(block_id).await? {
            Some(block) => Ok(block.iter().all(|v| *v == T::default())),
            None => Ok(true),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorGeometry and TensorArray impls
// ---------------------------------------------------------------------------
impl<FE, T> TensorGeometry for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type DType = T;

    fn dtype(&self) -> NumberType {
        <T as number_general::DType>::dtype()
    }

    fn layout(&self) -> Layout {
        self.storage.storage_schema().layout
    }

    fn shape(&self) -> &[u64] {
        self.schema.shape()
    }
}

impl<FE, T> TensorArray for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    fn strides(&self) -> &[u64] {
        self.schema.strides()
    }
}

// ---------------------------------------------------------------------------
// TensorRead impls
// ---------------------------------------------------------------------------
impl<FE, T> TensorRead for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn read_blocks(&self) -> Result<crate::ValueBlockStream<'_, Self::DType>> {
        let requests = crate::request::linear_requests(self.shape())?;
        Ok(crate::expression::ordered_batches(self, requests)
            .map_ok(|(_, batch)| batch.values)
            .boxed())
    }

    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, Self::DType>> {
        crate::expression::coordinate_blocks(self)
    }

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move { Ok(self.read_batch(&BatchRequest::point(coord), None).await?[0]) })
    }
}

// ---------------------------------------------------------------------------
// TensorWrite impls
// ---------------------------------------------------------------------------

impl<FE, T> TensorWrite for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn write_value<'a>(
        &'a self,
        coord: &'a [u64],
        value: Self::DType,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            validate::validate_coord(self.schema.shape(), coord)?;
            let BlockPosition {
                block_id: block_grid_id,
                offset_in_block,
            } = self.block_position_from_base_coord(coord);

            match self.layout() {
                Layout::Dense => {
                    if !self.block_exists(block_grid_id).await {
                        self.write_block(block_grid_id, self.default_block())
                            .await?;
                    }

                    self.write_block_updates(block_grid_id, &[(0, offset_in_block, value)])
                        .await
                }
                Layout::Sparse { .. } => {
                    match self.plan_sparse_write(coord, block_grid_id, value).await? {
                        SparseWriteAction::Write(block_id) => {
                            self.write_block_updates(block_id, &[(0, offset_in_block, value)])
                                .await
                        }
                        SparseWriteAction::DeleteRow(block_id) => {
                            let key = self.sparse_key(coord, block_grid_id);
                            self.delete_row(key).await?;
                            if self.is_empty_block(block_id).await? {
                                self.delete_block(block_id).await;
                            }

                            Ok(())
                        }
                        SparseWriteAction::CreateBlockAndWrite(block_id) => {
                            self.write_block(block_id, self.default_block()).await?;
                            let key = self.sparse_key(coord, block_grid_id);
                            self.upsert_block_id(key, block_id).await?;
                            self.write_block_updates(block_id, &[(0, offset_in_block, value)])
                                .await?;
                            Ok(())
                        }
                        SparseWriteAction::NoOp => Ok(()),
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// TensorBlockStore impls
// ---------------------------------------------------------------------------

impl<FE, T> TensorBlockStore for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Block = Vec<T>;

    fn read_block<'a>(&'a self, block_id: u64) -> BoxFuture<'a, Result<Option<Self::Block>>> {
        Box::pin(async move {
            let blocks = self.storage.blocks().read().await;
            if let Some(file) = blocks.get_file(&block_id.to_string()) {
                let guard = file.read::<Vec<T>>().await?;
                if guard.len() != self.block_len() {
                    return Err(Error::InvalidLayout("invalid stored block length".into()));
                }

                Ok(Some(guard.clone()))
            } else {
                Ok(None)
            }
        })
    }

    fn write_block<'a>(&'a self, block_id: u64, block: Self::Block) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if block.len() != self.block_len() {
                return Err(Error::InvalidLayout(
                    "block length must match storage schema".into(),
                ));
            }

            let file = {
                let blocks = self.storage.blocks().read().await;
                blocks.get_file(&block_id.to_string()).cloned()
            };
            if let Some(file) = file {
                let mut guard = file.write::<Vec<T>>().await?;
                *guard = block;
                Ok(())
            } else {
                let mut blocks = self.storage.blocks().write().await;
                let size = block
                    .len()
                    .checked_mul(std::mem::size_of::<T>())
                    .ok_or_else(|| Error::InvalidLayout("block byte size overflow".into()))?;
                blocks
                    .create_file(block_id.to_string(), block, size)
                    .await?;
                Ok(())
            }
        })
    }
}

// ---------------------------------------------------------------------------
// TensorSparseIndex impls
// ---------------------------------------------------------------------------

impl<FE, T> TensorSparseIndex for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn lookup_block_id<'a>(&'a self, key: &'a [u64]) -> BoxFuture<'a, Result<Option<u64>>> {
        Box::pin(async move {
            let index = self.sparse_index()?;

            let index_lock = index.read().await;
            let Some(row) = index_lock.get_row(key).await? else {
                return Ok(None);
            };

            Ok(row.get(2).copied())
        })
    }

    fn upsert_block_id<'a>(&'a self, key: Vec<u64>, block_id: u64) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let index = self.sparse_index()?;

            let mut index_lock = index.write().await;
            index_lock
                .upsert(key, vec![block_id])
                .await
                .map(|_| ())
                .map_err(Error::from)
        })
    }

    fn delete_row<'a>(&'a self, key: Vec<u64>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            let index = self.sparse_index()?;

            let mut index_lock = index.write().await;
            index_lock.delete_row(&key).await.map_err(Error::from)
        })
    }
}

// ---------------------------------------------------------------------------
// TensorWriteBulk implementation
// ---------------------------------------------------------------------------
impl<FE, T> TensorWriteBulk for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn write_values<'a>(
        &'a self,
        range: Range,
        values: Vec<Self::DType>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // Cardinality is checked without expanding the selected range.
            let coords = validate::iter_range_coords(self.shape(), &range)?;
            if coords.remaining() != values.len() as u64 {
                return Err(Error::DataMismatch(format!(
                    "expected {} values for range but got {}",
                    coords.remaining(),
                    values.len()
                )));
            }

            for (coord, value) in coords.zip(values) {
                self.write_value(&coord, value).await?;
            }

            Ok(())
        })
    }

    fn write_tensor<'a, Src>(&'a self, other: &'a Src) -> BoxFuture<'a, Result<()>>
    where
        Src: TensorRead<DType = Self::DType> + Sync + ?Sized,
    {
        Box::pin(async move {
            if self.shape() != other.shape() {
                return Err(Error::InvalidLayout(format!(
                    "cannot write tensor of shape {:?} into tensor of shape {:?}",
                    other.shape(),
                    self.shape()
                )));
            }

            for coord in
                validate::iter_range_coords(self.shape(), &validate::full_range(self.shape()))?
            {
                let value = other.read_value(&coord).await?;
                self.write_value(&coord, value).await?;
            }

            Ok(())
        })
    }

    fn fill<'a>(&'a self, value: Self::DType) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for coord in
                validate::iter_range_coords(self.shape(), &validate::full_range(self.shape()))?
            {
                self.write_value(&coord, value).await?;
            }

            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------
async fn load_metadata_file<FE, T>(blocks: &DirLock<FE>) -> Result<(TensorSchema, StorageSchema)>
where
    FE: AsType<TensorMetadata<T>> + FileLoad,
    T: TensorElement,
{
    let file = blocks
        .read()
        .await
        .get_file(METADATA)
        .cloned()
        .ok_or_else(|| Error::InvalidSchema("missing tensor metadata file".into()))?;
    file.read::<TensorMetadata<T>>().await?.schemas()
}

async fn write_metadata_file<FE, T>(blocks: &DirLock<FE>, metadata: TensorMetadata<T>) -> Result<()>
where
    FE: AsType<TensorMetadata<T>> + FileLoad,
    T: TensorElement,
{
    let existing = blocks.read().await.get_file(METADATA).cloned();
    if let Some(file) = existing {
        *file.write::<TensorMetadata<T>>().await? = metadata;
    } else {
        let size = metadata.size();
        blocks
            .write()
            .await
            .create_file(METADATA.to_string(), metadata, size)
            .await?;
    }

    Ok(())
}

/// Creation and metadata loading validate axis bounds through StorageSchema.
/// Call only with a stored tensor's validated layout; never substitute another axis.
fn sparse_axis_for_layout(layout: Layout) -> usize {
    match layout {
        Layout::Sparse { axis } => axis.unwrap_or(0),
        Layout::Dense => 0,
    }
}

fn validate_tensor_dtype<T: TensorElement>(dtype: NumberType) -> Result<()> {
    if dtype != <T as number_general::DType>::dtype() {
        return Err(Error::InvalidSchema(format!(
            "tensor dtype mismatch: schema {:?} != tensor {:?}",
            dtype,
            <T as number_general::DType>::dtype(),
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn typed_tensor_rejects_schema_dtype_mismatch() {
        let err = validate_tensor_dtype::<f32>(NumberType::Float(FloatType::F64))
            .expect_err("expected mismatch");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }
}

#[cfg(test)]
mod sparse_axis_tests {
    use super::*;

    #[test]
    fn validated_layout_selects_its_axis() {
        for (layout, expected) in [
            (Layout::Dense, 0),
            (Layout::Sparse { axis: None }, 0),
            (Layout::Sparse { axis: Some(0) }, 0),
            (Layout::Sparse { axis: Some(1) }, 1),
            (Layout::Sparse { axis: Some(2) }, 2),
        ] {
            assert_eq!(sparse_axis_for_layout(layout), expected);
        }
    }
}

#[cfg(test)]
mod sparse_lifecycle_tests {
    use std::path::PathBuf;

    use crate::Shape;
    use ha_ndarray::shape;

    use crate::test_support::{FsEntry as TestFE, cleanup, new_dir, open_dir};

    use super::*;

    async fn create_sparse(
        name: &str,
        shape: Shape,
        max_capacity: usize,
        axis: Option<usize>,
    ) -> (PathBuf, Tensor<TestFE, f32>) {
        let (root, dir) = new_dir(name).await;
        let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape).expect("schema");
        let tensor =
            Tensor::<TestFE, f32>::create(dir, schema, Layout::Sparse { axis }, max_capacity)
                .await
                .expect("create sparse");
        (root, tensor)
    }

    async fn block_id_for_coord(tensor: &Tensor<TestFE, f32>, coord: &[u64]) -> Option<u64> {
        let BlockPosition { block_id, .. } = tensor.block_position_from_base_coord(coord);
        tensor
            .lookup_sparse_block_for_coord(coord, block_id)
            .await
            .expect("lookup")
    }

    #[tokio::test]
    async fn slice_index_pages_resume_within_and_between_coordinates() {
        use crate::TensorReduceAll;

        let entries = crate::expression::MAX_BATCH_ELEMENTS + 1;

        for axis in [None, Some(1)] {
            let (root, tensor) =
                create_sparse("slice_pages", shape![1_u64, entries as u64], 1, axis).await;

            for col in 0..entries as u64 {
                tensor.write_value(&[0, col], 1.).await.unwrap();
            }

            let hi = if axis.is_none() {
                0
            } else {
                entries as u64 - 1
            };
            let first = tensor.slice_index_page(None, 0, hi).await.unwrap();
            assert_eq!(first.len(), crate::expression::MAX_BATCH_ELEMENTS);
            let second = tensor
                .slice_index_page(first.last().copied(), 0, hi)
                .await
                .unwrap();
            assert_eq!(second.len(), 1);
            assert!(first.last().unwrap() < second.first().unwrap());
            assert!(
                tensor
                    .slice_index_page(second.last().copied(), 0, hi)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(tensor.sum_all().await.unwrap(), entries as f32);
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn sparse_zero_to_nonzero_creates_row_and_block() {
        let (root, tensor) = create_sparse("zero_to_nonzero", shape![2, 3, 4], 4, Some(1)).await;

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "no row before write"
        );

        tensor.write_value(&[0, 1, 2], 1.0).await.expect("write");
        let block_id = block_id_for_coord(&tensor, &[0, 1, 2])
            .await
            .expect("row must exist");
        assert!(
            tensor.read_block(block_id).await.expect("block").is_some(),
            "block must be materialized after first nonzero write"
        );
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 1.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_nonzero_to_zero_remove_row_policy() {
        let (root, tensor) = create_sparse("remove_row", shape![2, 3, 4], 4, Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("write nz");
        assert!(block_id_for_coord(&tensor, &[0, 1, 2]).await.is_some());

        tensor.write_value(&[0, 1, 2], 0.0).await.expect("write z");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "RemoveRow policy must drop the row when value becomes default"
        );
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 0.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_overwrite_nonzero_preserves_row_id() {
        let (root, tensor) = create_sparse("overwrite", shape![2, 3, 4], 4, Some(1)).await;

        tensor.write_value(&[0, 1, 2], 1.0).await.expect("write 1");
        let id_a = block_id_for_coord(&tensor, &[0, 1, 2])
            .await
            .expect("must exist");

        tensor.write_value(&[0, 1, 2], 2.0).await.expect("write 2");
        let id_b = block_id_for_coord(&tensor, &[0, 1, 2])
            .await
            .expect("must exist");

        assert_eq!(id_a, id_b, "block_id must be stable across nonzero updates");
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 2.0);

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_write_zero_to_new_coord_is_noop() {
        let (root, tensor) = create_sparse("noop", shape![2, 3, 4], 4, Some(1)).await;

        tensor
            .write_value(&[0, 1, 2], 0.0)
            .await
            .expect("write zero");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "zero write to new coord must not create a row"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn oversized_evaluation_is_rejected_before_reading_coordinates() {
        let (root, tensor) = create_sparse("batch_bound", shape![1], 1, None).await;
        let coords = vec![vec![99]; crate::expression::MAX_BATCH_ELEMENTS + 1];
        let error = BatchRequest::explicit(coords).err().unwrap();
        let _ = tensor;
        assert!(matches!(error, Error::InvalidLayout(_)));
        assert!(error.to_string().contains("coordinate batch"));
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn storage_batches_group_blocks_and_preserve_order() {
        for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
            let (root, dir) = new_dir("read_groups").await;
            let schema =
                TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 4]).unwrap();
            let tensor = Tensor::<TestFE, f32>::create(dir, schema, layout, 4)
                .await
                .unwrap();
            tensor.write_value(&[0, 0], 2.).await.unwrap();
            tensor.write_value(&[0, 3], 3.).await.unwrap();
            tensor.write_value(&[1, 1], 4.).await.unwrap();

            let coords = vec![vec![1, 1], vec![0, 3], vec![0, 0], vec![0, 3]];
            let groups: BTreeMap<_, _> = tensor
                .resolve_read_groups(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                .await
                .unwrap()
                .collect();
            assert_eq!(groups.len(), 2);
            assert_eq!(
                groups.values().flatten().map(|run| run.len).sum::<usize>(),
                coords.len()
            );
            assert!(groups.values().any(|positions| {
                positions
                    .iter()
                    .flat_map(|run| {
                        (0..run.len).map(move |i| {
                            (
                                run.output + i,
                                (run.offset as i128 + i as i128 * run.stride as i128) as usize,
                            )
                        })
                    })
                    .collect::<Vec<_>>()
                    == vec![(1, 3), (2, 0), (3, 3)]
            }));
            assert_eq!(
                tensor
                    .read_batch(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                    .await
                    .unwrap(),
                vec![4., 3., 2., 3.]
            );
            assert!(
                tensor
                    .read_batch(&BatchRequest::point(&[2, 0]), None)
                    .await
                    .is_err()
            );
            assert!(BatchRequest::explicit(vec![vec![0, 0]; 4097]).is_err());

            // Validate the entire borrowed block, even for one selected value.
            let id = *groups.keys().next().unwrap();
            let file = tensor
                .storage
                .blocks()
                .read()
                .await
                .get_file(&id.to_string())
                .unwrap()
                .clone();
            file.write::<Vec<f32>>().await.unwrap().pop();
            assert!(matches!(
                tensor
                    .read_batch(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                    .await,
                Err(Error::InvalidLayout(_))
            ));
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn invalid_affine_requests_fail_before_storage_io() {
        let (root, tensor) = create_sparse("invalid_affine", shape![2, 4], 4, None).await;
        let mut map = CoordinateMap::identity(shape![2, 4], tensor.strides());
        map.base_offset = -1;
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                assert!(matches!(
                    tensor
                        .read_batch(&BatchRequest::linear(0, 8).unwrap(), Some(&map))
                        .await,
                    Err(Error::InvalidCoord(_))
                ));
                crate::read_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    assert_eq!((m.lookups, m.borrows), (0, 0));
                });
            })
            .await;
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn storage_batch_uses_sparse_keys_not_grid_block_ids() {
        let (root, tensor) = create_sparse("batch_keys", shape![3, 2], 6, Some(0)).await;
        tensor.write_value(&[0, 0], 2.).await.unwrap();
        tensor.write_value(&[2, 1], 3.).await.unwrap();

        let coords = vec![vec![0, 0], vec![1, 0], vec![2, 1], vec![0, 0]];
        assert!(
            coords
                .iter()
                .all(|coord| tensor.block_position_from_base_coord(coord).block_id == 0)
        );
        let groups: BTreeMap<_, _> = tensor
            .resolve_read_groups(&BatchRequest::explicit(coords.clone()).unwrap(), None)
            .await
            .unwrap()
            .collect();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups.values().flatten().map(|run| run.len).sum::<usize>(),
            3
        );
        assert_eq!(
            tensor
                .read_batch(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                .await
                .unwrap(),
            vec![2., 0., 3., 2.]
        );
        // Two logical keys may name the same physical block. Lookup each key
        // once, merge aliases, and retain duplicate request outputs.
        let id = tensor.lookup_block_id(&[0, 0]).await.unwrap().unwrap();
        let mut block = tensor.read_block(id).await.unwrap().unwrap();
        block[5] = 3.;
        tensor.write_block(id, block).await.unwrap();
        assert!(tensor.delete_row(vec![2, 0]).await.unwrap());
        tensor.upsert_block_id(vec![2, 0], id).await.unwrap();
        assert_eq!(tensor.lookup_block_id(&[2, 0]).await.unwrap(), Some(id));
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                assert_eq!(
                    tensor
                        .read_batch(&BatchRequest::explicit(coords).unwrap(), None)
                        .await
                        .unwrap(),
                    vec![2., 0., 3., 2.]
                );
                crate::read_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    assert_eq!((m.lookups, m.borrows), (3, 1));
                });
            })
            .await;
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn dense_matrix_products_have_no_output_support_mask() {
        use crate::{TensorMatMul, TensorTransform};
        let (root, dir) = new_dir("dense_matrix_support").await;
        let dense = Tensor::<TestFE, f32>::create(
            dir,
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 2]).unwrap(),
            Layout::Dense,
            2,
        )
        .await
        .unwrap();
        let (sparse_root, sparse) =
            create_sparse("sparse_matrix_support", shape![2, 2], 2, None).await;
        let left = dense.view().matmul(&sparse.view()).await.unwrap();
        let right = sparse
            .view()
            .matmul(&dense.view().transpose(None).unwrap())
            .await
            .unwrap();
        assert!(
            crate::expression::Expression::build(&left, &BatchRequest::point(&[0, 0]))
                .await
                .unwrap()
                .support
                .is_none()
        );
        assert!(
            crate::expression::Expression::build(&right, &BatchRequest::point(&[0, 0]))
                .await
                .unwrap()
                .support
                .is_none()
        );
        cleanup(&root).await;
        cleanup(&sparse_root).await;
    }

    #[derive(Clone)]
    struct CompactSource<'a>(&'a Tensor<TestFE, f32>);

    impl TensorGeometry for CompactSource<'_> {
        type DType = f32;

        fn dtype(&self) -> NumberType {
            self.0.dtype()
        }

        fn shape(&self) -> &[u64] {
            self.0.shape()
        }

        fn layout(&self) -> Layout {
            self.0.layout()
        }
    }

    impl crate::expression::Expression for CompactSource<'_> {
        fn build<'a>(
            &'a self,
            request: &'a BatchRequest,
        ) -> BoxFuture<'a, Result<crate::expression::Batch<f32>>> {
            Box::pin(async move {
                assert!(matches!(
                    request.kind(),
                    crate::request::RequestKind::Rectangles(_)
                ));
                self.0.view().build(request).await
            })
        }
    }

    #[tokio::test]
    async fn compact_requests_survive_elementwise_composition() {
        use crate::{TensorCompareScalar, TensorMath, TensorUnary, TensorWhere};
        let (root, tensor) = create_sparse("compact_composition", shape![2, 2], 2, None).await;
        tensor.write_value(&[1, 1], 2.).await.unwrap();
        let source = CompactSource(&tensor);
        let values = source.round().await.unwrap().add(&source).await.unwrap();
        let condition = values.gt_scalar(0.).await.unwrap();
        let selected = condition.cond(&values, &source).await.unwrap();
        let request = BatchRequest::rectangles(vec![
            crate::request::Cartesian::new(vec![
                crate::request::Axis::range(0, 2),
                crate::request::Axis::range(0, 2),
            ])
            .unwrap(),
        ])
        .unwrap();
        assert_eq!(
            crate::expression::evaluate_batch(&selected, &request)
                .await
                .unwrap()
                .values,
            vec![0., 0., 0., 4.]
        );

        // An invalid mapped position is rejected before any data/index read.
        let mut mapping = crate::mapping::CoordinateMap::identity(shape![2, 2], tensor.strides());
        mapping.base_offset = 4;
        assert!(matches!(
            tensor.read_batch(&request, Some(&mapping)).await,
            Err(Error::InvalidCoord(_))
        ));
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn copy_batches_group_physical_blocks_and_sparse_keys() {
        for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
            let (root, dir) = new_dir("copy_groups").await;
            let tensor = Tensor::<TestFE, f32>::create(
                dir.clone(),
                TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 4]).unwrap(),
                layout,
                8,
            )
            .await
            .unwrap();
            let coords = vec![vec![1, 3], vec![0, 2], vec![1, 1], vec![0, 0]];
            let expected = if matches!(layout, Layout::Dense) {
                1
            } else {
                2
            };
            copy_metrics::CURRENT
                .scope(Default::default(), async {
                    for _ in 0..2 {
                        tensor
                            .write_copy_batch(&coords, vec![3., 2., 1., 4.])
                            .await
                            .unwrap();
                    }
                    copy_metrics::CURRENT.with(|m| {
                        let m = m.borrow();
                        assert_eq!(m.groups, 2 * expected);
                        assert_eq!(m.block_updates, 2 * expected);
                        assert_eq!(m.sparse_lookups, if expected == 1 { 0 } else { 4 });
                    });
                })
                .await;
            for (coord, value) in coords.iter().zip([3., 2., 1., 4.]) {
                assert_eq!(tensor.read_value(coord).await.unwrap(), value);
            }
            tensor.sync().await.unwrap();
            drop(tensor);
            drop(dir);
            let loaded = Tensor::<TestFE, f32>::load(open_dir(&root).unwrap())
                .await
                .unwrap();
            assert_eq!(loaded.read_value(&[1, 3]).await.unwrap(), 3.);
            cleanup(&root).await;
        }
    }

    #[tokio::test]
    async fn copy_batches_coalesce_existing_physical_blocks() {
        let (root, tensor) = create_sparse("copy_physical", shape![2, 4], 8, None).await;
        tensor.write_value(&[0, 0], 1.).await.unwrap();
        let id = tensor.lookup_block_id(&[0, 0]).await.unwrap().unwrap();
        tensor.upsert_block_id(vec![1, 0], id).await.unwrap();
        copy_metrics::CURRENT
            .scope(Default::default(), async {
                tensor
                    .write_copy_batch(&[vec![1, 1], vec![0, 1], vec![1, 1]], vec![2., 3., 4.])
                    .await
                    .unwrap();
                copy_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    assert_eq!((m.groups, m.block_updates, m.sparse_lookups), (1, 1, 2));
                });
            })
            .await;
        assert_eq!(tensor.read_value(&[1, 1]).await.unwrap(), 4.);
        assert_eq!(tensor.read_value(&[0, 1]).await.unwrap(), 3.);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn copy_batch_rejects_invalid_input_and_corrupt_blocks() {
        let (root, dir) = new_dir("copy_invalid").await;
        let tensor = Tensor::<TestFE, f32>::create(
            dir.clone(),
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![4]).unwrap(),
            Layout::Dense,
            4,
        )
        .await
        .unwrap();

        for (coords, values) in [
            (vec![vec![0], vec![4]], vec![1., 2.]),
            (vec![vec![0]], vec![]),
            (vec![vec![0]; 4097], vec![1.; 4097]),
        ] {
            assert!(tensor.write_copy_batch(&coords, values).await.is_err());
            assert_eq!(tensor.read_value(&[0]).await.unwrap(), 0.);
        }

        let mut block = vec![0.; 4];
        assert!(
            tensor
                .apply_block_updates(&mut block, &[(0, 0, 1.), (1, 4, 2.)])
                .is_err()
        );
        assert_eq!(block, vec![0.; 4]);
        let blocks = dir.read().await.get_dir(BLOCKS).unwrap().clone();
        let file = blocks.read().await.get_file("0").unwrap().clone();
        file.write::<Vec<f32>>().await.unwrap().pop();
        assert!(tensor.write_copy_batch(&[vec![0]], vec![1.]).await.is_err());
        assert_eq!(*file.read::<Vec<f32>>().await.unwrap(), vec![0.; 3]);
        cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_copy_zeros_do_not_create_storage() {
        let (root, tensor) = create_sparse("copy_zeros", shape![2, 4], 8, None).await;
        copy_metrics::CURRENT
            .scope(Default::default(), async {
                tensor
                    .write_copy_batch(&[vec![0, 0], vec![1, 1]], vec![0., -0.])
                    .await
                    .unwrap();
                copy_metrics::CURRENT.with(|m| {
                    let m = m.borrow();
                    assert_eq!((m.groups, m.block_updates, m.sparse_lookups), (0, 0, 0));
                });
            })
            .await;
        assert!(tensor.lookup_block_id(&[0, 0]).await.unwrap().is_none());
        assert!(tensor.lookup_block_id(&[1, 0]).await.unwrap().is_none());
        assert_eq!(tensor.storage.blocks().read().await.files().count(), 1); // metadata only
        cleanup(&root).await;
    }
}

// Task-local counters cannot mix observations from concurrent test fixtures.
// No instrumentation or configuration is present in production builds.
#[cfg(test)]
pub(crate) mod copy_metrics {
    use std::cell::RefCell;
    use std::time::Duration;

    #[derive(Default, Debug)]
    pub(crate) struct Metrics {
        pub initialize: Duration,
        pub consume: Duration,
        pub update: Duration,
        pub overlap: Duration,
        pub groups: usize,
        pub block_updates: usize,
        pub sparse_lookups: usize,
    }

    tokio::task_local! {
        pub(crate) static CURRENT: RefCell<Metrics>;
    }

    pub(crate) fn record(update: impl FnOnce(&mut Metrics)) {
        let _ = CURRENT.try_with(|metrics| update(&mut metrics.borrow_mut()));
    }
}
