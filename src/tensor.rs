use std::collections::VecDeque;
use std::io::{Error as IoError, ErrorKind};
use std::sync::Arc;

use b_table::{TableLock, collate::Collator};
use destream::{de, en};
use freqfs::{DirLock, FileLoad};
use futures::StreamExt as _;
use futures::stream;
use ha_ndarray::{Axes, Range};
use safecast::AsType;

use crate::error::{Error, Result};
use crate::schema::{
    BlockPosition, DType, Layout, MAX_BLOCK_CAPACITY, SparseIndexSchema, SparseTableSchema,
    StorageSchema, TensorSchema,
};
use crate::traits::{
    BoxFuture, SparseElementStream, TensorArray, TensorBlockStore, TensorGeometry, TensorRead,
    TensorReadBulk, TensorSparseIndex, TensorWrite, TensorWriteBulk,
};
use crate::validate::{self, validate_coord};
use crate::view::TensorView;

const BLOCKS: &str = "blocks";
const INDEX: &str = "index";
const METADATA: &str = "metadata";
const METADATA_VERSION: u32 = 2;

pub trait TensorElement:
    Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + de::FromStream<Context = ()>
    + for<'en> en::ToStream<'en>
    + for<'en> en::IntoStream<'en>
{
    const DTYPE: DType;
}

impl TensorElement for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorElement for f64 {
    const DTYPE: DType = DType::F64;
}

pub trait TensorFileEntry<T: TensorElement>:
    FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<T>> + Send + Sync + 'static
{
}

impl<FE, T> TensorFileEntry<T> for FE
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<T>> + Send + Sync + 'static,
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

struct SparseIterState<T> {
    pending_blocks: VecDeque<(u64, u64)>,
    buffered: VecDeque<(Vec<u64>, T)>,
    range: Range,
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
    pub async fn create(
        dir: DirLock<FE>,
        schema: TensorSchema,
        layout: Layout,
        max_capacity: usize,
    ) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
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

    pub async fn load(dir: DirLock<FE>) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let mut dir_guard = dir.try_write()?;
        let blocks_dir = dir_guard.get_or_create_dir(BLOCKS.to_string())?;
        let (schema, storage_schema) = load_metadata_file(&blocks_dir).await?;
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
        let block_shape = self.block_shape();
        let block_strides = self.block_strides();
        let grid_strides = self.grid_strides();

        let mut block_id: u64 = 0;
        let mut offset_in_block: usize = 0;
        for (i, coord) in base_coord.iter().enumerate() {
            let block_dim = block_shape[i] as u64;
            block_id += (coord / block_dim) * grid_strides[i] as u64;
            offset_in_block += ((coord % block_dim) as usize) * block_strides[i];
        }
        BlockPosition {
            block_id,
            offset_in_block,
        }
    }

    pub(crate) fn block_shape(&self) -> &[usize] {
        &self.storage.storage_schema().block_schema.shape
    }

    pub(crate) fn block_strides(&self) -> &[usize] {
        &self.storage.storage_schema().block_schema.strides
    }

    pub(crate) fn grid_strides(&self) -> &[usize] {
        &self.storage.storage_schema().strides
    }

    pub(crate) fn grid_shape(&self) -> &[usize] {
        &self.storage.storage_schema().shape
    }

    pub(crate) fn block_origin_from_grid_id(&self, block_grid_id: u64) -> Vec<u64> {
        let block_shape = self.block_shape();
        let grid_shape = self.grid_shape();
        let grid_strides = self.grid_strides();

        (0..block_shape.len())
            .map(|axis| {
                let grid_coord =
                    (block_grid_id / grid_strides[axis] as u64) as usize % grid_shape[axis];
                (grid_coord * block_shape[axis]) as u64
            })
            .collect()
    }

    pub(crate) fn base_coord_from_block_offset(
        &self,
        origin: &[u64],
        offset_in_block: usize,
    ) -> Vec<u64> {
        let block_shape = self.block_shape();
        let block_strides = self.block_strides();

        origin
            .iter()
            .enumerate()
            .map(|(axis, &start)| {
                let local = (offset_in_block / block_strides[axis]) % block_shape[axis];
                start + local as u64
            })
            .collect()
    }

    pub(crate) fn num_blocks(&self) -> u64 {
        self.storage
            .storage_schema()
            .shape
            .iter()
            .product::<usize>() as u64
    }

    pub(crate) fn block_len(&self) -> usize {
        self.block_shape().iter().product::<usize>().max(1)
    }

    pub async fn compact_sparse(&self) -> Result<()> {
        let index = self.sparse_index()?;
        let all_rows = {
            let guard = index.read().await;
            let mut rows = guard.into_rows().await.map_err(Error::from)?;
            let mut collected: Vec<Vec<u64>> = Vec::new();
            while let Some(row) = rows.next().await {
                let row = row.map_err(Error::from)?;
                collected.push(row.to_vec());
            }
            collected
        };

        let mut to_delete: Vec<(Vec<u64>, u64)> = Vec::new();
        for row in &all_rows {
            let key = vec![row[0], row[1]];
            let block_id = row[2];
            let all_zero = self.is_empty_block(block_id).await?;
            if all_zero {
                to_delete.push((key, block_id));
            }
        }

        for (key, block_id) in to_delete {
            self.delete_row(key).await?;
            self.delete_block(block_id).await;
        }

        Ok(())
    }

    fn default_block(&self) -> Vec<T> {
        vec![T::default(); self.block_len()]
    }

    async fn materialize(&self) -> Result<()> {
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

    async fn write_value_to_block(
        &self,
        block_id: u64,
        offset_in_block: usize,
        value: T,
    ) -> Result<()> {
        let mut block = match self.read_block(block_id).await? {
            Some(block) => block,
            None => {
                return Err(IoError::new(ErrorKind::NotFound, "Missing block".to_string()).into());
            }
        };
        validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
        block[offset_in_block] = value;
        self.write_block(block_id, block).await
    }

    async fn persist_metadata(&self) -> Result<()>
    where
        FE: AsType<String> + From<String>,
    {
        let payload = encode_schema(&self.schema, self.storage.storage_schema());
        let _ = decode_schema(&payload)?;
        write_metadata_file(self.storage.blocks(), METADATA, &payload).await
    }

    async fn delete_block(&self, block_id: u64) {
        let mut blocks = self.storage.blocks().write().await;
        blocks.delete(&block_id.to_string()).await;
    }

    pub(crate) fn sparse_key(&self, coords: &[u64], block_offset: u64) -> Vec<u64> {
        let axis = sparse_axis_for_layout(self.layout(), coords.len());
        vec![coords[axis], block_offset]
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
    ) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
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

        tensor.materialize().await?;
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

    fn dtype(&self) -> Self::DType {
        T::default()
    }

    fn layout(&self) -> Layout {
        self.storage.storage_schema().layout
    }

    fn shape(&self) -> &[usize] {
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

    fn strides(&self) -> &[usize] {
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
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            validate::validate_coord(self.schema.shape(), coord)?;

            let BlockPosition {
                block_id: block_grid_id,
                offset_in_block,
            } = self.block_position_from_base_coord(coord);

            let block_id = match self.layout() {
                Layout::Dense => Some(block_grid_id),
                Layout::Sparse { .. } => {
                    self.lookup_sparse_block_for_coord(coord, block_grid_id)
                        .await?
                }
            };

            let Some(id) = block_id else {
                return Ok(T::default());
            };

            let Some(block) = self.read_block(id).await? else {
                return Err(
                    IoError::new(ErrorKind::NotFound, "Block is missing".to_string()).into(),
                );
            };

            validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
            Ok(block[offset_in_block])
        })
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let base_order: Vec<usize> = (0..self.ndim()).collect();
            let requested_order: Vec<usize> = requested_order.into_iter().collect();

            if requested_order != base_order {
                return Err(Error::UnsupportedSparseIterationOrder {
                    requested_order,
                    base_order,
                    hint: "materialize or perform external sort for incompatible order".to_string(),
                });
            }

            if range.len() != self.ndim() {
                return Err(Error::InvalidLayout(format!(
                    "range has {} axes but tensor has {} dimensions",
                    range.len(),
                    self.ndim()
                )));
            }

            let index = self.sparse_index()?;

            let mut block_refs: Vec<(u64, u64)> = {
                let guard = index.read().await;
                let mut rows = guard.into_rows().await.map_err(Error::from)?;
                let mut refs = Vec::new();
                while let Some(row) = rows.next().await {
                    let row = row.map_err(Error::from)?;
                    refs.push((row[1], row[2]));
                }
                refs
            };
            block_refs.sort_by_key(|&(block_grid_id, _)| block_grid_id);

            let state = SparseIterState {
                pending_blocks: block_refs.into(),
                buffered: VecDeque::new(),
                range,
            };

            let elements = stream::unfold(state, move |mut state| async move {
                loop {
                    if let Some(item) = state.buffered.pop_front() {
                        return Some((Ok(item), state));
                    }

                    let (block_grid_id, block_id) = state.pending_blocks.pop_front()?;

                    let block = match self.read_block(block_id).await {
                        Ok(Some(block)) => block,
                        Ok(None) => {
                            let err: Error =
                                IoError::new(ErrorKind::NotFound, "Block is missing".to_string())
                                    .into();
                            return Some((Err(err), state));
                        }
                        Err(e) => return Some((Err(e), state)),
                    };

                    let origin = self.block_origin_from_grid_id(block_grid_id);
                    for (offset_in_block, &value) in block.iter().enumerate() {
                        if value == T::default() {
                            continue;
                        }

                        let coord = self.base_coord_from_block_offset(&origin, offset_in_block);
                        if let Err(err) = validate_coord(self.shape(), &coord) {
                            return Some((Err(err), state));
                        };

                        match validate::range_contains_coord(&state.range, &coord) {
                            Ok(true) => state.buffered.push_back((coord, value)),
                            Ok(false) => {}
                            Err(e) => return Some((Err(e), state)),
                        }
                    }
                }
            });

            Ok(elements.boxed())
        })
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
                    if self.read_block(block_grid_id).await?.is_none() {
                        self.write_block(block_grid_id, self.default_block())
                            .await?;
                    }
                    self.write_value_to_block(block_grid_id, offset_in_block, value)
                        .await
                }
                Layout::Sparse { .. } => {
                    match self.plan_sparse_write(coord, block_grid_id, value).await? {
                        SparseWriteAction::Write(block_id) => {
                            self.write_value_to_block(block_id, offset_in_block, value)
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
                            self.write_value_to_block(block_id, offset_in_block, value)
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
                Ok(Some(guard.clone()))
            } else {
                Ok(None)
            }
        })
    }

    fn write_block<'a>(&'a self, block_id: u64, block: Self::Block) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
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
                blocks.create_file(block_id.to_string(), block, 0)?;
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
// TensorReadBulk / TensorWriteBulk impls
// ---------------------------------------------------------------------------
impl<FE, T> TensorReadBulk for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn read_values<'a>(&'a self, range: Range) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move {
            let mut values = Vec::new();
            for coord in validate::iter_range_coords(self.shape(), &range)? {
                values.push(self.read_value(&coord).await?);
            }
            Ok(values)
        })
    }

    fn read_all<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move { self.read_values(validate::full_range(self.shape())).await })
    }
}

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
            let coords: Vec<Vec<u64>> =
                validate::iter_range_coords(self.shape(), &range)?.collect();
            if coords.len() != values.len() {
                return Err(Error::DataMismatch(format!(
                    "expected {} values for range but got {}",
                    coords.len(),
                    values.len()
                )));
            }
            for (coord, value) in coords.into_iter().zip(values) {
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
async fn load_metadata_file<FE>(blocks: &DirLock<FE>) -> Result<(TensorSchema, StorageSchema)>
where
    FE: AsType<String> + FileLoad + Send + Sync + 'static,
{
    let file = {
        let dir = blocks.read().await;
        dir.get_file(METADATA)
            .cloned()
            .ok_or_else(|| Error::InvalidSchema("missing tensor metadata file".to_string()))?
    };
    let payload = {
        let guard = file.read::<String>().await?;
        guard.clone()
    };
    decode_schema(&payload)
}

async fn write_metadata_file<FE>(blocks: &DirLock<FE>, name: &str, payload: &str) -> Result<()>
where
    FE: AsType<String> + From<String> + FileLoad + Send + Sync + 'static,
{
    let existing = {
        let dir = blocks.read().await;
        dir.get_file(name).cloned()
    };
    if let Some(file) = existing {
        let mut guard = file.write::<String>().await?;
        *guard = payload.to_string();
    } else {
        let mut dir = blocks.write().await;
        dir.create_file(name.to_string(), payload.to_string(), payload.len())?;
    }
    Ok(())
}

fn encode_schema(schema: &TensorSchema, storage_schema: &StorageSchema) -> String {
    let dtype = schema.dtype().as_str();
    let layout = match storage_schema.layout {
        Layout::Dense => "dense".to_string(),
        Layout::Sparse { axis } => format!(
            "sparse:{}",
            axis.map(|a| a.to_string())
                .unwrap_or_else(|| "none".to_string())
        ),
    };
    let shape = schema
        .shape()
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let block_shape = storage_schema
        .block_schema
        .shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let strides = schema
        .strides()
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let s = format!(
        "version={METADATA_VERSION}\ndtype={dtype}\nlayout={layout}\nshape={shape}\nblock_shape={block_shape}\nstrides={strides}\n"
    );
    s
}

fn decode_schema(payload: &str) -> Result<(TensorSchema, StorageSchema)> {
    let mut fields = std::collections::HashMap::<String, String>::new();
    for line in payload.lines().filter(|line| !line.is_empty()) {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| Error::InvalidSchema(format!("invalid metadata line: {line}")))?;
        fields.insert(key.to_string(), value.to_string());
    }

    let version = fields
        .get("version")
        .ok_or_else(|| Error::InvalidSchema("missing metadata version".to_string()))?
        .parse::<u32>()
        .map_err(|cause| Error::InvalidSchema(format!("invalid metadata version: {cause}")))?;

    if version != METADATA_VERSION {
        return Err(Error::InvalidSchema(format!(
            "unsupported metadata version {version}; expected {METADATA_VERSION}"
        )));
    }

    let dtype_value = fields
        .get("dtype")
        .ok_or_else(|| Error::InvalidSchema("missing dtype in metadata".to_string()))?;
    let dtype = DType::try_parse(dtype_value).ok_or_else(|| {
        Error::InvalidSchema(format!("unsupported dtype in metadata: {dtype_value}"))
    })?;

    let layout = parse_layout(
        fields
            .get("layout")
            .ok_or_else(|| Error::InvalidSchema("missing layout in metadata".to_string()))?,
    )?;

    let shape = parse_usize_vec(
        fields
            .get("shape")
            .ok_or_else(|| Error::InvalidSchema("missing shape in metadata".to_string()))?,
    )?;
    let block_shape = parse_usize_vec(
        fields
            .get("block_shape")
            .ok_or_else(|| Error::InvalidSchema("missing block_shape in metadata".to_string()))?,
    )?;
    let _strides = parse_usize_vec(
        fields
            .get("strides")
            .ok_or_else(|| Error::InvalidSchema("missing strides in metadata".to_string()))?,
    )?;

    let tensor_schema = TensorSchema::new(dtype, shape.clone().into())?;
    let storage_schema = StorageSchema::from_block_shape(&shape, layout, block_shape.into())?;

    Ok((tensor_schema, storage_schema))
}

fn parse_layout(layout: &str) -> Result<Layout> {
    if layout == "dense" {
        return Ok(Layout::Dense);
    }
    if let Some(axis_hint) = layout.strip_prefix("sparse:") {
        let axis = if axis_hint == "none" {
            None
        } else {
            Some(axis_hint.parse::<usize>().map_err(|cause| {
                Error::InvalidSchema(format!("invalid sparse axis hint: {cause}"))
            })?)
        };
        return Ok(Layout::Sparse { axis });
    }
    Err(Error::InvalidSchema(format!(
        "invalid layout in metadata: {layout}"
    )))
}

fn parse_usize_vec(value: &str) -> Result<Vec<usize>> {
    if value.is_empty() {
        return Ok(vec![]);
    }
    value
        .split(',')
        .map(|dim| {
            dim.parse::<usize>()
                .map_err(|cause| Error::InvalidSchema(format!("invalid usize value: {cause}")))
        })
        .collect()
}

pub(crate) fn sparse_axis_for_layout(layout: Layout, ndim: usize) -> usize {
    let axis = match layout {
        Layout::Sparse { axis } => axis.unwrap_or(0),
        Layout::Dense => 0,
    };
    axis.min(ndim.saturating_sub(1))
}

fn validate_tensor_dtype<T: TensorElement>(dtype: DType) -> Result<()> {
    if dtype != T::DTYPE {
        return Err(Error::InvalidSchema(format!(
            "tensor dtype mismatch: schema {:?} != tensor {:?}",
            dtype,
            T::DTYPE,
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
    use ha_ndarray::shape;

    #[test]
    fn schema_metadata_roundtrip() {
        let schema = TensorSchema::new(DType::F32, shape![3, 4, 5]).expect("schema");
        let storage_schema = StorageSchema::from_block_shape(
            &[3, 4, 5],
            Layout::Sparse { axis: Some(1) },
            shape![1, 2, 5],
        )
        .expect("storage schema");

        let encoded = encode_schema(&schema, &storage_schema);
        let (decoded_schema, _decoded_storage) = decode_schema(&encoded).expect("decode");
        assert_eq!(decoded_schema, schema);
    }

    #[test]
    fn schema_metadata_roundtrip_f64() {
        let schema = TensorSchema::new(DType::F64, shape![2, 2]).expect("schema");
        let storage_schema = StorageSchema::from_block_shape(&[2, 2], Layout::Dense, shape![1, 2])
            .expect("storage schema");

        let encoded = encode_schema(&schema, &storage_schema);
        assert!(encoded.contains("dtype=f64"));

        let (decoded_schema, _) = decode_schema(&encoded).expect("decode");
        assert_eq!(decoded_schema, schema);
    }

    #[test]
    fn metadata_rejects_unknown_version() {
        let payload =
            "version=999\ndtype=f32\nlayout=dense\nshape=2,3\nblock_shape=1,3\nstrides=3,1\n";
        let err = decode_schema(payload).expect_err("should reject");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn metadata_rejects_invalid_layout() {
        let payload =
            "version=2\ndtype=f32\nlayout=weird\nshape=2,3\nblock_shape=1,3\nstrides=3,1\n";
        let err = decode_schema(payload).expect_err("should reject");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn metadata_rejects_missing_fields() {
        let payload = "version=2\ndtype=f32\nlayout=dense\nshape=2,3\n";
        let err = decode_schema(payload).expect_err("should reject");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn typed_tensor_rejects_schema_dtype_mismatch() {
        let err = validate_tensor_dtype::<f32>(DType::F64).expect_err("expected mismatch");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }
}

#[cfg(test)]
mod sparse_axis_tests {
    use super::*;

    #[test]
    fn sparse_axis_none_defaults_to_zero() {
        assert_eq!(sparse_axis_for_layout(Layout::Sparse { axis: None }, 3), 0);
    }

    #[test]
    fn sparse_axis_some_selects_given_axis() {
        assert_eq!(
            sparse_axis_for_layout(Layout::Sparse { axis: Some(1) }, 3),
            1
        );
        assert_eq!(
            sparse_axis_for_layout(Layout::Sparse { axis: Some(2) }, 3),
            2
        );
    }

    #[test]
    fn sparse_axis_none_and_some_zero_are_equivalent() {
        assert_eq!(
            sparse_axis_for_layout(Layout::Sparse { axis: None }, 3),
            sparse_axis_for_layout(Layout::Sparse { axis: Some(0) }, 3),
        );
    }

    #[test]
    fn sparse_axis_dense_is_always_zero() {
        assert_eq!(sparse_axis_for_layout(Layout::Dense, 3), 0);
    }

    #[test]
    fn sparse_axis_clamps_to_last_axis_when_out_of_range() {
        // `sparse_key` receives `coords.len()` as `ndim`, which is always the
        // tensor's own rank -- axis selection must never index past it.
        assert_eq!(
            sparse_axis_for_layout(Layout::Sparse { axis: Some(5) }, 3),
            2
        );
    }
}

#[cfg(test)]
mod sparse_lifecycle_tests {
    use std::io;
    use std::path::{Path, PathBuf};

    use b_table::Node;
    use destream::{de, en};
    use freqfs::Cache;
    use ha_ndarray::{Shape, shape};
    use safecast::as_type;

    use super::*;

    #[derive(Clone, Debug)]
    enum TestFE {
        Node(Node<u64>),
        F32(Vec<f32>),
        Text(String),
    }

    impl<'en> en::ToStream<'en> for TestFE {
        fn to_stream<E: en::Encoder<'en>>(
            &'en self,
            encoder: E,
        ) -> std::result::Result<E::Ok, E::Error> {
            match self {
                Self::Node(node) => node.to_stream(encoder),
                Self::F32(values) => values.to_stream(encoder),
                Self::Text(text) => text.to_stream(encoder),
            }
        }
    }

    // Only ever read via a concrete `AsType` target (mirrors `tests/common.rs::FsEntry`).
    impl de::FromStream for TestFE {
        type Context = ();

        async fn from_stream<D: de::Decoder>(
            _: (),
            _decoder: &mut D,
        ) -> std::result::Result<Self, D::Error> {
            Err(de::Error::custom(
                "TestFE does not support generic decoding; read via a concrete AsType target",
            ))
        }
    }

    as_type!(TestFE, Node, Node<u64>);
    as_type!(TestFE, F32, Vec<f32>);
    as_type!(TestFE, Text, String);

    fn unique_tmp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        path.push(format!("fensor_sparse_lifecycle_{name}_{unique}"));
        path
    }

    fn open_dir(root: &Path) -> io::Result<DirLock<TestFE>> {
        let cache = Cache::<TestFE>::new(1_000_000, None);
        cache.load(root.to_path_buf())
    }

    async fn new_dir(name: &str) -> (PathBuf, DirLock<TestFE>) {
        let root = unique_tmp_dir(name);
        tokio::fs::create_dir(&root).await.expect("create tmp dir");
        let dir = open_dir(&root).expect("load tmp dir");
        (root, dir)
    }

    async fn cleanup(root: &Path) {
        let _ = tokio::fs::remove_dir_all(root).await;
    }

    async fn create_sparse(
        name: &str,
        shape: Shape,
        max_capacity: usize,
        axis: Option<usize>,
    ) -> (PathBuf, Tensor<TestFE, f32>) {
        let (root, dir) = new_dir(name).await;
        let schema = TensorSchema::new(DType::F32, shape).expect("schema");
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
    async fn compact_sparse_removes_all_zero_rows() {
        let (root, tensor) = create_sparse("compact", shape![2, 3, 4], 4, Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("nz");
        tensor.write_value(&[0, 1, 2], 0.0).await.expect("zero");

        tensor
            .compact_sparse()
            .await
            .expect("compact must be supported");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "compaction must drop all-zero rows"
        );

        cleanup(&root).await;
    }

    #[tokio::test]
    async fn compact_sparse_preserves_nonzero_rows() {
        let (root, tensor) = create_sparse("compact_preserve", shape![2, 3, 4], 4, Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("write nz");

        tensor.compact_sparse().await.expect("compact");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_some(),
            "compact must not remove rows with nonzero values"
        );
        assert_eq!(tensor.read_value(&[0, 1, 2]).await.expect("read"), 5.0);

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
    async fn compact_sparse_idempotent() {
        let (root, tensor) = create_sparse("compact_idem", shape![2, 3, 4], 4, Some(1)).await;

        tensor.write_value(&[0, 1, 2], 5.0).await.expect("nz");
        tensor.write_value(&[1, 2, 3], 3.0).await.expect("nz2");
        tensor.write_value(&[0, 1, 2], 0.0).await.expect("zero");

        tensor.compact_sparse().await.expect("first compact");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "zero row removed after first compact"
        );
        assert!(
            block_id_for_coord(&tensor, &[1, 2, 3]).await.is_some(),
            "nonzero row preserved after first compact"
        );

        tensor.compact_sparse().await.expect("second compact");

        assert!(
            block_id_for_coord(&tensor, &[0, 1, 2]).await.is_none(),
            "still absent after second compact"
        );
        assert!(
            block_id_for_coord(&tensor, &[1, 2, 3]).await.is_some(),
            "nonzero row still present after second compact"
        );

        cleanup(&root).await;
    }
}
