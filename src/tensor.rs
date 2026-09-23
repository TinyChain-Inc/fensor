use std::io::{Error as IoError, ErrorKind};
use std::sync::Arc;

use b_table::{TableLock, collate::Collator};
use destream::{de, en};
use freqfs::{DirLock, FileLoad};
use futures::TryStreamExt as _;
use ha_ndarray::Range;
#[cfg(test)]
use number_general::FloatType;
use number_general::NumberType;
use safecast::AsType;

use crate::error::{Error, Result};
use crate::schema::{
    BlockPosition, Layout, MAX_BLOCK_CAPACITY, SparseIndexSchema, SparseTableSchema, StorageSchema,
    TensorSchema,
};
use crate::traits::{
    BoxFuture, TensorArray, TensorBlockStore, TensorGeometry, TensorRead, TensorSparseIndex,
    TensorWrite, TensorWriteBulk,
};
use crate::view::TensorView;
use crate::{TensorMetadata, validate};

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
    /// partial destination storage to the caller.
    pub async fn copy_from<R>(dir: DirLock<FE>, source: &R, max_capacity: usize) -> Result<Self>
    where
        R: TensorRead<DType = T> + ?Sized,
    {
        let mut blocks = source.read_blocks()?;
        let schema = TensorSchema::new(
            <T as number_general::DType>::dtype(),
            source.shape().to_vec().into(),
        )?;
        let layout = match source.layout() {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: None },
        };
        let output = Self::create(dir, schema, layout, max_capacity).await?;
        let mut coords = crate::schema::row_major_coords(source.shape())?;
        while let Some(values) = blocks.try_next().await? {
            for value in values {
                let coord = coords.next().ok_or_else(|| {
                    Error::InvalidLayout("reader returned more values than its shape".into())
                })?;
                if matches!(layout, Layout::Sparse { .. }) && value == T::default() {
                    continue;
                }
                output.write_value(&coord, value).await?;
            }
        }
        if coords.next().is_some() {
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

    async fn write_value_to_block(
        &self,
        block_id: u64,
        offset_in_block: usize,
        value: T,
    ) -> Result<()> {
        let file = {
            let blocks = self.storage.blocks().read().await;
            blocks
                .get_file(&block_id.to_string())
                .cloned()
                .ok_or_else(|| Error::from(IoError::new(ErrorKind::NotFound, "Missing block")))?
        };
        let mut block = file.write::<Vec<T>>().await?;
        if block.len() != self.block_len() {
            return Err(Error::InvalidLayout("invalid stored block length".into()));
        }
        validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
        block[offset_in_block] = value;
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
            validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
            Ok(block[offset_in_block])
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
                    if !self.block_exists(block_grid_id).await {
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
            if coords.len() != values.len() {
                return Err(Error::DataMismatch(format!(
                    "expected {} values for range but got {}",
                    coords.len(),
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

pub(crate) fn sparse_axis_for_layout(layout: Layout, ndim: usize) -> usize {
    let axis = match layout {
        Layout::Sparse { axis } => axis.unwrap_or(0),
        Layout::Dense => 0,
    };
    axis.min(ndim.saturating_sub(1))
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
        MetadataF32(crate::TensorMetadata<f32>),
    }
    impl<'en> en::ToStream<'en> for TestFE {
        fn to_stream<E: en::Encoder<'en>>(
            &'en self,
            encoder: E,
        ) -> std::result::Result<E::Ok, E::Error> {
            match self {
                Self::Node(value) => en::IntoStream::into_stream((0u8, value), encoder),
                Self::F32(value) => en::IntoStream::into_stream((1u8, value), encoder),
                Self::MetadataF32(value) => en::IntoStream::into_stream((2u8, value), encoder),
            }
        }
    }
    struct TestFEVisitor;
    impl de::Visitor for TestFEVisitor {
        type Value = TestFE;
        fn expecting() -> &'static str {
            "a typed filesystem entry"
        }
        async fn visit_seq<A: de::SeqAccess>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let entry = match seq.expect_next::<u8>(()).await? {
                0 => TestFE::Node(seq.expect_next(()).await?),
                1 => TestFE::F32(seq.expect_next(()).await?),
                2 => TestFE::MetadataF32(seq.expect_next(()).await?),
                tag => return Err(de::Error::custom(format!("unknown entry tag {tag}"))),
            };
            if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                return Err(de::Error::custom("unexpected entry field"));
            }
            Ok(entry)
        }
    }
    impl de::FromStream for TestFE {
        type Context = ();
        async fn from_stream<D: de::Decoder>(
            _: (),
            decoder: &mut D,
        ) -> std::result::Result<Self, D::Error> {
            decoder.decode_seq(TestFEVisitor).await
        }
    }

    impl freqfs::FileLoad for TestFE {
        async fn load(
            _: &std::path::Path,
            file: tokio::fs::File,
            _: std::fs::Metadata,
        ) -> std::io::Result<Self> {
            tbon::de::read_from((), file)
                .await
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        }
    }
    impl freqfs::FileSave for TestFE {
        async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
            use futures::TryStreamExt;
            use tokio::io::AsyncWriteExt;
            let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
            let mut size = 0;
            while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
                file.write_all(&chunk).await?;
                size += chunk.len() as u64;
            }
            Ok(size)
        }
    }
    as_type!(TestFE, Node, Node<u64>);
    as_type!(TestFE, F32, Vec<f32>);
    as_type!(TestFE, MetadataF32, crate::TensorMetadata<f32>);

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
        let cache = Cache::<TestFE>::new(1_000_000, None, 0, std::time::Duration::from_secs(1));
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
        let error = crate::expression::evaluate_batch(&tensor, &coords)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidLayout(_)));
        assert!(error.to_string().contains("coordinate batch"));
        cleanup(&root).await;
    }
}
