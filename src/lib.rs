use std::io::{Error as IoError, ErrorKind};
use std::sync::Arc;

use b_table::{TableLock, collate::Collator};
use destream::{de, en};
use freqfs::{DirLock, FileLoad};
use futures::StreamExt as _;
use ha_ndarray::{Axes, Range, Shape};
use safecast::AsType;

mod error;
mod schema;
mod stream;
pub mod tensor;
mod traits;
mod validate;
mod view;
mod wire_tags;

pub use error::{Error, Result};
pub use schema::{
    DType, Layout, SparseIndexSchema, SparseTableSchema, TensorShape, contiguous_strides,
};
pub use stream::{TensorViewDecoder, TensorViewEncoder};
pub use tensor::TensorSchema;
pub use traits::{
    BoxFuture, TensorArray, TensorBlockStore, TensorMatMul, TensorMath, TensorMathScalar,
    TensorRead, TensorReadBulk, TensorReduce, TensorReduceAll, TensorReduceBoolean,
    TensorSparseIndex, TensorTransform, TensorUnary, TensorViewSemantics, TensorWrite,
    TensorWriteBulk,
};

use view::{TensorView, default_permutation};

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
    storage_schema: tensor::StorageSchema,
}

type SparseIndex<FE> = TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>;

struct SparseStorage<FE> {
    blocks: DirLock<FE>,
    index: SparseIndex<FE>,
    storage_schema: tensor::StorageSchema,
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

    pub(crate) fn storage_schema(&self) -> &tensor::StorageSchema {
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

// ---------------------------------------------------------------------------
// Sparse
// ---------------------------------------------------------------------------
enum SparseWriteAction {
    Write(u64),
    DeleteRow(u64),
    NoOp,
    CreateBlockAndWrite(u64),
}

// ---------------------------------------------------------------------------
// Tensor enum
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Tensor<FE, T> {
    storage: Arc<Storage<FE>>,
    schema: TensorSchema,
    view: TensorView,
    _dtype: std::marker::PhantomData<T>,
}

pub type TensorF32<FE> = Tensor<FE, f32>;
pub type TensorF64<FE> = Tensor<FE, f64>;

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
        if max_capacity == 0 || max_capacity > tensor::MAX_BLOCK_CAPACITY {
            return Err(Error::InvalidSchema(format!(
                "max block capacity must be non-zero and at most {}, got {max_capacity}",
                tensor::MAX_BLOCK_CAPACITY
            )));
        }

        validate_tensor_dtype::<T>(schema.dtype())?;

        let storage_schema =
            tensor::StorageSchema::new(schema.shape().as_slice(), layout, max_capacity)?;

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

    // pub async fn load(dir: DirLock<FE>) -> Result<Self>
    // where
    //     FE: AsType<String> + From<String>,
    // {
    //     let mut dir_guard = dir.try_write()?;
    //     let blocks_dir = dir_guard.get_or_create_dir(BLOCKS.to_string())?;
    //     let schema = load_metadata_file(&blocks_dir).await?;
    //     validate_tensor_dtype::<T>(&schema)?;
    //     let index: Option<SparseIndex<FE>> = if let Layout::Sparse { .. } = schema.layout() {
    //         let index_dir = dir_guard.get_dir(INDEX).cloned().ok_or_else(|| {
    //             Error::InvalidSchema("sparse tensor missing index directory".to_string())
    //         })?;
    //         Some(TableLock::load(
    //             SparseTableSchema::default(),
    //             Collator::default(),
    //             index_dir,
    //         )?)
    //     } else {
    //         None
    //     };

    //     Ok(Self::new_storage(blocks_dir, index, schema))
    // }

    /// Build a lazily-streamed encoder for this tensor's current view
    /// (identity or transformed, dense or sparse): schema followed by a
    /// nested sequence of non-default `(coord, value)` pairs and a trailing
    /// verification record. No full in-memory buffering -- each value is
    /// read from storage only as the returned value is actually driven by
    /// a destream encoder (e.g. `tbon::en::encode(tensor.view_encoder())`).
    pub fn view_encoder(&self) -> stream::TensorViewEncoder<'_, FE, T> {
        stream::TensorViewEncoder::new(self)
    }

    pub(crate) fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        validate::validate_coord(self.view.shape(), coord)?;
        let k = self.view.flat_offset(coord)?;
        if k < 0 {
            return Err(Error::InvalidCoord("negative linear offset".to_string()));
        }
        let k = k as u64;
        let base_coord: Vec<u64> = self
            .schema
            .strides()
            .iter()
            .zip(self.schema.shape().iter())
            .map(|(stride, dim)| (k / *stride as u64) % *dim as u64)
            .collect();
        validate::validate_coord(self.schema.shape(), &base_coord)?;
        Ok(base_coord)
    }

    pub(crate) fn block_position_from_base_coord(
        &self,
        base_coord: &[u64],
    ) -> tensor::BlockPosition {
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
        tensor::BlockPosition {
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
            self.write_block(block_id, self.default_block()).await?;
        }
        Ok(())
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
        let sparse_axis = match self.layout() {
            Layout::Sparse { axis } => axis.unwrap_or(0),
            Layout::Dense => 0,
        };
        let axis = sparse_axis.min(coords.len().saturating_sub(1));
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
        storage_schema: tensor::StorageSchema,
    ) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let view = TensorView::identity(schema.shape().clone(), schema.strides().clone());

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
            view,
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
// TensorArray impls
// ---------------------------------------------------------------------------
impl<FE, T> TensorArray for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type DType = T;

    fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    fn dtype(&self) -> Self::DType {
        T::default()
    }

    fn layout(&self) -> Layout {
        self.storage.storage_schema().layout
    }

    fn shape(&self) -> &[usize] {
        self.view.shape()
    }

    fn strides(&self) -> &[usize] {
        self.view.strides()
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
            let base_coord = self.resolve_base_coord(coord)?;
            let tensor::BlockPosition {
                block_id: block_grid_id,
                offset_in_block,
            } = self.block_position_from_base_coord(&base_coord);

            let block_id = match self.layout() {
                Layout::Dense => Some(block_grid_id),
                Layout::Sparse { .. } => {
                    self.lookup_sparse_block_for_coord(&base_coord, block_grid_id)
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
            let base_coord = self.resolve_base_coord(coord)?;
            let tensor::BlockPosition {
                block_id: block_grid_id,
                offset_in_block,
            } = self.block_position_from_base_coord(&base_coord);

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
                    match self
                        .plan_sparse_write(&base_coord, block_grid_id, value)
                        .await?
                    {
                        SparseWriteAction::Write(block_id) => {
                            self.write_value_to_block(block_id, offset_in_block, value)
                                .await
                        }
                        SparseWriteAction::DeleteRow(block_id) => {
                            let key = self.sparse_key(&base_coord, block_grid_id);
                            self.delete_row(key).await?;
                            if self.is_empty_block(block_id).await? {
                                self.delete_block(block_id).await;
                            }

                            Ok(())
                        }
                        SparseWriteAction::CreateBlockAndWrite(block_id) => {
                            self.write_block(block_id, self.default_block()).await?;
                            let key = self.sparse_key(&base_coord, block_grid_id);
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
// TensorTransform impls
// ---------------------------------------------------------------------------

impl<FE, T> TensorTransform for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn reshape(mut self, shape: Shape) -> Result<Self> {
        let old_size: usize = self.view.shape().iter().product();
        let new_size: usize = shape.iter().product();
        if old_size != new_size {
            return Err(Error::InvalidLayout(
                "reshape requires an equal number of elements".to_string(),
            ));
        }
        self.view = self.view.reshape(&shape)?;
        Ok(self)
    }

    fn slice(mut self, range: Range) -> Result<Self> {
        self.view = self.view.slice(&range)?;
        Ok(self)
    }

    fn transpose(mut self, permutation: Option<Axes>) -> Result<Self> {
        let permutation = default_permutation(self.view.shape().len(), permutation)?;
        self.view = self.view.transpose(&permutation)?;
        Ok(self)
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
// TensorViewSemantics impls
// ---------------------------------------------------------------------------
impl<FE, T> TensorViewSemantics for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn is_base_tensor(&self) -> bool {
        self.view.is_identity(self.schema.shape())
    }

    fn supports_write_through(&self) -> bool {
        !self.view.has_gather_axes()
    }
}

// ---------------------------------------------------------------------------
// Unsupported bulk traits (trait-surface wiring only)
// ---------------------------------------------------------------------------
impl<FE, T> TensorReadBulk for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
}

impl<FE, T> TensorWriteBulk for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

async fn load_metadata_file<FE>(
    blocks: &DirLock<FE>,
) -> Result<(TensorSchema, tensor::StorageSchema)>
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

fn encode_schema(schema: &TensorSchema, storage_schema: &tensor::StorageSchema) -> String {
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

fn decode_schema(payload: &str) -> Result<(TensorSchema, tensor::StorageSchema)> {
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
    let storage_schema =
        tensor::StorageSchema::from_block_shape(&shape, layout, block_shape.into())?;

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
        let storage_schema = tensor::StorageSchema::from_block_shape(
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
        let storage_schema =
            tensor::StorageSchema::from_block_shape(&[2, 2], Layout::Dense, shape![1, 2])
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
