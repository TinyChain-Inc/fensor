use std::io;
use std::sync::Arc;

use b_table::{TableLock, collate::Collator};
use destream::{de, en};
use freqfs::{DirLock, FileLoad};
use ha_ndarray::{Axes, Range, Shape};
use safecast::AsType;

mod error;
mod schema;
mod stream;
mod traits;
mod validate;
mod view;
mod wire_tags;

pub use error::{Error, Result};
pub use schema::{
    AxisContribSchema, DType, Layout, SparseIndexSchema, SparseTableSchema, TensorSchema,
    TensorShape, ViewSchema, contiguous_strides,
};
pub use traits::{
    BoxFuture, SparseZeroPolicy, TensorArray, TensorBlockStore, TensorMatMul, TensorMath,
    TensorMathScalar, TensorRead, TensorReadBulk, TensorReduce, TensorReduceAll,
    TensorReduceBoolean, TensorSparseIndex, TensorSparseLifecycle, TensorTransform, TensorUnary,
    TensorViewSemantics, TensorWrite, TensorWriteBulk,
};

use view::{TensorView, default_permutation};

const BLOCKS: &str = "blocks";
const INDEX: &str = "index";
const METADATA: &str = "metadata";
const METADATA_VERSION: u32 = 1;

pub trait TensorElement:
    Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + de::FromStream<Context = ()>
    + for<'en> en::ToStream<'en>
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

struct TensorStorage<FE> {
    blocks: DirLock<FE>,
    index: Option<TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>>,
    schema: TensorSchema,
}

#[derive(Clone)]
pub struct Tensor<FE, T = f32> {
    storage: Arc<TensorStorage<FE>>,
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
    pub async fn create(dir: DirLock<FE>, schema: TensorSchema) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        validate_tensor_dtype::<T>(&schema)?;

        let mut dir_guard = dir.try_write()?;

        let blocks_dir = dir_guard.create_dir(BLOCKS.to_string())?;
        let index = create_sparse_index(&schema, &mut dir_guard)?;

        let tensor = Self::new_storage(blocks_dir, index, schema);
        tensor.persist_metadata().await?;
        Ok(tensor)
    }

    pub async fn load(dir: DirLock<FE>) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let mut dir_guard = dir.try_write()?;

        let blocks_dir = dir_guard.get_or_create_dir(BLOCKS.to_string())?;
        let schema = Self::load_metadata(&blocks_dir).await?;
        validate_tensor_dtype::<T>(&schema)?;
        let index = load_sparse_index(&schema, &mut dir_guard)?;

        Ok(Self::new_storage(blocks_dir, index, schema))
    }

    pub async fn load_with_schema(dir: DirLock<FE>, expected: &TensorSchema) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        validate_tensor_dtype::<T>(expected)?;

        let tensor = Self::load(dir).await?;

        if &tensor.schema != expected {
            return Err(Error::InvalidSchema(format!(
                "persisted metadata mismatch: expected {:?}, found {:?}",
                expected, tensor.schema
            )));
        }

        Ok(tensor)
    }

    fn new_storage(
        blocks: DirLock<FE>,
        index: Option<TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>>,
        schema: TensorSchema,
    ) -> Self {
        let view = TensorView::identity(&schema);
        let storage = Arc::new(TensorStorage {
            blocks,
            index,
            schema: schema.clone(),
        });

        Self {
            storage,
            schema,
            view,
            _dtype: std::marker::PhantomData,
        }
    }

    pub fn view_schema(&self) -> Result<ViewSchema> {
        self.view.to_schema()
    }

    pub fn with_view_schema(mut self, view_schema: &ViewSchema) -> Result<Self> {
        let view = TensorView::from_schema(view_schema)?;
        if view.rank() != self.schema.rank() {
            return Err(Error::InvalidSchema(
                "view rank must match tensor rank".to_string(),
            ));
        }

        self.view = view;
        Ok(self)
    }

    pub(crate) fn block_len(&self) -> usize {
        self.storage.schema.block_len().max(1)
    }

    pub(crate) fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        self.schema.validate_coord(coord)?;

        let k = self.view.flat_offset(coord)?;
        if k < 0 {
            return Err(Error::InvalidCoord("negative linear offset".to_string()));
        }
        let k = k as u64;

        let base_coord: Vec<u64> = self.storage.schema.strides().iter()
            .zip(self.storage.schema.shape().iter())
            .map(|(stride, dim)| (k / *stride as u64) % *dim as u64)
            .collect();

        self.storage.schema.validate_coord(&base_coord)?;
        Ok(base_coord)
    }

    pub(crate) fn block_position_from_base_coord(&self, base_coord: &[u64]) -> (u64, usize) {
        let offset: u64 = base_coord
            .iter()
            .zip(self.storage.schema.strides().iter())
            .map(|(coord, stride)| *coord * (*stride as u64))
            .sum();

        let block_len = self.block_len() as u64;
        let block_offset = offset / block_len;
        let offset_in_block = (offset % block_len) as usize;

        (block_offset, offset_in_block)
    }

    pub(crate) fn sparse_key(&self, coords: &[u64], block_offset: u64) -> Vec<u64> {
        let sparse_axis = match self.storage.schema.layout() {
            Layout::Sparse { axis } => axis.unwrap_or(0),
            Layout::Dense => 0,
        };

        let axis = sparse_axis.min(coords.len().saturating_sub(1));
        vec![coords[axis], block_offset]
    }

    pub(crate) fn has_sparse_index(&self) -> bool {
        self.storage.index.is_some()
    }

    async fn lookup_sparse_block_for_coord(
        &self,
        base_coord: &[u64],
        block_offset: u64,
    ) -> Result<Option<u64>> {
        let key = self.sparse_key(base_coord, block_offset);
        self.lookup_block_id(&key).await
    }

    fn default_block(&self) -> Vec<T> {
        vec![T::default(); self.block_len()]
    }

    async fn resolve_sparse_block_for_write(
        &self,
        base_coord: &[u64],
        block_offset: u64,
        value: T,
    ) -> Result<Option<u64>> {
        let key = self.sparse_key(base_coord, block_offset);

        if let Some(block_id) = self
            .lookup_sparse_block_for_coord(base_coord, block_offset)
            .await?
        {
            return Ok(Some(block_id));
        }

        if value == T::default() {
            return Ok(None);
        }

        let block_id: u64 = rand::random();
        self.write_block(block_id, self.default_block()).await?;
        self.upsert_block_id(key, block_id).await?;

        Ok(Some(block_id))
    }

    async fn write_value_to_block(
        &self,
        block_id: u64,
        offset_in_block: usize,
        value: T,
        create_if_missing: bool,
    ) -> Result<()> {
        let mut block = match self.read_block(block_id).await? {
            Some(block) => block,
            None if create_if_missing => self.default_block(),
            None => {
                return Err(Error::SparseIndex(
                    "sparse index points to missing block".to_string(),
                ));
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
        let payload = encode_schema(&self.storage.schema);

        // Validate metadata payload before writing it.
        let _ = decode_schema(&payload)?;

        self.write_metadata_file(METADATA, &payload).await?;

        Ok(())
    }

    async fn load_metadata(blocks: &DirLock<FE>) -> Result<TensorSchema>
    where
        FE: AsType<String> + From<String>,
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

    async fn write_metadata_file(&self, name: &str, payload: &str) -> Result<()>
    where
        FE: AsType<String> + From<String>,
    {
        let existing = {
            let dir = self.storage.blocks.read().await;
            dir.get_file(name).cloned()
        };

        if let Some(file) = existing {
            let mut guard = file.write::<String>().await?;
            *guard = payload.to_string();
        } else {
            let mut dir = self.storage.blocks.write().await;
            dir.create_file(name.to_string(), payload.to_string(), payload.len())?;
        }

        Ok(())
    }
}

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

    fn schema_dtype(&self) -> DType {
        self.schema.dtype()
    }
}

impl<FE, T> TensorRead for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let base_coord = self.resolve_base_coord(coord)?;
            let (block_offset, offset_in_block) = self.block_position_from_base_coord(&base_coord);

            let block_id = if self.has_sparse_index() {
                self.lookup_sparse_block_for_coord(&base_coord, block_offset)
                    .await?
            } else {
                Some(block_offset)
            };

            if let Some(block_id) = block_id {
                if let Some(block) = self.read_block(block_id).await? {
                    validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
                    Ok(block[offset_in_block])
                } else {
                    Ok(T::default())
                }
            } else {
                Ok(T::default())
            }
        })
    }
}

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
            let (block_offset, offset_in_block) = self.block_position_from_base_coord(&base_coord);

            if self.has_sparse_index() {
                if let Some(block_id) = self
                    .resolve_sparse_block_for_write(&base_coord, block_offset, value)
                    .await?
                {
                    self.write_value_to_block(block_id, offset_in_block, value, false)
                        .await
                } else {
                    Ok(())
                }
            } else {
                self.write_value_to_block(block_offset, offset_in_block, value, true)
                    .await
            }
        })
    }
}

impl<FE, T> TensorTransform for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn reshape(mut self, shape: Shape) -> Result<Self> {
        let old_shape = self.schema.shape();
        let old_size: usize = old_shape.iter().product();
        let new_size: usize = shape.iter().product();

        if old_size != new_size {
            return Err(Error::InvalidLayout(
                "reshape requires an equal number of elements".to_string(),
            ));
        }

        self.view = self.view.reshape(self.schema.shape(), &shape)?;
        self.schema.set_shape(shape)?;
        self.schema.set_strides(contiguous_strides(self.schema.shape()))?;

        Ok(self)
    }

    fn slice(mut self, range: Range) -> Result<Self> {
        let current_shape = self.schema.shape();
        let (view, shape, strides) = self.view.slice(current_shape, &range)?;
        self.view = view;
        self.schema.set_shape(shape)?;
        self.schema.set_strides(strides)?;

        Ok(self)
    }

    fn transpose(mut self, permutation: Option<Axes>) -> Result<Self> {
        let current_shape = self.schema.shape();
        let current_strides = self.schema.strides().clone();

        let permutation = default_permutation(current_shape.len(), permutation)?;
        self.view = self.view.transpose(&permutation)?;

        let mut shape = Shape::with_capacity(current_shape.len());
        let mut strides = Vec::with_capacity(current_shape.len());

        for axis in permutation {
            shape.push(current_shape[axis]);
            strides.push(current_strides[axis]);
        }

        self.schema.set_shape(shape)?;
        self.schema.set_strides(strides.into())?;

        Ok(self)
    }
}

impl<FE, T> TensorBlockStore for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Block = Vec<T>;

    fn read_block<'a>(&'a self, block_id: u64) -> BoxFuture<'a, Result<Option<Self::Block>>> {
        Box::pin(async move {
            let blocks = self.storage.blocks.read().await;
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
                let blocks = self.storage.blocks.read().await;
                blocks.get_file(&block_id.to_string()).cloned()
            };

            if let Some(file) = file {
                let mut guard = file.write::<Vec<T>>().await?;
                *guard = block;

                Ok(())
            } else {
                let mut blocks = self.storage.blocks.write().await;
                blocks.create_file(block_id.to_string(), block, 0)?;

                Ok(())
            }
        })
    }
}

impl<FE, T> TensorSparseIndex for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn lookup_block_id<'a>(&'a self, key: &'a [u64]) -> BoxFuture<'a, Result<Option<u64>>> {
        Box::pin(async move {
            if let Some(index) = &self.storage.index {
                let index_lock = index.read().await;
                if let Some(row) = index_lock.get_row(key).await? {
                    Ok(row.get(2).copied())
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        })
    }

    fn upsert_block_id<'a>(&'a self, key: Vec<u64>, block_id: u64) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if let Some(index) = &self.storage.index {
                let mut index_lock = index.write().await;
                index_lock
                    .upsert(key, vec![block_id])
                    .await
                    .map(|_| ())
                    .map_err(Error::from)
            } else {
                Err(Error::Unsupported(
                    "sparse index is not available for dense layout".to_string(),
                ))
            }
        })
    }
}

// Trait-surface wiring only: every method inherits the default `Unsupported`
// behavior from `traits.rs`. These exist so the test matrix can call into the
// trait surface and observe failures in the right places. They are NOT a
// feature implementation — see Requirements_1.md.
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

impl<FE, T> TensorViewSemantics for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
}

impl<FE, T> TensorSparseLifecycle for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
}

fn default_block_shape(shape: &Shape) -> Shape {
    if shape.is_empty() {
        return Shape::new();
    }

    let mut block_shape = Shape::with_capacity(shape.len());
    for (i, dim) in shape.iter().enumerate() {
        if i + 1 == shape.len() {
            block_shape.push((*dim).min(4096));
        } else {
            block_shape.push(1);
        }
    }

    block_shape
}

fn encode_schema(schema: &TensorSchema) -> String {
    let dtype = schema.dtype().as_str();

    let layout = match schema.layout() {
        Layout::Dense => "dense".to_string(),
        Layout::Sparse { axis } => format!(
            "sparse:{}",
            axis.map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        ),
    };

    let shape = schema
        .shape()
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let block_shape = schema
        .block_shape()
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

    format!(
        "version={METADATA_VERSION}\ndtype={dtype}\nlayout={layout}\nshape={shape}\nblock_shape={block_shape}\nstrides={strides}\n"
    )
}

fn decode_schema(payload: &str) -> Result<TensorSchema> {
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
    let strides = parse_usize_vec(
        fields
            .get("strides")
            .ok_or_else(|| Error::InvalidSchema("missing strides in metadata".to_string()))?,
    )?;

    TensorSchema::new(
        dtype,
        shape.into(),
        layout,
        block_shape.into(),
        strides.into(),
    )
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

fn validate_tensor_dtype<T: TensorElement>(schema: &TensorSchema) -> Result<()> {
    if schema.dtype() != T::DTYPE {
        return Err(Error::InvalidSchema(format!(
            "tensor dtype mismatch: schema {:?} != tensor {:?}",
            schema.dtype(),
            T::DTYPE,
        )));
    }

    Ok(())
}

type SparseIndex<FE> = TableLock<SparseTableSchema, SparseIndexSchema, Collator<u64>, FE>;

fn create_sparse_index<FE>(
    schema: &TensorSchema,
    dir_guard: &mut freqfs::DirWriteGuard<'_, FE>,
) -> io::Result<Option<SparseIndex<FE>>>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + Send + Sync + 'static,
{
    if matches!(schema.layout(), Layout::Sparse { .. }) {
        let index_dir = dir_guard.create_dir(INDEX.to_string())?;

        let table_schema = SparseTableSchema::default();
        let collator = Collator::default();

        TableLock::create(table_schema, collator, index_dir).map(Some)
    } else {
        Ok(None)
    }
}

fn load_sparse_index<FE>(
    schema: &TensorSchema,
    dir_guard: &mut freqfs::DirWriteGuard<'_, FE>,
) -> io::Result<Option<SparseIndex<FE>>>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + Send + Sync + 'static,
{
    if matches!(schema.layout(), Layout::Sparse { .. })
        && let Some(index_dir) = dir_guard.get_dir(INDEX).cloned()
    {
        let table_schema = SparseTableSchema::default();
        let collator = Collator::default();

        TableLock::load(table_schema, collator, index_dir).map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use ha_ndarray::shape;

    #[test]
    fn schema_metadata_roundtrip() {
        let schema = TensorSchema::new(
            DType::F32,
            shape![3, 4, 5],
            Layout::Sparse { axis: Some(1) },
            shape![1, 2, 5],
            vec![20, 5, 1].into(),
        )
        .expect("schema");

        let encoded = encode_schema(&schema);
        let decoded = decode_schema(&encoded).expect("decode");
        assert_eq!(decoded, schema);
    }

    #[test]
    fn schema_metadata_roundtrip_f64() {
        let schema = TensorSchema::new(
            DType::F64,
            shape![2, 2],
            Layout::Dense,
            shape![1, 2],
            vec![2, 1].into(),
        )
        .expect("schema");

        let encoded = encode_schema(&schema);
        assert!(encoded.contains("dtype=f64"));

        let decoded = decode_schema(&encoded).expect("decode");
        assert_eq!(decoded, schema);
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
            "version=1\ndtype=f32\nlayout=weird\nshape=2,3\nblock_shape=1,3\nstrides=3,1\n";
        let err = decode_schema(payload).expect_err("should reject");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn metadata_rejects_missing_fields() {
        let payload = "version=1\ndtype=f32\nlayout=dense\nshape=2,3\n";
        let err = decode_schema(payload).expect_err("should reject");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }

    #[test]
    fn typed_tensor_rejects_schema_dtype_mismatch() {
        let schema = TensorSchema::new(
            DType::F64,
            shape![2, 2],
            Layout::Dense,
            shape![1, 2],
            vec![2, 1].into(),
        )
        .expect("schema");

        let err = validate_tensor_dtype::<f32>(&schema).expect_err("expected mismatch");
        assert!(matches!(err, Error::InvalidSchema(_)));
    }
}
