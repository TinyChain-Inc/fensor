use std::sync::Arc;

use b_table::TableLock;
use freqfs::{DirLock, FileLoad};
use ha_ndarray::{Axes, Range, Shape};
use safecast::AsType;

mod dense;
mod error;
mod io;
mod schema;
mod sparse;
mod traits;
mod validate;
mod view;
mod view_ops;

pub use error::{Error, Result};
pub use schema::{
    DType, Layout, SparseIndexSchema, SparseTableSchema, TensorSchema, contiguous_strides,
};
pub use traits::{
    BoxFuture, Tensor, TensorBlockStore, TensorRead, TensorSparseIndex, TensorTransform,
    TensorWrite,
};

use view::{TensorView, default_permutation};

const BLOCKS: &str = "blocks";
const INDEX: &str = "index";
const METADATA: &str = "metadata";
const METADATA_VERSION: u32 = 1;

struct TensorStorage<FE> {
    blocks: DirLock<FE>,
    index: Option<
        TableLock<SparseTableSchema, SparseIndexSchema, b_table::collate::Collator<u64>, FE>,
    >,
    schema: TensorSchema,
}

#[derive(Clone)]
pub struct Tensor<FE> {
    storage: Arc<TensorStorage<FE>>,
    schema: TensorSchema,
    view: TensorView,
}

impl<FE> Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    pub async fn create(dir: DirLock<FE>, schema: TensorSchema) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let mut dir_guard = dir.try_write()?;

        let blocks_dir = dir_guard.create_dir(BLOCKS.to_string())?;
        let index = create_or_load_index(&schema, &mut dir_guard, true)?;

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
        let index = create_or_load_index(&schema, &mut dir_guard, false)?;

        Ok(Self::new_storage(blocks_dir, index, schema))
    }

    pub async fn load_with_schema(dir: DirLock<FE>, expected: &TensorSchema) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
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
        index: Option<
            TableLock<SparseTableSchema, SparseIndexSchema, b_table::collate::Collator<u64>, FE>,
        >,
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
        }
    }

    pub async fn create_legacy(
        dir: DirLock<FE>,
        shape: Shape,
        axes: Axes,
        sparse: bool,
    ) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let block_shape = default_block_shape(&shape);
        let layout = if sparse {
            Layout::Sparse { axis: None }
        } else {
            Layout::Dense
        };

        let mut strides = contiguous_strides(&shape);
        if axes.len() == shape.len() {
            let mut permuted = strides.clone();
            for (i, axis) in axes.iter().enumerate() {
                if *axis < strides.len() {
                    permuted[i] = strides[*axis];
                }
            }
            strides = permuted;
        }

        let schema = TensorSchema::new(DType::F32, shape, layout, block_shape, strides)?;
        Self::create(dir, schema).await
    }

    pub async fn load_legacy(
        dir: DirLock<FE>,
        shape: Shape,
        axes: Axes,
        sparse: bool,
    ) -> Result<Self>
    where
        FE: AsType<String> + From<String>,
    {
        let block_shape = default_block_shape(&shape);
        let layout = if sparse {
            Layout::Sparse { axis: None }
        } else {
            Layout::Dense
        };

        let mut strides = contiguous_strides(&shape);
        if axes.len() == shape.len() {
            let mut permuted = strides.clone();
            for (i, axis) in axes.iter().enumerate() {
                if *axis < strides.len() {
                    permuted[i] = strides[*axis];
                }
            }
            strides = permuted;
        }

        let schema = TensorSchema::new(DType::F32, shape, layout, block_shape, strides)?;
        Self::load_with_schema(dir, &schema).await
    }

    pub(crate) fn block_len(&self) -> usize {
        self.storage.schema.block_len().max(1)
    }

    pub(crate) fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        self.schema.validate_coord(coord)?;

        let base_coord = self.view.resolve_coord(coord)?;
        self.storage.schema.validate_coord(&base_coord)?;

        Ok(base_coord)
    }

    pub(crate) fn linear_offset_from_base_coord(&self, base_coord: &[u64]) -> u64 {
        base_coord
            .iter()
            .zip(self.storage.schema.strides.iter())
            .map(|(coord, stride)| *coord * (*stride as u64))
            .sum()
    }

    pub(crate) fn sparse_key(&self, coords: &[u64], block_offset: u64) -> Vec<u64> {
        let sparse_axis = match self.storage.schema.layout {
            Layout::Sparse { axis } => axis.unwrap_or(0),
            Layout::Dense => 0,
        };

        let axis = sparse_axis.min(coords.len().saturating_sub(1));
        vec![coords[axis], block_offset]
    }

    pub(crate) fn has_sparse_index(&self) -> bool {
        self.storage.index.is_some()
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

    pub(crate) async fn read_value_impl(&self, coord: &[u64]) -> Result<f32> {
        let base_coord = self.resolve_base_coord(coord)?;

        let offset = self.linear_offset_from_base_coord(&base_coord);
        let block_len = self.block_len() as u64;
        let block_offset = offset / block_len;
        let offset_in_block = (offset % block_len) as usize;

        let block_id = if self.has_sparse_index() {
            let key = self.sparse_key(&base_coord, block_offset);
            self.lookup_block_id(&key).await?
        } else {
            Some(block_offset)
        };

        if let Some(block_id) = block_id {
            if let Some(block) = self.read_block(block_id).await? {
                validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
                Ok(block[offset_in_block])
            } else {
                Ok(0.0)
            }
        } else {
            Ok(0.0)
        }
    }

    pub(crate) async fn write_value_impl(&self, coord: &[u64], value: f32) -> Result<()> {
        let base_coord = self.resolve_base_coord(coord)?;

        let offset = self.linear_offset_from_base_coord(&base_coord);
        let block_len = self.block_len() as u64;
        let block_offset = offset / block_len;
        let offset_in_block = (offset % block_len) as usize;

        if self.has_sparse_index() {
            let key = self.sparse_key(&base_coord, block_offset);
            let existing_block = self.lookup_block_id(&key).await?;

            let block_id = if let Some(block_id) = existing_block {
                block_id
            } else if value != 0.0 {
                let block_id: u64 = rand::random();
                self.write_block(block_id, vec![0.0; self.block_len()])
                    .await?;
                self.upsert_block_id(key, block_id).await?;
                block_id
            } else {
                return Ok(());
            };

            if let Some(mut block) = self.read_block(block_id).await? {
                validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
                block[offset_in_block] = value;
                self.write_block(block_id, block).await
            } else {
                Err(Error::SparseIndex(
                    "sparse index points to missing block".to_string(),
                ))
            }
        } else {
            let block_id = block_offset;
            let mut block = self
                .read_block(block_id)
                .await?
                .unwrap_or_else(|| vec![0.0; self.block_len()]);

            validate::ensure_offset_in_bounds(offset_in_block, block.len())?;
            block[offset_in_block] = value;
            self.write_block(block_id, block).await
        }
    }

    pub(crate) fn reshape_impl(mut self, shape: Shape) -> Result<Self> {
        let old_size: usize = self.schema.shape.iter().product();
        let new_size: usize = shape.iter().product();

        if old_size != new_size {
            return Err(Error::InvalidLayout(
                "reshape requires an equal number of elements".to_string(),
            ));
        }

        if self.schema.shape != self.storage.schema.shape {
            return Err(Error::Unsupported(
                "reshape over transformed views is not implemented".to_string(),
            ));
        }

        self.schema.shape = shape;
        self.schema.strides = contiguous_strides(&self.schema.shape);
        self.view = TensorView::identity(&self.schema);
        Ok(self)
    }

    pub(crate) fn slice_impl(mut self, range: Range) -> Result<Self> {
        let (view, shape, strides) = self.view.slice(&self.schema.shape, &range)?;
        self.view = view;
        self.schema.shape = shape;
        self.schema.strides = strides;

        Ok(self)
    }

    pub(crate) fn transpose_impl(mut self, permutation: Option<Axes>) -> Result<Self> {
        let permutation = default_permutation(self.schema.shape.len(), permutation)?;
        self.view = self.view.transpose(&permutation)?;

        let mut shape = Shape::with_capacity(self.schema.shape.len());
        let mut strides = Vec::with_capacity(self.schema.shape.len());

        for axis in permutation {
            shape.push(self.schema.shape[axis]);
            strides.push(self.schema.strides[axis]);
        }

        self.schema.shape = shape;
        self.schema.strides = strides.into();

        Ok(self)
    }

    pub(crate) async fn read_block_impl(&self, block_id: u64) -> Result<Option<Vec<f32>>> {
        let blocks = self.storage.blocks.read().await;
        if let Some(file) = blocks.get_file(&block_id.to_string()) {
            let guard = file.read::<Vec<f32>>().await?;
            Ok(Some(guard.clone()))
        } else {
            Ok(None)
        }
    }

    pub(crate) async fn write_block_impl(&self, block_id: u64, block: Vec<f32>) -> Result<()> {
        let file = {
            let blocks = self.storage.blocks.read().await;
            blocks.get_file(&block_id.to_string()).cloned()
        };

        if let Some(file) = file {
            let mut guard = file.write::<Vec<f32>>().await?;
            *guard = block;
            Ok(())
        } else {
            let mut blocks = self.storage.blocks.write().await;
            blocks.create_file(block_id.to_string(), block, 0)?;
            Ok(())
        }
    }

    pub(crate) async fn lookup_block_id_impl(&self, key: &[u64]) -> Result<Option<u64>> {
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
    }

    pub(crate) async fn upsert_block_id_impl(&self, key: Vec<u64>, block_id: u64) -> Result<()> {
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
    }
}

impl<FE> Tensor for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    type DType = f32;

    fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    fn dtype(&self) -> Self::DType {
        0.0
    }
}

impl<FE> TensorRead for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        dense::read_value(self, coord)
    }
}

impl<FE> TensorWrite for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    fn write_value<'a>(
        &'a self,
        coord: &'a [u64],
        value: Self::DType,
    ) -> BoxFuture<'a, Result<()>> {
        dense::write_value(self, coord, value)
    }
}

impl<FE> TensorTransform for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        view_ops::reshape(self, shape)
    }

    fn slice(self, range: Range) -> Result<Self> {
        view_ops::slice(self, range)
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        view_ops::transpose(self, permutation)
    }
}

impl<FE> TensorBlockStore for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    type Block = Vec<f32>;

    fn read_block<'a>(&'a self, block_id: u64) -> BoxFuture<'a, Result<Option<Self::Block>>> {
        io::read_block(self, block_id)
    }

    fn write_block<'a>(&'a self, block_id: u64, block: Self::Block) -> BoxFuture<'a, Result<()>> {
        io::write_block(self, block_id, block)
    }
}

impl<FE> TensorSparseIndex for Tensor<FE>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    fn lookup_block_id<'a>(&'a self, key: &'a [u64]) -> BoxFuture<'a, Result<Option<u64>>> {
        sparse::lookup_block_id(self, key)
    }

    fn upsert_block_id<'a>(&'a self, key: Vec<u64>, block_id: u64) -> BoxFuture<'a, Result<()>> {
        sparse::upsert_block_id(self, key, block_id)
    }
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
    let layout = match schema.layout {
        Layout::Dense => "dense".to_string(),
        Layout::Sparse { axis } => format!(
            "sparse:{}",
            axis.map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        ),
    };

    let shape = schema
        .shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let block_shape = schema
        .block_shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let strides = schema
        .strides
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "version={METADATA_VERSION}\ndtype=f32\nlayout={layout}\nshape={shape}\nblock_shape={block_shape}\nstrides={strides}\n"
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

    let dtype = match fields.get("dtype").map(String::as_str) {
        Some("f32") => DType::F32,
        Some(other) => {
            return Err(Error::InvalidSchema(format!(
                "unsupported dtype in metadata: {other}"
            )));
        }
        None => {
            return Err(Error::InvalidSchema(
                "missing dtype in metadata".to_string(),
            ));
        }
    };

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
}

fn create_or_load_index<FE>(
    schema: &TensorSchema,
    dir_guard: &mut freqfs::DirWriteGuard<'_, FE>,
    create_if_missing: bool,
) -> std::io::Result<
    Option<TableLock<SparseTableSchema, SparseIndexSchema, b_table::collate::Collator<u64>, FE>>,
>
where
    FE: FileLoad + AsType<b_table::Node<u64>> + AsType<Vec<f32>> + Send + Sync + 'static,
{
    if matches!(schema.layout, Layout::Sparse { .. }) {
        let index_dir = if create_if_missing {
            dir_guard.create_dir(INDEX.to_string())?
        } else if dir_guard.contains(INDEX) {
            dir_guard.get_or_create_dir(INDEX.to_string())?
        } else {
            return Ok(None);
        };

        let table_schema = SparseTableSchema::default();
        let collator = b_table::collate::Collator::default();

        if create_if_missing {
            TableLock::create(table_schema, collator, index_dir).map(Some)
        } else {
            TableLock::load(table_schema, collator, index_dir).map(Some)
        }
    } else {
        Ok(None)
    }
}
