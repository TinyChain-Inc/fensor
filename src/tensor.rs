mod construction;
mod dense_storage;
mod sparse_storage;

#[cfg(test)]
#[path = "../tests/unit/tensor/physical_tests.rs"]
mod physical_tests;

#[cfg(test)]
#[path = "../tests/unit/tensor/dtype_storage.rs"]
mod dtype_storage;

use std::sync::Arc;

use freqfs::{DirLock, FileLoad};
use futures::{StreamExt, TryStreamExt};
use get_size::GetSize;
use number_general::NumberType;
use safecast::AsType;

use self::dense_storage::DenseStorage;
use crate::error::{Error, Result};
use crate::mapping::CoordinateMap;
use crate::request::BatchRequest;
use crate::schema::{BlockPosition, Layout, MAX_BLOCK_CAPACITY, StorageSchema, TensorSchema};
use crate::traits::{
    BoxFuture, TensorArray, TensorGeometry, TensorRead, TensorTransform, TensorWrite,
    TensorWriteBulk,
};
use crate::view::TensorView;
use crate::{Range, TensorMetadata, TensorSource, validate};

const BLOCKS: &str = "blocks";

const INDEX: &str = "index";

const METADATA: &str = "metadata";

/// Maximum sparse-index keys retained per page, independent of execution batches.
const SPARSE_INDEX_PAGE_ENTRIES: usize = 4096;

/// A concrete numerical element. Filesystem adapters own payload serialization;
/// elements themselves need no destream implementation.
pub trait TensorElement:
    ha_ndarray::Number + number_general::DType + Copy + Default + PartialEq + Send + Sync + 'static
{
}

macro_rules! tensor_elements {
    ($($ty:ty),+ $(,)?) => { $(impl TensorElement for $ty {})+ };
}

tensor_elements!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cfg(feature = "complex")]
tensor_elements!(
    ha_ndarray::complex::Complex32,
    ha_ndarray::complex::Complex64
);

pub trait TensorFileEntry<T: TensorElement>:
    FileLoad
    + AsType<crate::SparseNode<T>>
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
        + AsType<crate::SparseNode<T>>
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

// Tensor already shares this owner through Arc; boxing a variant adds another allocation.
#[allow(clippy::large_enum_variant)]
enum Storage<FE, T> {
    Dense(DenseStorage<FE>),
    Sparse(sparse_storage::SparseStorage<FE, T>),
}

impl<FE, T> Storage<FE, T> {
    fn sparse(&self) -> Option<&sparse_storage::SparseStorage<FE, T>> {
        match self {
            Self::Sparse(s) => Some(s),
            Self::Dense(_) => None,
        }
    }

    pub(crate) fn storage_schema(&self) -> &StorageSchema {
        match self {
            Self::Dense(s) => s.storage_schema(),
            Self::Sparse(s) => &s.geometry().storage,
        }
    }
}

// A typed buffer plus its adapter envelope. Cache reconciliation charges the
// actual entry after creation/replacement; reserve all retained Vec capacity.
fn block_allocation<FE, T>(values: &Vec<T>) -> usize {
    std::mem::size_of::<FE>()
        + std::mem::size_of::<Vec<T>>()
        + values.capacity() * std::mem::size_of::<T>()
}

pub struct Tensor<FE, T> {
    directory: DirLock<FE>,
    storage: Arc<Storage<FE, T>>,
    schema: TensorSchema,
}

impl<FE, T> Clone for Tensor<FE, T> {
    fn clone(&self) -> Self {
        Self {
            directory: self.directory.clone(),
            storage: self.storage.clone(),
            schema: self.schema.clone(),
        }
    }
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
    /// quiesces mutations for a precise persistence boundary; synchronization does
    /// not exclude writes for its entire call. The caller owns durable directory
    /// synchronization and transaction policy.
    pub async fn sync(&self) -> Result<()>
    where
        FE: freqfs::FileSave + Clone,
    {
        match self.storage.as_ref() {
            Storage::Dense(dense) => dense.sync().await,
            Storage::Sparse(sparse) => sparse.sync().await,
        }
    }

    pub async fn create(
        dir: DirLock<FE>,
        schema: TensorSchema,
        layout: Layout,
        max_capacity: usize,
    ) -> Result<Self> {
        let tensor = Self::unpublished(dir, schema, layout, max_capacity).await?;
        tensor.initialize_dense_blocks().await?;
        tensor.publish_metadata().await?;
        Ok(tensor)
    }

    async fn unpublished(
        dir: DirLock<FE>,
        schema: TensorSchema,
        layout: Layout,
        max_capacity: usize,
    ) -> Result<Self> {
        if max_capacity == 0 || max_capacity > MAX_BLOCK_CAPACITY {
            return Err(Error::InvalidSchema(format!(
                "max block capacity must be non-zero and at most {MAX_BLOCK_CAPACITY}, got {max_capacity}"
            )));
        }
        validate_tensor_dtype::<T>(schema.dtype())?;
        let storage = StorageSchema::new(schema.shape().as_slice(), layout, max_capacity)?;
        Self::unpublished_with_geometry(dir, crate::StorageGeometry { schema, storage }).await
    }

    /// Create empty delegated storage with validated, caller-selected block geometry.
    pub async fn create_with_geometry(
        dir: DirLock<FE>,
        geometry: crate::StorageGeometry,
    ) -> Result<Self> {
        let tensor = Self::unpublished_with_geometry(dir, geometry).await?;
        tensor.initialize_dense_blocks().await?;
        tensor.publish_metadata().await?;
        Ok(tensor)
    }

    async fn unpublished_with_geometry(
        dir: DirLock<FE>,
        geometry: crate::StorageGeometry,
    ) -> Result<Self> {
        validate_tensor_dtype::<T>(geometry.schema.dtype())?;
        let blocks = {
            let mut guard = dir.try_write()?;
            if !guard.is_empty() {
                return Err(Error::InvalidSchema(
                    "tensor creation requires empty storage".into(),
                ));
            }
            guard.create_dir(BLOCKS.to_string())?
        };

        let schema = geometry.schema.clone();
        let storage = match geometry.layout() {
            Layout::Dense => Storage::Dense(DenseStorage::new(blocks, geometry.storage)),
            Layout::Sparse { .. } => Storage::Sparse(
                sparse_storage::SparseStorage::create(&dir, blocks, geometry).await?,
            ),
        };

        Ok(Self {
            directory: dir,
            storage: Arc::new(storage),
            schema,
        })
    }

    /// Construct dense storage from exactly one row-major value per element.
    /// Input may borrow its caller and need not be Unpin. Metadata is published
    /// only on success; callers own cleanup of failed or cancelled construction.
    pub async fn from_values<S, E>(
        dir: DirLock<FE>,
        schema: TensorSchema,
        max_capacity: usize,
        values: S,
    ) -> std::result::Result<Self, E>
    where
        S: futures::Stream<Item = std::result::Result<T, E>> + Send,
        E: From<Error>,
    {
        let output = Self::unpublished(dir, schema, Layout::Dense, max_capacity).await?;
        let expected = crate::schema::checked_product(output.shape())?;
        let output = dense_storage::Construction(output);
        let values = values.fuse();
        futures::pin_mut!(values);
        let mut received = 0u64;

        loop {
            let mut batch = Vec::new();

            while batch.len() < crate::expression::MAX_BATCH_ELEMENTS {
                let Some(value) = values.try_next().await? else {
                    break;
                };

                if received + batch.len() as u64 >= expected {
                    return Err(Error::InvalidLayout("too many Tensor values".into()).into());
                }
                batch.push(value);
            }

            if batch.is_empty() {
                break;
            }

            let request = BatchRequest::linear(received, batch.len())?;
            received += batch.len() as u64;
            output.stage(request, batch).await?;
        }

        if received != expected {
            return Err(Error::InvalidLayout("too few Tensor values".into()).into());
        }

        Ok(output.finish().await?)
    }

    /// Construct sparse storage from strictly ordered, unique coordinates.
    /// Zeros are validated but omitted. Input errors are preserved unchanged;
    /// failed or cancelled construction has no loadable metadata.
    pub async fn from_sparse_elements<S, E>(
        dir: DirLock<FE>,
        schema: TensorSchema,
        layout: Layout,
        max_capacity: usize,
        entries: S,
    ) -> std::result::Result<Self, E>
    where
        S: futures::Stream<Item = std::result::Result<(Vec<u64>, T), E>> + Send,
        E: From<Error>,
    {
        if !matches!(layout, Layout::Sparse { .. }) {
            return Err(
                Error::InvalidLayout("sparse construction requires sparse layout".into()).into(),
            );
        }

        let output = Self::unpublished(dir, schema, layout, max_capacity).await?;
        construction::sparse(output, entries).await
    }

    /// Create an independent tensor by consuming a reader in bounded batches.
    ///
    /// Evaluation is driven by reads; no intermediate tensor is created. The
    /// destination uses the reader's dtype and shape with the caller's layout.
    /// The reported source layout does not prescribe destination allocation; sparse
    /// axes may differ. Dense/sparse kind changes require explicit conversion first.
    /// Sparse output omits zeros. Errors propagate; the caller cleans up partial
    /// destination storage. Coordinate-bearing blocks may be
    /// unordered; bounds, lengths, and total count are checked. Exactly-once coverage
    /// is the source contract, without a whole-output duplicate detector.
    /// Updates are grouped by destination block within each bounded incoming batch;
    /// no block guard or computed result cache is retained across batches. Failures
    /// may leave partial output without a guaranteed update prefix or rollback.
    /// Each destination update completes before the next source batch is requested.
    /// Dropping the copy cancels its active operation and releases the source stream.
    /// The source retains its own bounded buffering in addition to the current batch.
    pub async fn copy_from<R>(
        dir: DirLock<FE>,
        source: &R,
        destination_layout: Layout,
        max_capacity: usize,
    ) -> Result<Self>
    where
        R: TensorRead<DType = T> + ?Sized,
    {
        let schema =
            TensorSchema::new(<T as number_general::DType>::dtype(), source.shape().into())?;
        if matches!(source.layout(), Layout::Dense) != matches!(destination_layout, Layout::Dense) {
            return Err(Error::Unsupported(
                "copy requires matching dense/sparse kinds; convert the source explicitly first"
                    .into(),
            ));
        }

        let output = Self::unpublished(dir, schema, destination_layout, max_capacity).await?;
        if matches!(destination_layout, Layout::Sparse { .. }) {
            let entries = source
                .read_sparse_elements_in_order(
                    crate::validate::full_range(source.shape()),
                    (0..source.ndim()).collect(),
                )
                .await?;
            return construction::sparse(output, entries).await;
        }

        let output = dense_storage::Construction(output);
        let mut blocks = source.read_coordinate_blocks()?;
        let expected = crate::schema::checked_product(source.shape())?;
        let mut received = 0u64;

        while let Some((coords, values)) = blocks.try_next().await? {
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
            output
                .stage(BatchRequest::explicit(coords)?, values)
                .await?;
        }

        if received != expected {
            return Err(Error::InvalidLayout(
                "reader returned fewer values than its shape".into(),
            ));
        }

        drop(blocks);
        output.finish().await
    }

    /// Strictly reopen materialized storage without creating or repairing files.
    /// The caller excludes concurrent native writes while loading and validating.
    pub async fn load(dir: DirLock<FE>) -> Result<Self> {
        let blocks = dir
            .try_read()?
            .get_dir(BLOCKS)
            .cloned()
            .ok_or_else(|| Error::InvalidSchema("tensor missing blocks directory".into()))?;
        let geometry = load_metadata_file::<FE, T>(&blocks).await?;
        validate_tensor_dtype::<T>(geometry.schema.dtype())?;
        let schema = geometry.schema.clone();
        let storage = match geometry.layout() {
            Layout::Dense => {
                if dir.try_read()?.get_dir(INDEX).is_some() {
                    return Err(Error::InvalidSchema(
                        "dense tensor contains a sparse index".into(),
                    ));
                }
                Storage::Dense(DenseStorage::new(blocks, geometry.storage))
            }
            Layout::Sparse { .. } => {
                Storage::Sparse(sparse_storage::SparseStorage::load(&dir, blocks, geometry)?)
            }
        };

        let tensor = Self {
            directory: dir,
            storage: Arc::new(storage),
            schema,
        };
        tensor.validate().await?;
        Ok(tensor)
    }

    /// Validate required blocks and sparse index references with bounded scratch.
    /// Validation is not a transaction snapshot across concurrent mutations.
    pub async fn validate(&self) -> Result<()> {
        match self.storage.as_ref() {
            Storage::Sparse(sparse) => {
                sparse.validate().await?;
            }
            Storage::Dense(dense) => {
                dense.validate::<T>().await?;
            }
        }

        Ok(())
    }

    /// Durably synchronize only this native materialization.
    pub async fn sync_all(&self) -> Result<()>
    where
        FE: freqfs::FileSave + Clone,
    {
        match self.storage.as_ref() {
            Storage::Dense(dense) => dense.sync_all(&self.directory).await,
            Storage::Sparse(sparse) => sparse.sync_all(&self.directory).await,
        }
    }

    pub fn view(&self) -> TensorView<Self> {
        TensorView::new(self.clone())
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

    pub(crate) fn block_len(&self) -> usize {
        self.block_shape().iter().product::<u64>() as usize
    }

    // One validated storage block, independent of tensor size.
    fn default_block(&self) -> Vec<T> {
        vec![T::default(); self.block_len()]
    }

    async fn initialize_dense_blocks(&self) -> Result<()> {
        let Storage::Dense(dense) = self.storage.as_ref() else {
            return Ok(());
        };

        dense.initialize::<T>().await
    }

    async fn write_block_updates(&self, id: u64, updates: &[(usize, T)]) -> Result<()> {
        match self.storage.as_ref() {
            Storage::Sparse(sparse) => sparse.update(id, updates).await?,
            Storage::Dense(dense) => dense.update(id, updates).await?,
        }

        #[cfg(test)]
        copy_metrics::record(|m| m.block_updates += 1);
        Ok(())
    }

    // Publish metadata only after construction succeeds; synchronization is caller-owned.
    async fn publish_metadata(&self) -> Result<()> {
        let metadata = TensorMetadata::<T>::new(
            self.schema.shape().clone(),
            self.layout(),
            self.storage.storage_schema().block_schema.shape.clone(),
        )?;
        match self.storage.as_ref() {
            Storage::Dense(dense) => dense.publish_metadata(metadata).await,
            Storage::Sparse(sparse) => sparse.publish_metadata(metadata).await,
        }
    }

    /// Plan one bounded request by logical block and scatter position.
    fn plan_read_groups(
        &self,
        request: &BatchRequest,
        mapping: Option<&CoordinateMap>,
    ) -> Result<crate::storage_read::LogicalGroups> {
        let storage = crate::storage_read::StorageShape {
            shape: self.shape(),
            strides: self.strides(),
            block_shape: self.block_shape(),
            block_strides: self.block_strides(),
            grid_strides: self.grid_strides(),
        };
        storage.plan(request, mapping)
    }

    /// Borrow each needed block once; all requests and results fit one execution batch.
    pub(crate) async fn read_batch(
        &self,
        request: &BatchRequest,
        mapping: Option<&CoordinateMap>,
    ) -> Result<Vec<T>> {
        let dense = match self.storage.as_ref() {
            Storage::Sparse(sparse) => {
                let groups = self.plan_read_groups(request, mapping)?;
                let mut values = vec![T::ZERO; request.len()];

                sparse.read_groups(&groups, &mut values).await?;
                return Ok(values);
            }
            Storage::Dense(dense) => dense,
        };

        let groups = self.plan_read_groups(request, mapping)?;

        let mut values = vec![T::ZERO; request.len()];

        for (id, positions) in groups {
            let block = dense.read::<T>(id).await?;

            #[cfg(test)]
            crate::read_metrics::record(|m| {
                m.borrows += 1;
                m.borrowed += block.len();
            });

            for run in positions {
                run.scatter(&block, &mut values)?;
            }

            // This block guard is released before the next block is awaited.
        }

        Ok(values)
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
    crate::expression::reader_members!(
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );

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
                block_id,
                offset_in_block,
            } = self.block_position_from_base_coord(coord);
            self.write_block_updates(block_id, &[(offset_in_block, value)])
                .await
        })
    }
}

// Required dense reads borrow validated payloads; consumers clone only when needed.
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
            let expected = validate::iter_range_coords(self.shape(), &range)?.remaining();
            if expected != values.len() as u64 {
                return Err(Error::DataMismatch(format!(
                    "expected {} values for range but got {}",
                    expected,
                    values.len()
                )));
            }

            if values.is_empty() {
                return Ok(());
            }

            let view = self.view().slice(range)?;
            let geometry = self.storage_geometry();
            let mut values = values.into_iter();
            let mut start = 0;

            loop {
                let batch: Vec<_> = values
                    .by_ref()
                    .take(crate::expression::MAX_BATCH_ELEMENTS)
                    .collect();
                if batch.is_empty() {
                    break;
                }

                let len = batch.len() as u64;

                for (id, updates) in view.plan_updates(&geometry, start, batch)? {
                    self.write_block_updates(id, &updates).await?;
                }
                start += len;
            }

            Ok(())
        })
    }

    fn fill<'a>(&'a self, value: Self::DType) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            use crate::TensorSource;
            let geometry = self.storage_geometry();

            for id in 0..geometry.block_count() {
                // Validate the selected stored block before overwriting it, as scalar writes do.
                let mut values = self.read_logical_block(id).await?;
                values.fill(T::ZERO);

                for offset in geometry.block_offsets(id)? {
                    values[offset] = value;
                }

                self.replace_logical_block(id, values).await?;
            }

            Ok(())
        })
    }
}

// Dense reads and replacement require an existing, correctly typed payload.
fn validate_stored_length<T>(block: &[T], expected: usize) -> Result<()> {
    if block.len() != expected {
        return Err(Error::InvalidLayout("invalid stored block length".into()));
    }

    Ok(())
}

// Validate all offsets before applying a bounded batch in its original order.
fn apply_block_updates<T: Copy>(
    block: &mut [T],
    expected: usize,
    updates: &[(usize, T)],
) -> Result<()> {
    validate_stored_length(block, expected)?;

    for &(offset, _) in updates {
        validate::ensure_offset_in_bounds(offset, block.len())?;
    }

    for &(offset, value) in updates {
        block[offset] = value;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------
async fn create_metadata_file<FE: TensorFileEntry<T>, T: TensorElement>(
    blocks: &DirLock<FE>,
    metadata: TensorMetadata<T>,
) -> Result<()> {
    let size = std::mem::size_of::<FE>() + metadata.get_size();
    blocks
        .write()
        .await
        .create_file(METADATA.to_string(), metadata, size)
        .await?;
    Ok(())
}

async fn load_metadata_file<FE, T>(blocks: &DirLock<FE>) -> Result<crate::StorageGeometry>
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
    let (schema, storage) = file.read::<TensorMetadata<T>>().await?.schemas()?;
    Ok(crate::StorageGeometry { schema, storage })
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
#[path = "../tests/unit/tensor/metadata_tests.rs"]
mod metadata_tests;

#[cfg(test)]
#[path = "../tests/unit/tensor/sparse_axis_tests.rs"]
mod sparse_axis_tests;

#[cfg(test)]
#[path = "../tests/unit/tensor/sparse_lifecycle_tests.rs"]
mod sparse_lifecycle_tests;

// Task-local counters cannot mix observations from concurrent test fixtures.
// No instrumentation or configuration is present in production builds.
#[cfg(test)]
#[path = "../tests/unit/tensor/copy_metrics.rs"]
pub(crate) mod copy_metrics;

impl<FE: TensorFileEntry<T>, T: TensorElement> crate::TensorSource for Tensor<FE, T> {
    fn storage_geometry(&self) -> crate::StorageGeometry {
        crate::StorageGeometry {
            schema: self.schema.clone(),
            storage: self.storage.storage_schema().clone(),
        }
    }

    fn read_storage<'a>(&'a self, read: crate::StorageRead<'a>) -> BoxFuture<'a, Result<Vec<T>>> {
        Box::pin(self.read_batch(read.request, read.mapping))
    }

    fn occupied_blocks(
        &self,
        range: std::ops::Range<u64>,
    ) -> futures::stream::BoxStream<'static, Result<u64>> {
        futures::stream::try_unfold((self.clone(), range), move |(tensor, range)| async move {
            let keys = tensor
                .storage
                .sparse()
                .ok_or_else(|| Error::InvalidLayout("dense storage has no occupancy".into()))?
                .occupied_page(range.clone())
                .await?;
            let Some(&last) = keys.last() else {
                return Ok::<_, Error>(None);
            };

            Ok(Some((
                futures::stream::iter(keys.into_iter().map(Ok)),
                (tensor, last + 1..range.end),
            )))
        })
        .try_flatten()
        .boxed()
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, Result<Vec<T>>> {
        Box::pin(async move {
            match self.storage.as_ref() {
                Storage::Sparse(sparse) => sparse.read(id).await,
                Storage::Dense(dense) => {
                    self.storage_geometry().block_bounds(id)?;
                    Ok(dense.read::<T>(id).await?.to_vec())
                }
            }
        })
    }
}

impl<FE: TensorFileEntry<T>, T: TensorElement> Tensor<FE, T> {
    /// Replace one complete logical block. Callers exclude related native writes.
    /// Sparse replacement serializes payload and index changes under one native table write guard.
    /// After failed or cancelled mutation, callers must discard all handles to the
    /// affected storage and coordinate recovery before reuse.
    /// Failure may leave partial changes; this method supplies no transaction policy.
    pub async fn replace_logical_block(&self, id: u64, block: Vec<T>) -> Result<()> {
        match self.storage.as_ref() {
            Storage::Sparse(sparse) => sparse.replace(id, block).await?,
            Storage::Dense(dense) => dense.replace(id, block).await?,
        }

        #[cfg(test)]
        copy_metrics::record(|m| m.block_updates += 1);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) use sparse_storage::corruption;
