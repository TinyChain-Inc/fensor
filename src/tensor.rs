mod dense_storage;
mod sparse_storage;

pub use sparse_storage::SparseBlocks;

#[cfg(test)]
#[path = "../tests/unit/tensor/construction/tests.rs"]
mod construction_tests;

#[cfg(test)]
#[path = "../tests/unit/tensor/physical_tests.rs"]
mod physical_tests;

#[cfg(test)]
#[path = "../tests/unit/tensor/dtype_storage.rs"]
mod dtype_storage;

use std::sync::Arc;

use freqfs::{DirLock, FileLoad};
use futures::{StreamExt, TryStreamExt};
use number_general::NumberType;
use safecast::AsType;

use self::dense_storage::DenseStorage;
use crate::error::{Error, Result};
use crate::request::BatchRequest;
use crate::schema::{Layout, MAX_BLOCK_CAPACITY, StorageSchema, TensorSchema};
use crate::traits::{BoxFuture, TensorArray, TensorGeometry, TensorRead, TensorWrite};
use crate::view::TensorView;
use crate::{Range, TensorMetadata, TensorSource, validate};

const BLOCKS: &str = "blocks";

const METADATA: &str = "metadata";

/// Maximum occupied block IDs buffered by one scan step.
const OCCUPIED_BATCH_SIZE: usize = 4096;

/// A concrete numerical element supported by Tensor operations.
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
    + AsType<b_tree::Node<Vec<Vec<crate::SparseCell<T>>>>>
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
        + AsType<b_tree::Node<Vec<Vec<crate::SparseCell<T>>>>>
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
    Sparse(sparse_storage::SparseBlocks<FE, T>),
}

impl<FE, T> Storage<FE, T> {
    fn sparse(&self) -> Option<&sparse_storage::SparseBlocks<FE, T>> {
        match self {
            Self::Sparse(s) => Some(s),
            Self::Dense(_) => None,
        }
    }

    fn geometry(&self) -> &crate::StorageGeometry {
        match self {
            Self::Dense(s) => s.geometry(),
            Self::Sparse(s) => s.geometry(),
        }
    }
}

pub struct Tensor<FE, T> {
    directory: DirLock<FE>,
    storage: Arc<Storage<FE, T>>,
}

impl<FE, T> Clone for Tensor<FE, T> {
    fn clone(&self) -> Self {
        Self {
            directory: self.directory.clone(),
            storage: self.storage.clone(),
        }
    }
}

impl<FE, T> Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    /// Write back this native materialization through its owning directory.
    ///
    /// The caller quiesces mutations for a precise writeback boundary and uses
    /// `sync_all` when durability is required.
    pub async fn sync(&self) -> Result<()>
    where
        FE: freqfs::FileSave,
    {
        self.directory.sync().await.map_err(Error::from)
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

        let storage = match geometry.layout() {
            Layout::Dense => Storage::Dense(DenseStorage::new(blocks, geometry)),
            Layout::Sparse { .. } => Storage::Sparse(
                sparse_storage::SparseBlocks::create_canonical(&dir, blocks, geometry).await?,
            ),
        };

        Ok(Self {
            directory: dir,
            storage: Arc::new(storage),
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
        output.initialize_sparse(entries).await
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
                .read_sparse_elements(crate::validate::full_range(source.shape()))
                .await?;

            return output.initialize_sparse(entries).await;
        }

        let output = dense_storage::Construction(output);
        let mut blocks = source.read_coordinate_blocks()?;
        let expected = crate::schema::checked_product(source.shape())?;
        let mut received = 0u64;

        while let Some(batch) = blocks.try_next().await? {
            tokio::task::coop::consume_budget().await;
            let (request, values) = batch.into_request(source.shape())?;

            received = received
                .checked_add(values.len() as u64)
                .filter(|n| *n <= expected)
                .ok_or_else(|| {
                    Error::InvalidLayout("reader returned more values than its shape".into())
                })?;

            output.stage(request, values).await?;
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
        let (blocks, entries) = {
            let root = dir.try_read()?;
            let blocks = root.get_dir(BLOCKS).cloned().ok_or_else(|| {
                freqfs::IntegrityError::invalid_data("missing tensor blocks directory")
            })?;

            (blocks, root.len())
        };

        let geometry = load_metadata_file::<FE, T>(&blocks).await?;
        validate_tensor_dtype::<T>(geometry.schema.dtype())?;
        let storage = match geometry.layout() {
            Layout::Dense => {
                if entries != 1 {
                    return Err(freqfs::IntegrityError::invalid_data(
                        "unexpected dense tensor root entry",
                    )
                    .into());
                }

                let dense = DenseStorage::new(blocks, geometry);
                dense.validate::<T>().await?;
                Storage::Dense(dense)
            }
            Layout::Sparse { .. } => {
                Storage::Sparse(sparse_storage::SparseBlocks::load(dir.clone(), geometry).await?)
            }
        };

        let tensor = Self {
            directory: dir,
            storage: Arc::new(storage),
        };

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

    /// Durably synchronize this native materialization through its owning directory.
    /// The caller quiesces mutations for a precise durability boundary.
    pub async fn sync_all(&self) -> Result<()>
    where
        FE: freqfs::FileSave,
    {
        self.directory.sync_all().await.map_err(Error::from)
    }

    pub fn view(&self) -> TensorView<Self> {
        TensorView::new(self.clone())
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

    fn initialize_sparse<S, E>(
        mut self,
        entries: S,
    ) -> impl std::future::Future<Output = std::result::Result<Self, E>>
    where
        S: futures::Stream<Item = std::result::Result<(Vec<u64>, T), E>> + Send,
        E: From<Error>,
    {
        // Keep the packer and append futures off the caller's stack, including
        // when a decoder already carries the futures for all supported dtypes.
        Box::pin(async move {
            let storage =
                Arc::get_mut(&mut self.storage).expect("unpublished Tensor has one owner");
            let Storage::Sparse(sparse) = storage else {
                unreachable!("sparse Tensor initialization");
            };

            // The packer borrows geometry while the unpublished storage consumes its blocks.
            let geometry = sparse.geometry().clone();
            futures::pin_mut!(entries);
            let blocks = geometry.pack_sparse_blocks(entries)?;
            sparse.append_sorted(blocks).await?;
            self.publish_metadata().await?;
            Ok(self)
        })
    }

    // Publish metadata only after construction succeeds; synchronization is caller-owned.
    async fn publish_metadata(&self) -> Result<()> {
        let metadata = TensorMetadata::<T>::new(
            self.schema().shape().clone(),
            self.layout(),
            self.storage.geometry().block_capacity(),
        )?;

        match self.storage.as_ref() {
            Storage::Dense(dense) => dense.publish_metadata(metadata).await,
            Storage::Sparse(sparse) => sparse.publish_metadata(metadata).await,
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
        self.storage.geometry().layout()
    }

    fn shape(&self) -> &[u64] {
        self.schema().shape()
    }
}

impl<FE, T> TensorArray for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn schema(&self) -> &TensorSchema {
        self.storage.geometry().schema()
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
    crate::expression::reader_members!(read_blocks, read_coordinate_blocks, read_sparse_elements);

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let (id, offset) = self.storage_geometry().block_position(coord)?;
            self.read_block_value(id, offset).await
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
            let (id, offset) = self.storage_geometry().block_position(coord)?;
            self.write_block_updates(id, &[(offset, value)]).await
        })
    }
    fn write_values<'a>(
        &'a self,
        range: Range,
        values: Vec<Self::DType>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // Cardinality is checked without expanding the selected range.
            let geometry = self.storage_geometry();
            let selection = geometry.write_selection(Some(range))?;
            let expected = selection.size();
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

                for (id, updates) in selection.plan_updates(start, batch)? {
                    self.write_block_updates(id, &updates).await?;
                }

                start += len;
            }

            Ok(())
        })
    }

    fn fill<'a>(&'a self, value: Self::DType) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if value == T::ZERO
                && let Some(sparse) = self.storage.sparse()
            {
                return sparse.clear().await;
            }

            let geometry = self.storage_geometry();

            for id in 0..geometry.block_count() {
                self.replace_logical_block(id, vec![value; geometry.block_len(id)?])
                    .await?;
            }

            Ok(())
        })
    }
}

// Dense reads and replacement require an existing, correctly typed payload.
fn validate_stored_length<T>(block: &[T], expected: usize) -> Result<()> {
    if block.len() != expected {
        return Err(freqfs::IntegrityError::invalid_data("invalid stored block length").into());
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
    blocks
        .write()
        .await
        .create_file(METADATA.to_string(), metadata)
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
        .ok_or_else(|| freqfs::IntegrityError::invalid_data("missing tensor metadata file"))?;

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
    fn storage_geometry(&self) -> &crate::StorageGeometry {
        self.storage.geometry()
    }

    fn visit_blocks<'a, C>(&'a self, ids: &'a [u64], mut consume: C) -> BoxFuture<'a, Result<()>>
    where
        C: FnMut(u64, &[T]) -> Result<()> + Send + 'a,
    {
        Box::pin(async move {
            match self.storage.as_ref() {
                Storage::Dense(dense) => dense.visit_blocks(ids, &mut consume).await,
                Storage::Sparse(sparse) => sparse.visit_blocks(ids, &mut consume).await,
            }
        })
    }

    fn occupied_blocks(
        &self,
        range: std::ops::Range<u64>,
    ) -> futures::stream::BoxStream<'static, Result<u64>> {
        match self.storage.sparse() {
            Some(sparse) => sparse.occupied_blocks(range),
            None => futures::stream::once(async {
                Err(Error::InvalidLayout(
                    "dense storage has no occupancy".into(),
                ))
            })
            .boxed(),
        }
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, Result<Vec<T>>> {
        Box::pin(async move {
            match self.storage.as_ref() {
                Storage::Sparse(sparse) => match sparse.read(id).await? {
                    Some(values) => Ok(values),
                    None => Ok(vec![T::ZERO; sparse.geometry().block_len(id)?]),
                },
                Storage::Dense(dense) => Ok(dense.read::<T>(id).await?.to_vec()),
            }
        })
    }
}

impl<FE: TensorFileEntry<T>, T: TensorElement> Tensor<FE, T> {
    /// Replace one complete logical block. Callers exclude related native writes.
    /// Sparse replacement serializes payload changes under one native BTree write guard.
    /// After failed or cancelled mutation, callers must discard all handles to the
    /// affected storage and coordinate recovery before reuse.
    /// Failure may leave partial changes; this method supplies no transaction policy.
    pub async fn replace_logical_block(&self, id: u64, block: Vec<T>) -> Result<()> {
        match self.storage.as_ref() {
            Storage::Sparse(sparse) => {
                sparse.replace(id, Some(block)).await?;
            }
            Storage::Dense(dense) => dense.replace(id, block).await?,
        }

        #[cfg(test)]
        copy_metrics::record(|m| m.block_updates += 1);

        Ok(())
    }
}

#[cfg(test)]
pub(crate) use sparse_storage::corruption;
