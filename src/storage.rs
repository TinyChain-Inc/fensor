//! Native storage geometry and bounded sources for geometric expression leaves.
//!
//! Implementations own their visibility and mutation policy. fensor only maps and
//! evaluates bounded requests; it does not acquire transaction reservations.

use std::ops::Range;
use std::pin::Pin;

use futures::stream::{BoxStream, Peekable, Stream, try_unfold};
use futures::{StreamExt, TryStreamExt};
use ha_ndarray::Number;
use number_general::DType;

use crate::mapping::{CoordinateMap, StorageSlice};
use crate::request::BatchRequest;
use crate::schema::{StorageSchema, TensorSchema};
use crate::{
    AxisRange, BoxFuture, Error, Layout, Result, Shape, TensorArray, TensorElement,
    TensorExpression, TensorGeometry,
};

/// Logical-block updates from one bounded batch, in input order within each block.
/// Coordinates are block-local offsets, never physical payload identifiers.
pub type BlockUpdates<T> = std::collections::BTreeMap<u64, Vec<(usize, T)>>;

/// Pack ordered entries with at most one unfinished destination block.
#[derive(Default)]
struct SparseBlockBuilder<T>(Option<(u64, Vec<T>)>);

impl<T: TensorElement> SparseBlockBuilder<T> {
    fn push(
        &mut self,
        geometry: &StorageGeometry,
        (id, offset): (u64, usize),
        value: T,
    ) -> Option<(u64, Vec<T>)> {
        let previous = if self.0.as_ref().is_some_and(|(current, _)| *current != id) {
            self.take()
        } else {
            None
        };

        let (_, values) = self.0.get_or_insert_with(|| {
            let len = geometry.block_len(id).expect("validated block address");
            (id, vec![T::ZERO; len])
        });
        values[offset] = value;
        previous
    }

    fn take(&mut self) -> Option<(u64, Vec<T>)> {
        self.0
            .take()
            .filter(|(_, values)| values.iter().any(|&value| value != T::ZERO))
    }
}

/// A validated base-coordinate selection used to plan bounded block updates.
pub struct WriteSelection<'a> {
    geometry: &'a StorageGeometry,
    shape: Shape,
    strides: Vec<i128>,
    offset: i128,
    size: u64,
}

impl WriteSelection<'_> {
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether the write covers the base in its original shape and order.
    pub fn is_full(&self) -> bool {
        self.shape() == self.geometry.schema().shape().as_slice()
    }

    /// Conservative interval of logical blocks addressed by the selection.
    pub fn block_range(&self) -> Range<u64> {
        if self.size == 0 {
            return 0..0;
        }

        let last = self
            .shape
            .iter()
            .zip(&self.strides)
            .fold(self.offset, |last, (&len, stride)| {
                last + i128::from(len - 1) * stride
            });

        // Validated forward spans bound both endpoints to the base.
        let blocks = &self.geometry.storage.blocks;
        blocks.position(self.offset as u64).block_id..blocks.position(last as u64).block_id + 1
    }

    /// Plan one bounded batch of row-major values beginning at `start`.
    pub fn plan_updates<T: Copy>(&self, start: u64, values: Vec<T>) -> Result<BlockUpdates<T>> {
        let request = BatchRequest::linear(start, values.len())?;
        self.plan_batch(&request, values)
    }

    /// Consume an expression in its preferred request order.
    pub fn updates_from<T: TensorElement>(
        &self,
        source: TensorExpression<T>,
    ) -> Result<BoxStream<'_, Result<BlockUpdates<T>>>> {
        if self.shape() != source.shape() {
            return Err(Error::InvalidCoord(
                "update source and destination shapes differ".into(),
            ));
        }

        Ok(
            crate::expression::preferred_batches(std::sync::Arc::new(source))?
                .map(move |result| {
                    let (request, batch) = result?;
                    self.plan_batch(&request, batch.values)
                })
                .boxed(),
        )
    }

    fn plan_batch<T: Copy>(
        &self,
        request: &BatchRequest,
        values: Vec<T>,
    ) -> Result<BlockUpdates<T>> {
        if self.is_full() {
            return self.geometry.plan_base_updates(request, values);
        }

        let geometry = crate::storage_read::StorageShape {
            geometry: self.geometry,
        };
        collect_updates(request, values, || {
            geometry.plan_affine(request, &self.shape, self.offset, &self.strides)
        })
    }
}

fn collect_updates<T: Copy>(
    request: &BatchRequest,
    values: Vec<T>,
    plan: impl FnOnce() -> Result<crate::storage_read::LogicalGroups>,
) -> Result<BlockUpdates<T>> {
    if values.len() != request.len() {
        return Err(Error::InvalidLayout(
            "write request/value length mismatch".into(),
        ));
    }

    let mut updates = BlockUpdates::new();
    for (id, runs) in plan()? {
        let block = updates.entry(id).or_default();

        for run in runs {
            // The planner validates source positions and block offsets.
            for i in 0..run.len {
                let offset = (run.offset as i128 + i as i128 * run.stride as i128) as usize;
                block.push((offset, values[run.output + i]));
            }
        }
    }

    Ok(updates)
}

/// Validated, immutable logical and physical geometry. Physical IDs are not exposed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageGeometry {
    pub(crate) schema: TensorSchema,
    pub(crate) storage: StorageSchema,
}

impl StorageGeometry {
    pub fn new(schema: TensorSchema, layout: Layout, capacity: usize) -> Result<Self> {
        let storage = StorageSchema::new(schema.shape(), layout, capacity)?;
        Ok(Self { schema, storage })
    }

    pub fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    pub(crate) fn plan_base_updates<T: Copy>(
        &self,
        request: &BatchRequest,
        values: Vec<T>,
    ) -> Result<BlockUpdates<T>> {
        collect_updates(request, values, || {
            crate::storage_read::StorageShape { geometry: self }.plan(request, None)
        })
    }

    /// Plan a write in base coordinates. `None` selects the whole tensor.
    pub fn write_selection(&self, range: Option<crate::Range>) -> Result<WriteSelection<'_>> {
        let base = self.schema.shape();
        let range = range.unwrap_or_else(|| crate::validate::full_range(base));
        if range.iter().any(|axis| matches!(axis, AxisRange::Of(_))) {
            return Err(Error::Unsupported("gather selections are read-only".into()));
        }

        let (lengths, size) = crate::validate::validate_range(base, &range)?;
        let mut shape = Shape::with_capacity(base.len());
        let mut strides = Vec::with_capacity(base.len());
        let mut offset = 0;

        for ((axis, len), &stride) in range.into_iter().zip(lengths).zip(self.schema.strides()) {
            let start = match axis {
                AxisRange::At(at) => at,
                AxisRange::In(start, _, step) => {
                    shape.push(len);
                    // A singleton has no address advance. Its base stride also
                    // preserves the identity of whole-base singleton axes.
                    strides.push(i128::from(stride) * i128::from(if len > 1 { step } else { 1 }));
                    start
                }
                AxisRange::Of(_) => unreachable!("gather writes were rejected"),
            };

            if size > 0 {
                offset += i128::from(stride) * i128::from(start);
            }
        }

        Ok(WriteSelection {
            geometry: self,
            shape,
            strides,
            offset,
            size,
        })
    }

    pub fn layout(&self) -> Layout {
        self.storage.layout
    }

    pub fn block_capacity(&self) -> usize {
        self.storage.blocks.capacity
    }

    pub fn block_count(&self) -> u64 {
        self.storage.blocks.count()
    }

    pub fn block_range(&self, id: u64) -> Result<std::ops::Range<u64>> {
        self.storage.blocks.range(id)
    }

    pub fn block_len(&self, id: u64) -> Result<usize> {
        let range = self.block_range(id)?;
        Ok((range.end - range.start) as usize)
    }

    /// Validate a bounded, sorted, unique selection for a block source visitor.
    pub fn validate_block_ids(&self, ids: &[u64]) -> Result<()> {
        if ids.len() > crate::MAX_BATCH_ELEMENTS {
            return Err(Error::InvalidLayout(
                "too many logical blocks in read batch".into(),
            ));
        }

        let count = self.block_count();
        for (index, &id) in ids.iter().enumerate() {
            if id >= count {
                return Err(Error::InvalidCoord("logical block out of bounds".into()));
            }

            if index != 0 && ids[index - 1] >= id {
                return Err(Error::InvalidCoord(
                    "logical block IDs must be sorted and unique".into(),
                ));
            }
        }

        Ok(())
    }

    pub fn validate_id(&self, id: u64) -> Result<()> {
        self.block_range(id).map(|_| ())
    }

    pub fn block_position(&self, coord: &[u64]) -> Result<(u64, usize)> {
        crate::validate::validate_coord(self.schema.shape(), coord)?;
        let flat = coord
            .iter()
            .zip(self.schema.strides())
            .map(|(&c, &s)| c * s)
            .sum();

        let position = self.storage.blocks.position(flat);
        Ok((position.block_id, position.offset_in_block))
    }

    pub fn validate_block<T: TensorElement>(&self, id: u64, values: &[T]) -> Result<()> {
        if values.len() != self.block_len(id)? {
            return Err(Error::InvalidLayout("invalid logical block length".into()));
        }

        Ok(())
    }

    pub fn sparse_axis(&self) -> Option<usize> {
        match self.layout() {
            Layout::Dense => None,
            Layout::Sparse { axis } => Some(axis.unwrap_or(self.schema.shape().len() - 1)),
        }
    }

    /// Pack an ordered stream of logical sparse entries into destination blocks.
    /// Coordinates must be unique and in row-major order. Zero blocks are omitted.
    /// Source errors are returned unchanged.
    /// Pin streams that are not `Unpin` before passing them here.
    pub fn pack_sparse_blocks<'a, T: TensorElement, E: From<Error>>(
        &'a self,
        entries: impl Stream<Item = std::result::Result<(Vec<u64>, T), E>> + Send + Unpin + 'a,
    ) -> Result<impl Stream<Item = std::result::Result<(u64, Vec<T>), E>> + Send + 'a> {
        if self.layout() == Layout::Dense || self.schema().dtype() != T::dtype() {
            return Err(Error::InvalidSchema(
                "sparse block packing requires matching sparse geometry".into(),
            ));
        }

        Ok(try_unfold(
            (entries.fuse(), SparseBlockBuilder::default(), None),
            move |(mut entries, mut blocks, mut previous)| async move {
                let ready = loop {
                    let Some((coord, value)) = entries.try_next().await? else {
                        break blocks.take();
                    };

                    tokio::task::coop::consume_budget().await;
                    let address = self.block_position(&coord)?;
                    if previous.is_some_and(|prior| prior >= address) {
                        return Err(Error::InvalidCoord(
                            "sparse entries must be ordered and unique".into(),
                        )
                        .into());
                    }

                    previous = Some(address);
                    if let Some(block) = blocks.push(self, address, value) {
                        break Some(block);
                    }
                };

                Ok(ready.map(|block| (block, (entries, blocks, previous))))
            },
        ))
    }

    /// Stream payloads in destination block order. Sparse output omits zero blocks.
    /// The shape, dtype, and layout kind must match. Retains one bounded source
    /// read and one destination block between items.
    pub fn read_blocks<'a, S: TensorSource>(
        &'a self,
        source: &'a S,
    ) -> Result<impl Stream<Item = Result<(u64, Vec<S::DType>)>> + Send + 'a>
    where
        S::DType: TensorElement,
    {
        let geometry = source.storage_geometry();
        let sparse = self.sparse_axis().is_some();
        if self.schema() != geometry.schema()
            || self.schema().dtype() != S::DType::dtype()
            || sparse != geometry.sparse_axis().is_some()
        {
            return Err(Error::InvalidSchema(
                "block copy requires matching shapes, dtypes, and layout kinds".into(),
            ));
        }

        let blocks = 0..geometry.block_count();
        let keys = if sparse {
            source.occupied_blocks(blocks).left_stream()
        } else {
            futures::stream::iter(blocks.map(Ok)).right_stream()
        };

        Ok(try_unfold(
            (keys.peekable(), geometry, None, None, None),
            move |(mut keys, geometry, mut previous, mut input, mut output)| async move {
                let ready = loop {
                    tokio::task::coop::consume_budget().await;
                    let (range, start, values) = match input.take() {
                        Some(block) => block,
                        None => {
                            let Some((range, values)) =
                                read_block_span(source, geometry, &mut keys, &mut previous).await?
                            else {
                                break output.take();
                            };

                            let start = range.start;
                            (range, start, values)
                        }
                    };

                    let id = self.storage.blocks.position(start).block_id;
                    if output.as_ref().is_some_and(|(current, _)| *current != id) {
                        // The next input starts beyond the current destination block.
                        input = Some((range, start, values));
                        break output.take();
                    }

                    let target = self.block_range(id)?;
                    if range == target {
                        if !sparse || values.iter().any(|value| *value != S::DType::ZERO) {
                            break Some((id, values));
                        }

                        continue;
                    }

                    let end = range.end.min(target.end);
                    let fragment =
                        &values[(start - range.start) as usize..(end - range.start) as usize];
                    if !sparse || fragment.iter().any(|value| *value != S::DType::ZERO) {
                        let (_, block) = output.get_or_insert_with(|| {
                            (
                                id,
                                vec![S::DType::ZERO; (target.end - target.start) as usize],
                            )
                        });

                        let offset = (start - target.start) as usize;
                        block[offset..offset + fragment.len()].copy_from_slice(fragment);
                    }

                    if end < range.end {
                        input = Some((range, end, values));
                    }

                    if end == target.end && output.is_some() {
                        break output.take();
                    }
                };

                Ok(ready.map(|block| (block, (keys, geometry, previous, input, output))))
            },
        ))
    }
}

// Group adjacent blocks through the ordinary bounded native reader.
// Keep gaps out of the request and move a single block's buffer when possible.
async fn read_block_span<S: TensorSource>(
    source: &S,
    geometry: &StorageGeometry,
    keys: &mut Peekable<impl Stream<Item = Result<u64>> + Unpin>,
    previous: &mut Option<u64>,
) -> Result<Option<(Range<u64>, Vec<S::DType>)>>
where
    S::DType: TensorElement,
{
    let Some(first) = keys.try_next().await? else {
        return Ok(None);
    };

    if previous.is_some_and(|previous| previous >= first) {
        return Err(Error::InvalidCoord("invalid source block order".into()));
    }

    let mut range = geometry.block_range(first)?;
    let mut last = first;
    while let Some(&Ok(next)) = Pin::new(&mut *keys).peek().await {
        if last.checked_add(1) != Some(next) {
            break;
        }

        let end = geometry.block_range(next)?.end;
        if end - range.start > crate::MAX_BATCH_ELEMENTS as u64 {
            break;
        }

        keys.try_next().await?;
        range.end = end;
        last = next;
    }

    *previous = Some(last);
    let len = (range.end - range.start) as usize;
    let values = if first == last {
        source.read_logical_block(first).await?
    } else {
        let request = BatchRequest::linear(range.start, len)?;
        read_batch(source, &request, None).await?
    };

    if values.len() != len {
        return Err(Error::InvalidLayout(
            "invalid source block span length".into(),
        ));
    }

    Ok(Some((range, values)))
}

/// Assemble one validated batch from source-owned logical blocks.
pub(crate) async fn read_batch<S: TensorSource>(
    source: &S,
    request: &BatchRequest,
    mapping: Option<&CoordinateMap>,
) -> Result<Vec<S::DType>>
where
    S::DType: TensorElement,
{
    let geometry = source.storage_geometry();
    let groups = crate::storage_read::StorageShape { geometry }.plan(request, mapping)?;
    let ids: Vec<_> = groups.keys().copied().collect();
    let mut groups = groups.iter();
    let mut values = vec![S::DType::ZERO; request.len()];

    source
        .visit_blocks(&ids, |id, block| {
            let (_, runs) = groups
                .find(|(expected, _)| **expected >= id)
                .filter(|(expected, _)| **expected == id)
                .ok_or_else(|| {
                    Error::InvalidCoord("storage source returned an unexpected block ID".into())
                })?;

            geometry.validate_block(id, block)?;
            for run in runs {
                run.scatter(block, &mut values)?;
            }

            Ok(())
        })
        .await?;

    Ok(values)
}

/// A statically typed source for fensor's existing geometric and computed views.
/// A returned logical block has exactly `block_len(id)` values, with no padding. Sparse absence is represented by a zero block, never a missing file.
pub trait TensorSource: TensorArray + Clone + 'static
where
    Self::DType: TensorElement,
{
    fn storage_geometry(&self) -> &StorageGeometry;

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, Result<Vec<Self::DType>>>;

    /// Read one value at a checked logical block address.
    fn read_block_value(
        &self,
        id: u64,
        offset: usize,
    ) -> impl std::future::Future<Output = Result<Self::DType>> + Send {
        async move {
            let geometry = self.storage_geometry();
            if offset >= geometry.block_len(id)? {
                return Err(Error::InvalidCoord(
                    "offset out of bounds for logical block".into(),
                ));
            }

            let mut value = None;
            self.visit_blocks(&[id], |actual, block| {
                if actual != id || value.is_some() {
                    return Err(Error::InvalidCoord(
                        "storage source returned an unexpected block ID".into(),
                    ));
                }

                geometry.validate_block(id, block)?;
                value = Some(block[offset]);
                Ok(())
            })
            .await?;

            Ok(crate::expression::output_value(self.layout())(
                value.unwrap_or(Self::DType::ZERO),
            ))
        }
    }

    /// Own a scan of sorted unique occupied logical block IDs in `range`.
    /// Retain bounded lookahead until completion or drop and release index guards
    /// before delivery. Resume with `last + 1..range.end` after a delivered ID.
    /// Dense sources return an error when consumed.
    fn occupied_blocks(&self, range: std::ops::Range<u64>) -> BoxStream<'static, Result<u64>>;

    /// Visit a bounded, strictly increasing selection of logical blocks.
    /// Borrow each payload for the duration of the synchronous callback. A source
    /// may omit known-zero blocks. A missing required payload is a storage error.
    fn visit_blocks<'a, C>(&'a self, ids: &'a [u64], mut consume: C) -> BoxFuture<'a, Result<()>>
    where
        C: FnMut(u64, &[Self::DType]) -> Result<()> + Send + 'a,
    {
        Box::pin(async move {
            let geometry = self.storage_geometry();
            geometry.validate_block_ids(ids)?;
            for &id in ids {
                let block = self.read_logical_block(id).await?;
                geometry.validate_block(id, &block)?;
                consume(id, &block)?;
            }

            Ok(())
        })
    }
}

pub(crate) fn slice_requests<S: TensorSource>(
    source: &S,
    slice: crate::slice::Slice,
    mapping: Option<StorageSlice>,
) -> Result<crate::slice::Requests<'static>>
where
    S::DType: TensorElement,
{
    if slice.len() <= crate::expression::MAX_BATCH_ELEMENTS as u64
        || matches!(source.layout(), Layout::Dense)
    {
        return Ok(slice.stream());
    }

    let geometry = source.storage_geometry().clone();
    let mapping = mapping.unwrap_or_else(|| crate::mapping::StorageSlice::identity(source.shape()));
    // Bound the selection in block order using only rank-sized mapping metadata.
    // Inner-axis slices can include unrelated occupied blocks in this interval;
    // intersect their geometry before requesting any numerical payload.
    let mut first = mapping.origins.clone();
    let mut last = first.clone();
    for (axis, &(base, step)) in mapping.axes.iter().enumerate() {
        let selection = &slice.axes[axis];
        let (lo, hi) = match selection {
            crate::request::Axis::Span { .. } => {
                (selection.at(0), selection.at(selection.len() - 1))
            }
            crate::request::Axis::Selected(values) => (
                *values.as_slice().iter().min().expect("nonempty slice"),
                *values.as_slice().iter().max().expect("nonempty slice"),
            ),
        };

        let mapped = |value: u64| {
            value
                .checked_mul(step)
                .and_then(|offset| mapping.origins[base].checked_add(offset))
                .ok_or_else(|| Error::InvalidCoord("sparse selection bound overflow".into()))
        };

        first[base] = mapped(lo)?;
        last[base] = mapped(hi)?;
    }

    let first = geometry.block_position(&first)?.0;
    let end = geometry.block_position(&last)?.0 + 1;

    struct Cursor {
        keys: Option<BoxStream<'static, Result<u64>>>,
        pending: std::collections::VecDeque<crate::slice::SliceRequests>,
        pending_len: u64,
        // At most one bounded rectangle which did not fit the current batch.
        overflow: Option<crate::request::Cartesian>,
    }

    let cursor = Cursor {
        keys: Some(source.occupied_blocks(first..end)),
        pending: std::collections::VecDeque::new(),
        pending_len: 0,
        overflow: None,
    };

    Ok(futures::stream::try_unfold(
        (cursor, slice, mapping, geometry),
        move |(mut cursor, slice, mapping, geometry)| async move {
            let mut rectangles = Vec::new();
            let mut len = 0;

            loop {
                if len != 0
                    && cursor.pending_len <= crate::MAX_BATCH_ELEMENTS as u64
                    && cursor.pending_len > (crate::MAX_BATCH_ELEMENTS - len) as u64
                {
                    break;
                }

                let rectangle = cursor.overflow.take().or_else(|| {
                    while let Some(requests) = cursor.pending.front_mut() {
                        if let Some(rectangle) = requests.next_rectangle() {
                            cursor.pending_len -= rectangle.len() as u64;
                            return Some(rectangle);
                        }

                        cursor.pending.pop_front();
                    }

                    None
                });

                if let Some(rectangle) = rectangle {
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

                let Some(keys) = &mut cursor.keys else {
                    break;
                };

                // Rejected candidates must cooperate even when the source is always ready.
                // Occupied streams release native guards before delivering IDs.
                tokio::task::coop::consume_budget().await;
                let Some(id) = keys.try_next().await? else {
                    cursor.keys = None;
                    break;
                };

                for bounds in
                    crate::schema::flat_rectangles(geometry.schema(), geometry.block_range(id)?)?
                {
                    if let Some(bounds) = mapping.bounds(&bounds) {
                        let selected = slice.intersect(&bounds)?;
                        cursor.pending_len += selected.len();
                        cursor.pending.push_back(selected.requests());
                    }
                }
            }

            if rectangles.is_empty() {
                Ok(None)
            } else {
                Ok(Some((
                    BatchRequest::rectangles(rectangles)?,
                    (cursor, slice, mapping, geometry),
                )))
            }
        },
    )
    .boxed())
}

/// Use occupied blocks when they preserve logical row-major candidate order.
pub(crate) fn ordered_requests<S: TensorSource>(
    source: &S,
    slice: crate::slice::Slice,
    mapping: Option<StorageSlice>,
) -> Result<crate::slice::Requests<'static>>
where
    S::DType: TensorElement,
{
    if slice.len() <= crate::MAX_BATCH_ELEMENTS as u64 || matches!(source.layout(), Layout::Dense) {
        return Ok(slice.stream());
    }

    let mapping = mapping.unwrap_or_else(|| StorageSlice::identity(source.shape()));
    let mut previous = None;
    for (selection, &(base, _)) in slice.axes.iter().zip(&mapping.axes) {
        if selection.len() <= 1 {
            continue;
        }

        if previous.is_some_and(|previous| previous >= base) {
            return Ok(slice.stream());
        }

        previous = Some(base);
    }

    slice_requests(source, slice, Some(mapping))
}

/// Full-coverage physical traversal for consumers which retain coordinates.
/// Sparse occupancy is deliberately irrelevant: implicit zeros are values too.
pub(crate) fn preferred_requests<S: TensorSource>(
    source: &S,
    shape: &[u64],
    mapping: Option<StorageSlice>,
) -> Result<Option<crate::expression::RequestIterator>>
where
    S::DType: TensorElement,
{
    let geometry = source.storage_geometry();
    let mapping = mapping.unwrap_or_else(|| StorageSlice::identity(source.shape()));
    if shape == source.shape()
        && mapping.origins.iter().all(|&n| n == 0)
        && mapping
            .axes
            .iter()
            .enumerate()
            .all(|(i, &(axis, step))| shape[i] == 1 || (axis == i && step == 1))
    {
        return Ok(None);
    }

    let slice = crate::slice::Slice::full(shape)?;
    let mut selected: Vec<_> = mapping.origins.iter().map(|&start| (start, 1, 1)).collect();
    for (axis, &(base, step)) in mapping.axes.iter().enumerate() {
        let last = (shape[axis] - 1)
            .checked_mul(step)
            .and_then(|n| mapping.origins[base].checked_add(n))
            .ok_or_else(|| Error::InvalidCoord("storage slice bound overflow".into()))?;

        if last >= source.shape()[base] {
            return Err(Error::InvalidCoord("storage slice out of bounds".into()));
        }

        selected[base] = (mapping.origins[base], step, shape[axis]);
    }

    let geometry = geometry.clone();
    let mut next = 0;
    let mut coord = crate::schema::Coord::new();
    let mut pending = None;
    let requests = std::iter::from_fn(move || {
        let mut rectangles = Vec::new();
        let mut len = 0;
        loop {
            let item = pending.take().or_else(|| {
                loop {
                    let size = geometry.storage.blocks.size;
                    if next >= size {
                        return None;
                    }

                    crate::request::decode_flat(next, geometry.schema().shape(), &mut coord)
                        .expect("validated flat position");

                    let mut axis = 0;
                    while axis < coord.len() {
                        let (start, step, count) = selected[axis];
                        let index = coord[axis].saturating_sub(start).div_ceil(step);
                        if index < count {
                            let value = start + index * step;
                            if value > coord[axis] {
                                coord[axis] = value;
                                for j in axis + 1..coord.len() {
                                    coord[j] = selected[j].0;
                                }

                                break;
                            }

                            coord[axis] = value;
                            axis += 1;
                        } else {
                            let mut carry = axis;
                            loop {
                                if carry == 0 {
                                    next = size;
                                    return None;
                                }

                                carry -= 1;
                                let (lo, stride, n) = selected[carry];
                                let index = (coord[carry] - lo) / stride + 1;
                                if index < n {
                                    coord[carry] = lo + index * stride;
                                    for j in carry + 1..coord.len() {
                                        coord[j] = selected[j].0;
                                    }

                                    break;
                                }
                            }

                            break;
                        }
                    }

                    let id = geometry
                        .block_position(&coord)
                        .expect("validated selected coordinates")
                        .0;

                    let range = geometry.block_range(id).expect("validated block");
                    next = range.end;
                    let mut block = Vec::new();
                    let mut count = 0;
                    for bounds in crate::schema::flat_rectangles(geometry.schema(), range)
                        .expect("validated block span")
                    {
                        if let Some(bounds) = mapping.bounds(&bounds) {
                            let mut requests = slice
                                .intersect(&bounds)
                                .expect("validated intersection")
                                .requests();

                            while let Some(rectangle) = requests.next_rectangle() {
                                count += rectangle.len();
                                block.push(rectangle);
                            }
                        }
                    }

                    if count != 0 {
                        return Some((block, count));
                    }
                }
            });

            let Some((block, count)) = item else {
                break;
            };

            if len + count > crate::MAX_BATCH_ELEMENTS {
                pending = Some((block, count));
                break;
            }

            len += count;
            rectangles.extend(block);
            if len == crate::MAX_BATCH_ELEMENTS {
                break;
            }
        }

        (!rectangles.is_empty())
            .then(|| BatchRequest::rectangles(rectangles).expect("bounded block requests"))
    });

    Ok(Some(Box::new(requests)))
}

#[cfg(test)]
#[path = "../tests/unit/storage/traversal_tests.rs"]
mod traversal_tests;
