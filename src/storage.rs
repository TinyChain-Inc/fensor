//! Native storage geometry and bounded sources for geometric expression leaves.
//!
//! Implementations own their visibility and mutation policy. fensor only maps and
//! evaluates bounded requests; it does not acquire transaction reservations.

use futures::{StreamExt, TryStreamExt, stream::BoxStream};

use crate::mapping::{CoordinateMap, StorageSlice};
use crate::request::BatchRequest;
use crate::schema::{StorageSchema, TensorSchema};
use crate::{BoxFuture, Error, Layout, Result, TensorArray, TensorElement};

/// Logical-block updates from one bounded batch, in input order within each block.
/// Coordinates are block-local offsets, never physical payload identifiers.
pub type BlockUpdates<T> = std::collections::BTreeMap<u64, Vec<(usize, T)>>;

pub(crate) fn plan_updates<T: Copy>(
    geometry: &StorageGeometry,
    mapping: Option<&CoordinateMap>,
    request: &BatchRequest,
    values: Vec<T>,
) -> Result<BlockUpdates<T>> {
    if values.len() != request.len() {
        return Err(Error::InvalidLayout(
            "write request/value length mismatch".into(),
        ));
    }

    let shape = crate::storage_read::StorageShape { geometry };
    let groups = shape.plan(request, mapping)?;
    let mut updates = BlockUpdates::new();

    for (id, runs) in groups {
        let block = updates.entry(id).or_default();

        for run in runs {
            // The storage planner validates both source positions and block offsets.
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

    pub(crate) fn validate_block_ids(&self, ids: &[u64]) -> Result<()> {
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
}

/// A validated bounded read request produced by the geometric mapper.
pub struct StorageRead<'a> {
    pub(crate) request: &'a BatchRequest,
    pub(crate) mapping: Option<&'a CoordinateMap>,
}

impl StorageRead<'_> {
    pub fn plan(&self, geometry: &StorageGeometry) -> Result<StorageReadPlan> {
        let groups =
            crate::storage_read::StorageShape { geometry }.plan(self.request, self.mapping)?;

        Ok(StorageReadPlan {
            groups,
            len: self.request.len(),
            blocks: geometry.storage.blocks.clone(),
        })
    }
}

/// A bounded logical-block selection with validated scatter positions.
/// Callers select visible storage; the plan preserves request order and duplicates.
pub struct StorageReadPlan {
    groups: crate::storage_read::LogicalGroups,
    len: usize,
    blocks: crate::schema::LinearBlocks,
}

impl StorageReadPlan {
    /// Number of requested numerical values, including repeated selections.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Requested logical block IDs in strictly increasing order.
    pub fn block_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.groups.keys().copied()
    }

    /// Copy one complete logical block into its requested output positions.
    /// Unrequested IDs and incorrect buffer lengths are rejected before scattering.
    pub fn scatter<T: Copy>(&self, id: u64, block: &[T], output: &mut [T]) -> Result<()> {
        let range = self.blocks.range(id)?;
        if block.len() as u64 != range.end - range.start {
            return Err(Error::InvalidLayout("invalid logical block length".into()));
        }

        if output.len() != self.len {
            return Err(Error::InvalidLayout(
                "invalid storage read output length".into(),
            ));
        }

        let runs = self
            .groups
            .get(&id)
            .ok_or_else(|| Error::InvalidCoord("logical block is not in the read plan".into()))?;

        for run in runs {
            run.scatter(block, output)?;
        }

        Ok(())
    }
}

/// A statically typed source for fensor's existing geometric and computed views.
/// A returned logical block has exactly `block_len(id)` values, with no padding. Sparse absence is represented by a zero block, never a missing file.
pub trait TensorSource: TensorArray + Clone + 'static
where
    Self::DType: TensorElement,
{
    fn storage_geometry(&self) -> StorageGeometry;

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, Result<Vec<Self::DType>>>;

    /// Own a scan of sorted unique occupied logical block IDs in `range`.
    /// Retain bounded lookahead until completion or drop and release index guards
    /// before delivery. Resume with `last + 1..range.end` after a delivered ID.
    /// Dense sources return an error when consumed.
    fn occupied_blocks(&self, range: std::ops::Range<u64>) -> BoxStream<'static, Result<u64>>;

    /// Read numerical values through one bounded storage plan, filling sparse absence with zeros.
    fn read_storage<'a>(
        &'a self,
        read: StorageRead<'a>,
    ) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move {
            let plan = read.plan(&self.storage_geometry())?;
            let mut values = vec![Self::DType::default(); plan.len()];

            for id in plan.block_ids() {
                let block = self.read_logical_block(id).await?;
                plan.scatter(id, &block, &mut values)?;
            }

            Ok(values)
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

    let geometry = source.storage_geometry();
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

/// Preserve row-major candidate order without sorting or scanning implicit zeros.
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
            return Err(Error::Unsupported("sparse block order does not preserve this selection's row-major order; use copy_from_values to materialize this layout".into()));
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
