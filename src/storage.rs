//! Native storage geometry and bounded sources for geometric expression leaves.
//!
//! Implementations own their visibility and mutation policy. fensor only maps and
//! evaluates bounded requests; it does not acquire transaction reservations.

use futures::{StreamExt, TryStreamExt, stream::BoxStream};

use crate::mapping::{CoordinateMap, StorageSlice};
use crate::request::BatchRequest;
use crate::schema::{StorageSchema, TensorSchema};
use crate::{BoxFuture, Error, Layout, Result, Shape, TensorArray, TensorElement};

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

    let shape = crate::storage_read::StorageShape {
        shape: geometry.schema.shape(),
        strides: geometry.schema.strides(),
        block_shape: geometry.block_shape(),
        block_strides: &geometry.storage.block_schema.strides,
        grid_strides: &geometry.storage.strides,
    };

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
#[derive(Clone)]
pub struct StorageGeometry {
    pub(crate) schema: TensorSchema,
    pub(crate) storage: StorageSchema,
}

// The cursor reuses rank-sized scratch to traverse valid block offsets.
struct BlockCursor<'a> {
    geometry: &'a StorageGeometry,
    bounds: Vec<(u64, u64)>,
    coord: crate::schema::Coord,
    offset: usize,
    done: bool,
}

impl<'a> BlockCursor<'a> {
    #[inline]
    fn new(geometry: &'a StorageGeometry, id: u64) -> Result<Self> {
        let bounds = geometry.block_bounds(id)?;
        Ok(Self::from_bounds(geometry, bounds))
    }

    fn from_bounds(geometry: &'a StorageGeometry, bounds: Vec<(u64, u64)>) -> Self {
        let coord = bounds.iter().map(|(lo, _)| *lo).collect();
        Self {
            geometry,
            bounds,
            coord,
            offset: 0,
            done: false,
        }
    }

    fn offsets(mut self) -> impl Iterator<Item = usize> + 'a {
        std::iter::from_fn(move || {
            let offset = self.current()?.0;
            self.advance();
            Some(offset)
        })
    }

    #[inline]
    fn current(&self) -> Option<(usize, &[u64])> {
        (!self.done).then_some((self.offset, &self.coord))
    }

    // Advance after consumption, avoiding a first-call flag in per-element loops.
    #[inline(always)]
    fn advance(&mut self) {
        let mut axis = self.coord.len();

        loop {
            if axis == 0 {
                self.done = true;
                return;
            }
            axis -= 1;
            self.coord[axis] += 1;
            let stride = self.geometry.storage.block_schema.strides[axis] as usize;
            self.offset += stride;
            if self.coord[axis] < self.bounds[axis].1 {
                break;
            }

            self.offset -= (self.bounds[axis].1 - self.bounds[axis].0) as usize * stride;
            self.coord[axis] = self.bounds[axis].0;
        }
    }
}

impl StorageGeometry {
    pub fn new(schema: TensorSchema, layout: Layout, block_shape: Shape) -> Result<Self> {
        // Metadata owns block rank/capacity validation independently of dtype.
        if block_shape.len() != schema.shape().len()
            || crate::schema::checked_product(&block_shape)?
                > crate::schema::MAX_BLOCK_CAPACITY as u64
        {
            return Err(Error::InvalidLayout(
                "invalid storage block geometry".into(),
            ));
        }

        let storage = StorageSchema::from_block_shape(schema.shape(), layout, block_shape)?;
        Ok(Self { schema, storage })
    }

    pub fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    pub fn layout(&self) -> Layout {
        self.storage.layout
    }

    pub fn block_shape(&self) -> &[u64] {
        &self.storage.block_schema.shape
    }

    pub fn block_len(&self) -> usize {
        self.storage.block_schema.shape.iter().product::<u64>() as usize
    }

    pub fn block_count(&self) -> u64 {
        self.storage.shape.iter().product()
    }

    pub fn block_position(&self, coord: &[u64]) -> Result<(u64, usize)> {
        crate::validate::validate_coord(self.schema.shape(), coord)?;
        let position = crate::storage_read::block_position(
            coord.iter().copied(),
            self.block_shape(),
            &self.storage.block_schema.strides,
            &self.storage.strides,
        );
        Ok((position.block_id, position.offset_in_block))
    }

    /// Validate a logical block ID without allocating bounds or accessing storage.
    pub fn validate_id(&self, id: u64) -> Result<()> {
        if id >= self.block_count() {
            return Err(Error::InvalidCoord("logical block out of bounds".into()));
        }

        Ok(())
    }

    pub fn block_bounds(&self, id: u64) -> Result<Vec<(u64, u64)>> {
        self.validate_id(id)?;
        #[cfg(test)]
        crate::read_metrics::record(|m| m.block_bounds += 1);

        Ok((0..self.schema.shape().len())
            .map(|axis| {
                let grid = (id / self.storage.strides[axis]) % self.storage.shape[axis];
                let start = grid * self.block_shape()[axis];
                (
                    start,
                    start + self.block_shape()[axis].min(self.schema.shape()[axis] - start),
                )
            })
            .collect())
    }

    /// Validate a complete logical block, including zero-valued edge padding.
    /// Check the block ID before its length; valid positions may contain any value.
    pub fn validate_block<T: TensorElement>(&self, id: u64, values: &[T]) -> Result<()> {
        self.validate_id(id)?;
        if values.len() != self.block_len() {
            return Err(Error::InvalidLayout("invalid logical block length".into()));
        }

        // The quotient counts complete blocks without computing an overflowing end.
        if (0..self.schema.shape().len()).all(|axis| {
            let grid = (id / self.storage.strides[axis]) % self.storage.shape[axis];
            grid < self.schema.shape()[axis] / self.block_shape()[axis]
        }) {
            return Ok(());
        }

        let bounds = self.block_bounds(id)?;
        #[cfg(test)]
        crate::read_metrics::record(|m| m.padding_walks += 1);
        let mut next = 0;
        for offset in BlockCursor::from_bounds(self, bounds).offsets() {
            if values[next..offset].iter().any(|v| *v != T::ZERO) {
                return Err(Error::InvalidLayout("nonzero logical block padding".into()));
            }
            next = offset + 1;
        }

        if values[next..].iter().any(|v| *v != T::ZERO) {
            return Err(Error::InvalidLayout("nonzero logical block padding".into()));
        }

        Ok(())
    }

    /// Valid physical-buffer offsets in logical order, excluding edge padding.
    pub fn block_offsets(&self, id: u64) -> Result<impl Iterator<Item = usize> + '_> {
        Ok(BlockCursor::new(self, id)?.offsets())
    }

    pub fn sparse_axis(&self) -> Option<usize> {
        match self.layout() {
            Layout::Dense => None,
            Layout::Sparse { axis } => Some(axis.unwrap_or(0)),
        }
    }
}

/// A validated bounded read request produced by fensor's geometric mapper.
/// Sources normally use the default reader; native storage can preserve its
/// borrowed-block fast path without exposing physical IDs to callers.
pub struct StorageRead<'a> {
    pub(crate) request: &'a BatchRequest,
    pub(crate) mapping: Option<&'a CoordinateMap>,
}

impl StorageRead<'_> {
    /// Validate and group this bounded request without accessing storage.
    pub fn plan(&self, geometry: &StorageGeometry) -> Result<StorageReadPlan> {
        let shape = crate::storage_read::StorageShape {
            shape: geometry.schema.shape(),
            strides: geometry.schema.strides(),
            block_shape: geometry.block_shape(),
            block_strides: &geometry.storage.block_schema.strides,
            grid_strides: &geometry.storage.strides,
        };

        let groups = shape.plan(self.request, self.mapping)?;

        Ok(StorageReadPlan {
            groups,
            len: self.request.len(),
            block_len: geometry.block_len(),
        })
    }
}

/// A bounded logical-block selection with validated scatter positions.
/// Callers select visible storage; the plan preserves request order and duplicates.
pub struct StorageReadPlan {
    groups: crate::storage_read::LogicalGroups,
    len: usize,
    block_len: usize,
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
        if block.len() != self.block_len {
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
/// A returned logical block includes edge padding and has exactly `block_len`
/// values. Sparse absence is represented by a zero block, never a missing file.
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
                *values.iter().min().expect("nonempty slice"),
                *values.iter().max().expect("nonempty slice"),
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
        pending: Option<crate::slice::SliceRequests>,
        // At most one bounded rectangle which did not fit the current batch.
        overflow: Option<crate::request::Cartesian>,
    }

    let cursor = Cursor {
        keys: Some(source.occupied_blocks(first..end)),
        pending: None,
        overflow: None,
    };

    Ok(futures::stream::try_unfold(
        (cursor, slice, mapping, geometry),
        move |(mut cursor, slice, mapping, geometry)| async move {
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

                let Some(keys) = &mut cursor.keys else {
                    break;
                };

                let Some(id) = keys.try_next().await? else {
                    cursor.keys = None;
                    break;
                };

                let bounds = geometry.block_bounds(id)?;
                if let Some(bounds) = mapping.bounds(&bounds) {
                    cursor.pending = Some(slice.intersect(&bounds)?.requests());
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
    let geometry = source.storage_geometry();
    let mut previous = None;
    let mut spans = false;

    // Fixed and singleton axes do not constrain traversal order. Once a block
    // spans an active axis, every later active axis must stay within one block.
    for (selection, &(base, step)) in slice.axes.iter().zip(&mapping.axes) {
        if selection.len() <= 1 {
            continue;
        }

        let (first, last) = match selection {
            crate::request::Axis::Span { .. } => {
                (selection.at(0), selection.at(selection.len() - 1))
            }
            crate::request::Axis::Selected(values) => (
                *values.iter().min().expect("nonempty selection"),
                *values.iter().max().expect("nonempty selection"),
            ),
        };

        let first = first
            .checked_mul(step)
            .and_then(|n| mapping.origins[base].checked_add(n))
            .ok_or_else(|| Error::InvalidCoord("sparse selection bound overflow".into()))?;
        let last = last
            .checked_mul(step)
            .and_then(|n| mapping.origins[base].checked_add(n))
            .ok_or_else(|| Error::InvalidCoord("sparse selection bound overflow".into()))?;
        let block = geometry.block_shape()[base];

        if previous.is_some_and(|previous| previous >= base)
            || (spans && first / block != last / block)
        {
            return Err(Error::Unsupported(
                "sparse block order does not preserve this selection's row-major order; use copy_from_values to materialize this layout".into()
            ));
        }
        previous = Some(base);
        let stride = match selection {
            crate::request::Axis::Span {
                step: selected_step,
                ..
            } => step.checked_mul(*selected_step),
            crate::request::Axis::Selected(_) => Some(step),
        };
        spans |= stride.is_some_and(|stride| stride < block);
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
    let identity = shape == source.shape()
        && mapping.as_ref().is_none_or(|mapping| {
            mapping.origins.iter().all(|&origin| origin == 0)
                && mapping
                    .axes
                    .iter()
                    .enumerate()
                    .all(|(axis, &(base, step))| shape[axis] == 1 || (base == axis && step == 1))
        });
    if identity {
        let mut spans = false;
        let logical_order =
            geometry
                .block_shape()
                .iter()
                .zip(source.shape())
                .all(|(&block, &dim)| {
                    let ordered = !spans || block >= dim;
                    spans |= block > 1;
                    ordered
                });
        if logical_order {
            return Ok(None);
        }
    }

    let mapping = mapping.unwrap_or_else(|| StorageSlice::identity(source.shape()));
    let slice = crate::slice::Slice::full(shape)?;
    let mut last = mapping.origins.clone();
    let mut axes = vec![None; source.shape().len()];
    for (axis, (&dim, &(base, step))) in shape.iter().zip(&mapping.axes).enumerate() {
        last[base] = (dim - 1)
            .checked_mul(step)
            .and_then(|n| mapping.origins[base].checked_add(n))
            .ok_or_else(|| Error::InvalidCoord("storage slice bound overflow".into()))?;
        axes[base] = Some((axis, step));
    }
    geometry.block_position(&mapping.origins)?;
    geometry.block_position(&last)?;
    let fastest = axes
        .iter()
        .rposition(|axis| axis.is_some_and(|(axis, _)| shape[axis] > 1));
    let mut coordinates = mapping.origins.clone();
    let mut done = false;

    let regions = std::iter::from_fn(move || {
        if done {
            return None;
        }

        let id = geometry
            .block_position(&coordinates)
            .expect("validated storage slice")
            .0;
        let mut bounds = geometry.block_bounds(id).expect("validated storage block");
        let mut consumer_bounds = mapping.bounds(&bounds).expect("selected storage region");
        if let Some(base) = fastest {
            let (axis, _) = axes[base].expect("varying storage axis");
            // Concatenating adjacent physical regions preserves their sequence
            // when all preceding consumer axes are singleton within the region.
            // Faster storage axes are fixed; bounded batching still owns splitting.
            let consecutive = consumer_bounds[..axis]
                .iter()
                .zip(&slice.axes)
                .all(|(&(lo, hi), selection)| hi.min(selection.len()).saturating_sub(lo) <= 1);
            if consecutive {
                bounds[base].1 = geometry.schema.shape()[base];
                consumer_bounds[axis].1 = slice.axes[axis].len();
            }
        }

        let requests = slice
            .intersect(&consumer_bounds)
            .expect("validated region intersection")
            .requests();
        done = true;

        for base in (0..coordinates.len()).rev() {
            if let Some((axis, step)) = axes[base] {
                // Jump over empty block regions, rather than advancing each block
                // between widely separated selected coordinates.
                let next = bounds[base]
                    .1
                    .saturating_sub(mapping.origins[base])
                    .div_ceil(step);
                if next < slice.axes[axis].len() {
                    coordinates[base] = mapping.origins[base] + next * step;
                    done = false;
                    break;
                }
            }
            coordinates[base] = mapping.origins[base];
        }
        Some(requests)
    });
    let mut rectangles =
        regions.flat_map(|mut requests| std::iter::from_fn(move || requests.next_rectangle()));
    let mut overflow = None;
    let requests = std::iter::from_fn(move || {
        let mut output = Vec::new();
        let mut len = 0;

        while let Some(rectangle) = overflow.take().or_else(|| rectangles.next()) {
            if rectangle.len() > crate::MAX_BATCH_ELEMENTS - len {
                overflow = Some(rectangle);
                break;
            }
            len += rectangle.len();
            output.push(rectangle);
            if len == crate::MAX_BATCH_ELEMENTS {
                break;
            }
        }

        (!output.is_empty())
            .then(|| BatchRequest::rectangles(output).expect("bounded storage regions"))
    });
    Ok(Some(Box::new(requests)))
}

#[cfg(test)]
#[path = "../tests/unit/storage/traversal_tests.rs"]
mod traversal_tests;
