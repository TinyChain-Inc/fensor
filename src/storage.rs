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
        let coord = bounds.iter().map(|(lo, _)| *lo).collect();
        Ok(Self {
            geometry,
            bounds,
            coord,
            offset: 0,
            done: false,
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

    pub fn block_bounds(&self, id: u64) -> Result<Vec<(u64, u64)>> {
        if id >= self.block_count() {
            return Err(Error::InvalidCoord("logical block out of bounds".into()));
        }

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

    /// Valid physical-buffer offsets in logical order, excluding edge padding.
    pub fn block_offsets(&self, id: u64) -> Result<impl Iterator<Item = usize> + '_> {
        let mut cursor = BlockCursor::new(self, id)?;
        Ok(std::iter::from_fn(move || {
            let offset = cursor.current()?.0;
            cursor.advance();
            Some(offset)
        }))
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

/// A statically typed source for fensor's existing geometric and computed views.
/// A returned logical block includes edge padding and has exactly `block_len`
/// values. Sparse absence is represented by a zero block, never a missing file.
pub trait TensorSource: TensorArray + Clone + 'static
where
    Self::DType: TensorElement,
{
    fn storage_geometry(&self) -> StorageGeometry;

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, Result<Vec<Self::DType>>>;

    /// Own a scan of sorted unique `(sparse-axis coordinate, logical grid block)`
    /// keys strictly after `after`, within inclusive axis bounds. Retain bounded
    /// lookahead until completion or drop, releasing index guards before delivery.
    /// Dense sources return an error when consumed.
    fn occupied_regions(
        &self,
        after: Option<[u64; 2]>,
        lo: u64,
        hi: u64,
    ) -> BoxStream<'static, Result<[u64; 2]>>;

    fn read_storage<'a>(
        &'a self,
        read: StorageRead<'a>,
    ) -> BoxFuture<'a, Result<Vec<Self::DType>>> {
        Box::pin(async move {
            let geometry = self.storage_geometry();
            let shape = crate::storage_read::StorageShape {
                shape: self.shape(),
                strides: self.strides(),
                block_shape: geometry.block_shape(),
                block_strides: &geometry.storage.block_schema.strides,
                grid_strides: &geometry.storage.strides,
                // Logical blocks already combine every sparse key in their region.
            };

            let groups = shape.plan(read.request, read.mapping)?;
            let mut values = vec![Self::DType::default(); read.request.len()];

            for (id, runs) in groups {
                let block = self.read_logical_block(id).await?;
                if block.len() != geometry.block_len() {
                    return Err(Error::InvalidLayout("invalid logical block length".into()));
                }

                for run in runs {
                    run.scatter(&block, &mut values)?;
                }
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
    let source = source.clone();
    let geometry = source.storage_geometry();
    if slice.len() <= crate::expression::MAX_BATCH_ELEMENTS as u64
        || matches!(source.layout(), Layout::Dense)
    {
        return Ok(slice.stream());
    }

    let mapping = mapping.unwrap_or_else(|| crate::mapping::StorageSlice::identity(source.shape()));
    let sparse_axis = geometry
        .sparse_axis()
        .ok_or_else(|| Error::InvalidLayout("sparse traversal on dense storage".into()))?;
    let (lo, hi) = if let Some((axis, &(_, step))) = mapping
        .axes
        .iter()
        .enumerate()
        .find(|(_, (base, _))| *base == sparse_axis)
    {
        let selection = &slice.axes[axis];
        let (first, last) = match selection {
            crate::request::Axis::Span { .. } => {
                (selection.at(0), selection.at(selection.len() - 1))
            }
            crate::request::Axis::Selected(values) => (
                *values.iter().min().expect("nonempty slice"),
                *values.iter().max().expect("nonempty slice"),
            ),
        };
        (
            mapping.origins[sparse_axis] + first * step,
            mapping.origins[sparse_axis] + last * step,
        )
    } else {
        (mapping.origins[sparse_axis], mapping.origins[sparse_axis])
    };

    struct Cursor {
        keys: Option<BoxStream<'static, Result<[u64; 2]>>>,
        pending: Option<crate::slice::SliceRequests>,
        // At most one bounded rectangle which did not fit the current batch.
        overflow: Option<crate::request::Cartesian>,
    }

    let cursor = Cursor {
        keys: Some(source.occupied_regions(None, lo, hi)),
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

                let Some(row) = keys.try_next().await? else {
                    cursor.keys = None;
                    break;
                };

                let mut bounds = geometry.block_bounds(row[1])?;

                let coord = row[0];
                if !(bounds[sparse_axis].0..bounds[sparse_axis].1).contains(&coord) {
                    return Err(Error::InvalidLayout(
                        "sparse slice key disagrees with its grid block".into(),
                    ));
                }
                bounds[sparse_axis] = (coord, coord + 1);
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

/// Use indexed requests only when their region order is already logical order.
/// Other geometry uses bounded logical evaluation, without sorting or persistence.
pub(crate) fn ordered_requests<S: TensorSource>(
    source: &S,
    slice: crate::slice::Slice,
    mapping: StorageSlice,
) -> Result<crate::slice::Requests<'static>>
where
    S::DType: TensorElement,
{
    let geometry = source.storage_geometry();
    let Some(sparse_axis) = geometry.sparse_axis() else {
        return Ok(slice.stream());
    };

    if source.shape()[..sparse_axis].iter().any(|&dim| dim != 1)
        || mapping.axes.windows(2).any(|axes| axes[0].0 >= axes[1].0)
    {
        return Ok(slice.stream());
    }
    // A region must be a contiguous row-major run. Earlier axes are singleton;
    // after the first multi-element axis all block extents cover the full axis.
    let mut spans = false;

    for (axis, (&block, &dim)) in geometry
        .block_shape()
        .iter()
        .zip(source.shape())
        .enumerate()
    {
        if axis == sparse_axis {
            continue;
        }

        if spans && block < dim {
            return Ok(slice.stream());
        }
        spans |= block > 1;
    }
    slice_requests(source, slice, Some(mapping))
}

#[cfg(test)]
mod traversal_tests {
    use number_general::DType;

    use super::*;

    #[test]
    fn block_consumers_share_order_padding_and_rank_scratch() {
        assert!(TensorSchema::new(f64::dtype(), Shape::new()).is_err());
        let mut high = Shape::from_elem(1, crate::PORTABLE_INLINE_RANK + 2);
        high[0] = 3;
        let mut high_block = high.clone();
        high_block[0] = 2;

        for (shape, block) in [
            (smallvec::smallvec![1], smallvec::smallvec![1]),
            (smallvec::smallvec![5], smallvec::smallvec![3]),
            (smallvec::smallvec![3, 5], smallvec::smallvec![2, 3]),
            (high, high_block),
        ] {
            let layouts = std::iter::once(Layout::Dense)
                .chain((0..shape.len()).map(|axis| Layout::Sparse { axis: Some(axis) }));
            for layout in layouts {
                let geometry = StorageGeometry::new(
                    TensorSchema::new(f64::dtype(), shape.clone()).unwrap(),
                    layout,
                    block.clone(),
                )
                .unwrap();

                for id in 0..geometry.block_count() {
                    let bounds = geometry.block_bounds(id).unwrap();
                    let expected: Vec<_> = crate::schema::row_major_coords(&shape)
                        .unwrap()
                        .filter(|coord| {
                            coord
                                .iter()
                                .zip(&bounds)
                                .all(|(c, (lo, hi))| c >= lo && c < hi)
                        })
                        .map(|coord| {
                            let offset = coord
                                .iter()
                                .zip(&bounds)
                                .zip(&geometry.storage.block_schema.strides)
                                .map(|((c, (lo, _)), s)| (c - lo) * s)
                                .sum::<u64>() as usize;
                            (offset, coord)
                        })
                        .collect();
                    assert_eq!(
                        geometry.block_offsets(id).unwrap().collect::<Vec<_>>(),
                        expected.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                    );
                    let mut cursor = BlockCursor::new(&geometry, id).unwrap();
                    let address = cursor.coord.as_ptr();

                    for (offset, coord) in &expected {
                        let (actual, scratch) = cursor.current().unwrap();
                        assert_eq!(actual, *offset);
                        assert_eq!(scratch, coord);
                        assert_eq!(scratch.as_ptr(), address);
                        cursor.advance();
                    }
                    assert!(cursor.current().is_none());
                    assert!(cursor.current().is_none());
                }
                assert!(geometry.block_offsets(geometry.block_count()).is_err());
                assert!(geometry.block_offsets(u64::MAX).is_err());
            }
        }
    }
}
