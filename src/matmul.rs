//! Bounded matrix tiles over filesystem-backed expressions.

use std::collections::BTreeMap;

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{Axes, MatrixDual, NDArrayRead, NDArrayTransform, Number, Range, Shape};

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::{self, Axis, BatchRequest, Cartesian, RequestKind};
use crate::{
    BoxFuture, Layout, Result, SparseElementStream, TensorElement, TensorGeometry, TensorMatMul,
    TensorRead, TensorTransform, TensorViewSemantics, ValueBlockStream,
};

/// Maximum rows and columns per spatial tile; result size is TILE_SIDE squared.
pub(crate) const TILE_SIDE: usize = 32;

/// Maximum computed rectangle area per unique requested output.
const MAX_RECTANGLE_AMPLIFICATION: usize = 2;

/// A lazy, read-only matrix product with independent source adapters.
/// Sparse support is unioned across each contracted pair, then across contraction
/// positions. Transforms address the product's output, not its operands.
///
/// ```
/// use fensor::{Tensor, TensorFileEntry, TensorMatMul, TensorTransform};
/// async fn example<A: TensorFileEntry<f32>, B: TensorFileEntry<f32>>(a: &Tensor<A,f32>, b: &Tensor<B,f32>) -> fensor::Result<()> {
///     let view = a.view().matmul(&b.view()).await?.transpose(None)?.clone();
///     let _ = view.clone();
///     Ok(())
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMatMul, TensorWrite};
/// fn writable<T: TensorWrite>(_: &T) {}
/// async fn example<F: TensorFileEntry<u8>>(a: &Tensor<F,u8>) {
///     writable(&a.view().matmul(&a.view()).await.unwrap());
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMatMul, TensorArray};
/// fn stored<T: TensorArray>(_: &T) {}
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F,f32>) {
///     stored(&a.view().matmul(&a.view()).await.unwrap());
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMatMul};
/// async fn example<A: TensorFileEntry<f32>, B: TensorFileEntry<f64>>(a: &Tensor<A,f32>, b: &Tensor<B,f64>) {
///     let _ = a.view().matmul(&b.view()).await;
/// }
/// ```
#[derive(Clone)]
pub struct MatMulView<Left, Right> {
    left: Left,
    right: Right,
    // Original output geometry is rank-sized; transforms retain a shared mapping.
    output_shape: Shape,
    output_strides: Vec<usize>,
    mapping: CoordinateMap,
}

impl<L, R> TensorMatMul<R> for L
where
    L: Expression + Clone,
    R: Expression<DType = L::DType> + Clone,
    L::DType: TensorElement,
{
    type Output = MatMulView<Self, R>;

    fn matmul<'a>(&'a self, rhs: &'a R) -> BoxFuture<'a, Result<Self::Output>> {
        Box::pin(async move {
            let output_shape = self.matmul_output_shape(rhs)?;
            let output_strides = crate::schema::contiguous_strides(&output_shape)?.to_vec();
            let mapping = CoordinateMap::identity(output_shape.clone(), &output_strides);

            Ok(MatMulView {
                left: self.clone(),
                right: rhs.clone(),
                output_shape,
                output_strides,
                mapping,
            })
        })
    }
}

// At most 4096 requests across all tiles; each key is one rank-sized coordinate.
// Values retain request order and duplicates for scattering after computation.
type TileRequests = BTreeMap<Vec<u64>, Vec<(usize, usize, usize)>>;

fn plan_tiles(
    mapping: &CoordinateMap,
    shape: &[usize],
    strides: &[usize],
    coords: &BatchRequest,
) -> Result<TileRequests> {
    if coords.len() > expression::MAX_BATCH_ELEMENTS {
        return Err(crate::Error::InvalidLayout(format!(
            "matrix coordinate batch: expected at most {}, got {}",
            expression::MAX_BATCH_ELEMENTS,
            coords.len()
        )));
    }

    let mut tiles: TileRequests = BTreeMap::new();
    let mut cursor = coords.cursor(&mapping.shape)?;
    let mut input = Vec::new();
    let mut mapped = Vec::new();
    let mut position = 0;

    while cursor.next_into(&mut input) {
        mapping.resolve_into(&input, shape, strides, &mut mapped)?;
        let column = mapped[mapped.len() - 1] as usize;
        let row = mapped[mapped.len() - 2] as usize;
        let mut coord = mapped[..mapped.len() - 2].to_vec();
        coord.push((row / TILE_SIDE) as u64);
        coord.push((column / TILE_SIDE) as u64);
        tiles
            .entry(coord)
            .or_default()
            .push((position, row, column));
        position += 1;
    }

    Ok(tiles)
}

// All rectangle metadata is bounded by the current request batch. Scatter entries
// retain duplicates; arithmetic uses each unique requested pair only once.
struct Rectangle {
    rows: Vec<usize>,
    columns: Vec<usize>,
    scatter: Scatter,
}

fn plan_rectangles(requests: Vec<(usize, usize, usize)>) -> Vec<Rectangle> {
    // Sort one request vector rather than allocating a list for every unique pair.
    let mut requests = requests;
    requests.sort_unstable_by_key(|(_, row, column)| (*row, *column));
    let mut by_row: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for &(_, row, column) in &requests {
        let columns = by_row.entry(row).or_default();
        if columns.last() != Some(&column) {
            columns.push(column);
        }
    }

    let unique = by_row.values().map(Vec::len).sum::<usize>();
    let rows: Vec<_> = by_row.keys().copied().collect();
    let mut columns: Vec<_> = by_row.values().flatten().copied().collect();
    columns.sort_unstable();
    columns.dedup();

    let groups = if rows.len() * columns.len() <= MAX_RECTANGLE_AMPLIFICATION * unique {
        vec![(columns, rows)]
    } else {
        let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

        for (row, columns) in by_row {
            groups.entry(columns).or_default().push(row);
        }
        groups.into_iter().collect()
    };
    groups
        .into_iter()
        .map(|(columns, rows)| {
            let mut scatter = Vec::new();

            for &(position, row, column) in &requests {
                if let Ok(i) = rows.binary_search(&row) {
                    let j = columns.binary_search(&column).expect("planned column");
                    scatter.push((position, i * columns.len() + j));
                }
            }
            Rectangle {
                rows,
                columns,
                scatter: Scatter::Explicit(scatter),
            }
        })
        .collect()
}

fn contraction_step(rows: usize, columns: usize) -> usize {
    expression::MAX_BATCH_ELEMENTS / rows.max(columns)
}

// Regular scattering needs only axis/segment metadata, not one entry per value.
enum Scatter {
    Explicit(Vec<(usize, usize)>),
    Cartesian {
        start: usize,
        width: usize,
        rows: Vec<(usize, usize)>,
        columns: Vec<(usize, usize)>,
    },
    Spans(Vec<(usize, usize, usize)>), // destination start, tile offset, length
}

impl Scatter {
    fn each(&self, columns: usize, mut write: impl FnMut(usize, usize)) {
        match self {
            Self::Explicit(points) => {
                for &(position, offset) in points {
                    write(position, offset);
                }
            }
            Self::Cartesian {
                start,
                width,
                rows,
                columns: selected,
            } => {
                for &(position, i) in rows {
                    for &(column, j) in selected {
                        write(start + position * width + column, i * columns + j);
                    }
                }
            }
            Self::Spans(spans) => {
                for &(position, offset, len) in spans {
                    for j in 0..len {
                        write(position + j, offset + j);
                    }
                }
            }
        }
    }
}

struct MatrixPlan {
    prefix: Vec<u64>,
    rectangle: Rectangle,
}

type AxisTiles = BTreeMap<usize, Vec<(usize, usize)>>;

fn axis_tiles(axis: &Axis) -> AxisTiles {
    let mut tiles = BTreeMap::<usize, Vec<_>>::new();

    for position in 0..axis.len() {
        let value = axis.at(position) as usize;
        tiles
            .entry(value / TILE_SIDE)
            .or_default()
            .push((position, value));
    }
    tiles
}

fn selected_axis(positions: &[(usize, usize)]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut values: Vec<_> = positions.iter().map(|(_, value)| *value).collect();
    values.sort_unstable();
    values.dedup();
    let scatter = positions
        .iter()
        .map(|(position, value)| {
            (
                *position,
                values.binary_search(value).expect("selected axis"),
            )
        })
        .collect();
    (values, scatter)
}

fn cartesian_plans(rectangles: &[Cartesian], shape: &[usize]) -> Result<Vec<MatrixPlan>> {
    let rank = shape.len();
    let mut plans = Vec::new();
    let mut start = 0;

    for rect in rectangles {
        let axes = rect.axes();
        let prefix_request =
            BatchRequest::rectangles(vec![Cartesian::new(axes[..rank - 2].to_vec())?])?;
        let mut prefixes = prefix_request.cursor(&shape[..rank - 2])?;
        let row_tiles = axis_tiles(&axes[rank - 2]);
        let column_tiles = axis_tiles(&axes[rank - 1]);
        let mut prefix = Vec::new();

        while prefixes.next_into(&mut prefix) {
            for row_positions in row_tiles.values() {
                let (rows, row_scatter) = selected_axis(row_positions);

                for column_positions in column_tiles.values() {
                    let (columns, column_scatter) = selected_axis(column_positions);
                    plans.push(MatrixPlan {
                        prefix: prefix.clone(),
                        rectangle: Rectangle {
                            rows: rows.clone(),
                            columns,
                            scatter: Scatter::Cartesian {
                                start,
                                width: axes[rank - 1].len(),
                                rows: row_scatter.clone(),
                                columns: column_scatter,
                            },
                        },
                    });
                }
            }
            start += axes[rank - 2].len() * axes[rank - 1].len();
        }
    }

    Ok(plans)
}

fn linear_plans(start: usize, len: usize, shape: &[usize]) -> Vec<MatrixPlan> {
    // A span intersects at most len row segments. No coordinate/pair expansion.
    let rank = shape.len();
    let width = shape[rank - 1];
    let height = shape[rank - 2];
    let mut tiles = BTreeMap::<Vec<u64>, Vec<(usize, usize, usize, usize)>>::new();
    let mut done = 0;

    while done < len {
        let flat = start + done;
        let column = flat % width;
        let row = (flat / width) % height;
        let mut matrix = flat / width / height;
        let mut key = vec![0; rank];

        for i in (0..rank - 2).rev() {
            key[i] = (matrix % shape[i]) as u64;
            matrix /= shape[i];
        }
        key[rank - 2] = (row / TILE_SIDE) as u64;
        key[rank - 1] = (column / TILE_SIDE) as u64;
        let count = (len - done)
            .min(width - column)
            .min(TILE_SIDE - column % TILE_SIDE);
        tiles
            .entry(key)
            .or_default()
            .push((done, row, column, count));
        done += count;
    }
    tiles
        .into_iter()
        .map(|(mut prefix, segments)| {
            let row_origin = prefix[rank - 2] as usize * TILE_SIDE;
            let column_origin = prefix[rank - 1] as usize * TILE_SIDE;
            prefix.truncate(rank - 2);
            let mut row_mask = 0u64;
            let mut column_mask = 0u64;

            for &(_, row, column, len) in &segments {
                row_mask |= 1 << (row - row_origin);
                column_mask |= ((1u64 << len) - 1) << (column - column_origin);
            }

            let rows: Vec<_> = (0..TILE_SIDE)
                .filter(|i| row_mask & (1 << i) != 0)
                .map(|i| row_origin + i)
                .collect();
            let columns: Vec<_> = (0..TILE_SIDE)
                .filter(|i| column_mask & (1 << i) != 0)
                .map(|i| column_origin + i)
                .collect();
            let spans = segments
                .into_iter()
                .map(|(position, row, column, len)| {
                    (
                        position,
                        rows.binary_search(&row).expect("planned row") * columns.len()
                            + columns.binary_search(&column).expect("planned column"),
                        len,
                    )
                })
                .collect();
            MatrixPlan {
                prefix,
                rectangle: Rectangle {
                    rows,
                    columns,
                    scatter: Scatter::Spans(spans),
                },
            }
        })
        .collect()
}

fn plan_request(
    mapping: &CoordinateMap,
    shape: &[usize],
    strides: &[usize],
    request: &BatchRequest,
) -> Result<Vec<MatrixPlan>> {
    request.validate(&mapping.shape)?;
    if mapping.is_identity(shape, strides) {
        match request.kind() {
            RequestKind::Rectangles(rectangles) => return cartesian_plans(rectangles, shape),
            RequestKind::Linear { start } => return Ok(linear_plans(*start, request.len(), shape)),
            RequestKind::Explicit(_) => {}
        }
    }

    Ok(plan_tiles(mapping, shape, strides, request)?
        .into_iter()
        .flat_map(|(mut key, requests)| {
            key.truncate(key.len() - 2);
            plan_rectangles(requests)
                .into_iter()
                .map(move |rectangle| MatrixPlan {
                    prefix: key.clone(),
                    rectangle,
                })
        })
        .collect())
}

fn operand_request(
    shape: &[usize],
    prefix: &[u64],
    fixed: &[usize],
    start: usize,
    count: usize,
    left: bool,
) -> Result<BatchRequest> {
    let mut axes: Vec<_> = prefix.iter().map(|v| Axis::range(*v as usize, 1)).collect();
    let selection = Axis::Selected(fixed.iter().map(|v| *v as u64).collect());
    let contraction = Axis::range(start, count);
    if left {
        axes.extend([selection, contraction]);
    } else {
        axes.extend([contraction, selection]);
    }
    BatchRequest::rectangles(vec![crate::slice::Slice::new(shape, axes)?.rectangle()?])
}

fn combine_partial<T: TensorElement>(
    accumulated: &mut Option<Vec<T>>,
    partial: Vec<T>,
    expected: usize,
) -> Result<()> {
    for (context, actual) in std::iter::once(("partial", partial.len())).chain(
        accumulated
            .as_ref()
            .map(|values| ("accumulator", values.len())),
    ) {
        if actual != expected {
            return Err(crate::Error::InvalidLayout(format!(
                "matrix {context}: expected {expected} values, got {actual}"
            )));
        }
    }

    if let Some(values) = accumulated {
        for (value, partial) in values.iter_mut().zip(partial) {
            *value = <T as Number>::add(*value, partial);
        }
    } else {
        *accumulated = Some(partial);
    }

    Ok(())
}

impl<L, R> Expression for MatMulView<L, R>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
{
    fn preferred_requests(&self, shape: &[usize]) -> Result<Option<expression::RequestIterator>> {
        let requests: expression::RequestIterator = if shape.len() < 2 {
            Box::new(request::linear_requests(shape)?)
        } else {
            Box::new(request::tiled_requests(shape)?)
        };
        Ok(Some(requests))
    }

    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            #[cfg(test)]
            let planning = crate::read_metrics::Timer::new(|m| &mut m.planning);
            let tiles = plan_request(
                &self.mapping,
                &self.output_shape,
                &self.output_strides,
                coords,
            )?;

            #[cfg(test)]
            drop(planning);
            let mut values = vec![L::DType::ZERO; coords.len()];
            let mut support =
                matches!(self.layout(), Layout::Sparse { .. }).then(|| vec![0; coords.len()]);
            let inner = self.left.shape()[self.left.ndim() - 1];

            for MatrixPlan {
                prefix: key,
                rectangle:
                    Rectangle {
                        rows,
                        columns,
                        scatter,
                    },
            } in tiles
            {
                let mut summaries = support
                    .as_ref()
                    .map(|_| (vec![false; rows.len()], vec![false; columns.len()]));
                // Only one running result tile, at most TILE_SIDE * TILE_SIDE values.
                let mut accumulated = None;
                let step = contraction_step(rows.len(), columns.len());

                for start in (0..inner).step_by(step) {
                    let count = (inner - start).min(step);
                    let left_coords =
                        operand_request(self.left.shape(), &key, &rows, start, count, true)?;
                    let right_coords =
                        operand_request(self.right.shape(), &key, &columns, start, count, false)?;

                    // Direct evaluation: nested products/reductions start no buffered streams.
                    #[cfg(test)]
                    let operands = crate::read_metrics::Timer::new(|m| &mut m.operands);
                    let left = expression::evaluate_batch(&self.left, &left_coords).await?;
                    let right = expression::evaluate_batch(&self.right, &right_coords).await?;

                    #[cfg(test)]
                    drop(operands);
                    if let Some((row_support, column_support)) = &mut summaries {
                        for (i, supported) in row_support.iter_mut().enumerate() {
                            *supported |= left.support.as_ref().is_none_or(|s| {
                                s[i * count..(i + 1) * count].iter().any(|v| *v != 0)
                            });
                        }

                        for (j, supported) in column_support.iter_mut().enumerate() {
                            *supported |= right
                                .support
                                .as_ref()
                                .is_none_or(|s| (0..count).any(|k| s[k * columns.len() + j] != 0));
                        }
                    }

                    let left = expression::batch_array(left.values)?
                        .reshape(ha_ndarray::shape![rows.len(), count])?;
                    let right = expression::batch_array(right.values)?
                        .reshape(ha_ndarray::shape![count, columns.len()])?;
                    #[cfg(test)]
                    let backend = crate::read_metrics::Timer::new(|m| &mut m.backend);
                    let partial = left.matmul(right)?.buffer()?.to_slice()?.into_vec();

                    #[cfg(test)]
                    drop(backend);

                    #[cfg(test)]
                    crate::read_metrics::record(|m| m.backend_calls += 1);

                    #[cfg(test)]
                    let _accumulation = crate::read_metrics::Timer::new(|m| &mut m.accumulate);
                    combine_partial(&mut accumulated, partial, rows.len() * columns.len())?;
                }

                let accumulated = accumulated.expect("nonzero contraction dimension");
                scatter.each(columns.len(), |position, offset| {
                    if let (Some(support), Some((row_support, column_support))) =
                        (&mut support, &summaries)
                    {
                        support[position] = u8::from(
                            row_support[offset / columns.len()]
                                || column_support[offset % columns.len()],
                        );
                    }
                    values[position] = accumulated[offset];
                });
            }

            Batch {
                array: expression::batch_array(values)?,
                support,
            }
            .masked()
        })
    }
}

impl<L, R> TensorGeometry for MatMulView<L, R>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    type DType = L::DType;

    fn dtype(&self) -> crate::NumberType {
        self.left.dtype()
    }

    fn shape(&self) -> &[usize] {
        &self.mapping.shape
    }

    fn layout(&self) -> Layout {
        match (self.left.layout(), self.right.layout()) {
            (Layout::Sparse { .. }, Layout::Sparse { .. }) => Layout::Sparse { axis: None },
            _ => Layout::Dense,
        }
    }
}

impl<L, R> TensorViewSemantics for MatMulView<L, R>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    fn is_base_tensor(&self) -> bool {
        false
    }

    fn supports_write_through(&self) -> bool {
        false
    }
}

impl<L, R> TensorRead for MatMulView<L, R>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
{
    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, Self::DType>> {
        expression::coordinate_blocks(self)
    }

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            Ok(
                expression::evaluate_batch(self, &BatchRequest::point(coord))
                    .await?
                    .values[0],
            )
        })
    }

    fn read_blocks(&self) -> Result<ValueBlockStream<'_, Self::DType>> {
        let coords = request::linear_requests(self.shape())?;

        Ok(expression::ordered_batches(self, coords)
            .map_ok(|(_, batch)| batch.values)
            .boxed())
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        range: Range,
        requested_order: Axes,
    ) -> BoxFuture<'a, Result<SparseElementStream<'a, Self::DType>>> {
        Box::pin(async move {
            let coords = crate::traits::sparse_coords(self, range, requested_order)?;

            Ok(
                expression::ordered_batches(self, request::explicit_requests(coords))
                    .and_then(move |(coords, values)| async move {
                        Ok(futures::stream::iter(expression::sparse_elements(
                            coords,
                            values,
                            self.shape(),
                        )?))
                    })
                    .try_flatten()
                    .boxed(),
            )
        })
    }
}

impl<L, R> TensorTransform for MatMulView<L, R>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.reshape(shape)?,
            ..self
        })
    }

    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.broadcast(shape)?,
            ..self
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.slice(range)?,
            ..self
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.transpose(permutation)?,
            ..self
        })
    }

    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.flip(axis)?,
            ..self
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.squeeze(axes)?,
            ..self
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            mapping: self.mapping.unsqueeze(axes)?,
            ..self
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_shapes_reject_overflow_and_empty_dimensions() {
        assert!(crate::validate::matmul_output_shape(&[usize::MAX, 2], &[2, 1]).is_err());
        assert!(crate::validate::matmul_output_shape(&[usize::MAX, 1], &[1, 2]).is_err());
        assert!(crate::validate::matmul_output_shape(&[1, 0], &[0, 1]).is_err());
    }

    #[test]
    fn tile_planning_is_bounded_and_preserves_requests() {
        let shape = ha_ndarray::shape![2, 1000, 1000];
        let strides = crate::schema::contiguous_strides(&shape).unwrap();
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        // Fixed 64-by-64 fixture spans four current spatial tiles.
        let coords: Vec<_> = (0..64 * 64)
            .map(|i| vec![1, (i / 64) as u64, (i % 64) as u64])
            .collect();
        let tiles = plan_tiles(
            &mapping,
            &shape,
            &strides,
            &BatchRequest::explicit(coords).unwrap(),
        )
        .unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles.values().map(Vec::len).sum::<usize>(), 64 * 64);

        for requests in tiles.values() {
            let rows: std::collections::BTreeSet<_> = requests.iter().map(|(_, r, _)| r).collect();
            let columns: std::collections::BTreeSet<_> =
                requests.iter().map(|(_, _, c)| c).collect();
            assert!(
                rows.len() * contraction_step(rows.len(), columns.len())
                    <= expression::MAX_BATCH_ELEMENTS
            );
            assert!(
                columns.len() * contraction_step(rows.len(), columns.len())
                    <= expression::MAX_BATCH_ELEMENTS
            );
        }
        assert!(BatchRequest::explicit(vec![vec![0, 0, 0]; 4097]).is_err());
        let requests = plan_tiles(
            &mapping,
            &shape,
            &strides,
            &BatchRequest::explicit(vec![vec![0, 1, 2], vec![0, 1, 2]]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            requests.values().next().unwrap(),
            &vec![(0, 1, 2), (1, 1, 2)]
        );
    }

    #[test]
    fn irregular_rectangles_bound_arithmetic_and_retain_duplicates() {
        for requests in [
            (0..TILE_SIDE).map(|i| (i, i, i)).collect::<Vec<_>>(),
            (0..TILE_SIDE * TILE_SIDE)
                .map(|i| (i, i / TILE_SIDE, i % TILE_SIDE))
                .collect(),
            vec![(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 0, 1)],
        ] {
            let unique: std::collections::BTreeSet<_> =
                requests.iter().map(|(_, r, c)| (*r, *c)).collect();
            let rectangles = plan_rectangles(requests.clone());
            assert!(
                rectangles
                    .iter()
                    .map(|r| r.rows.len() * r.columns.len())
                    .sum::<usize>()
                    <= MAX_RECTANGLE_AMPLIFICATION * unique.len()
            );
            let mut seen = vec![false; requests.len()];

            for rect in rectangles {
                rect.scatter.each(rect.columns.len(), |position, offset| {
                    assert!(!seen[position]);
                    seen[position] = true;
                    assert_eq!(
                        (
                            rect.rows[offset / rect.columns.len()],
                            rect.columns[offset % rect.columns.len()]
                        ),
                        (requests[position].1, requests[position].2)
                    );
                });
            }
            assert!(seen.into_iter().all(|v| v));
        }
        assert_eq!(
            contraction_step(TILE_SIDE, TILE_SIDE),
            expression::MAX_BATCH_ELEMENTS / TILE_SIDE
        );
        assert_eq!(contraction_step(1, 1), expression::MAX_BATCH_ELEMENTS);
        assert_eq!(contraction_step(2, 1), expression::MAX_BATCH_ELEMENTS / 2);
        assert!(combine_partial(&mut Some(vec![1u8]), vec![2, 3], 1).is_err());
        assert!(combine_partial(&mut Some(vec![1u8, 2]), vec![3], 1).is_err());
    }

    #[test]
    fn planner_work_counts() {
        for (name, m, k, n) in [
            ("square", 32, 129, 32),
            ("wide", 32, 17, 4097),
            ("tall", 129, 129, 1),
            ("dot", 1, 4097, 1),
        ] {
            let shape = ha_ndarray::shape![m, n];
            let strides = crate::schema::contiguous_strides(&shape).unwrap();
            let mapping = CoordinateMap::identity(shape.clone(), &strides);
            let mut totals = Vec::new();

            for revised in [false, true] {
                let requests: expression::RequestIterator = if revised {
                    Box::new(request::tiled_requests(&shape).unwrap())
                } else {
                    Box::new(request::linear_requests(&shape).unwrap())
                };
                let mut operands = 0;
                let mut calls = 0;

                for request in requests {
                    for plan in plan_request(&mapping, &shape, &strides, &request).unwrap() {
                        let rect = plan.rectangle;
                        operands += (rect.rows.len() + rect.columns.len()) * k;
                        calls += k.div_ceil(if revised {
                            contraction_step(rect.rows.len(), rect.columns.len())
                        } else {
                            // Historical fixed-chunk baseline, not the current adaptive limit.
                            128
                        });
                    }
                }
                totals.push((operands, calls));
                println!("PLAN,{name},{revised},{operands},{calls}");
            }
            assert!(totals[1].0 <= totals[0].0);
            assert!(totals[1].1 <= totals[0].1);
        }
    }

    fn planned_coordinates(plans: &[MatrixPlan], len: usize) -> Vec<Vec<u64>> {
        let mut coords = vec![Vec::new(); len];

        for plan in plans {
            let rect = &plan.rectangle;
            rect.scatter.each(rect.columns.len(), |position, offset| {
                assert!(coords[position].is_empty());
                let mut coord = plan.prefix.clone();
                coord.extend([
                    rect.rows[offset / rect.columns.len()] as u64,
                    rect.columns[offset % rect.columns.len()] as u64,
                ]);
                coords[position] = coord;
            });
        }
        coords
    }

    #[test]
    fn compact_and_explicit_plans_agree() {
        let shape = ha_ndarray::shape![2, 35, 67];
        let strides = crate::schema::contiguous_strides(&shape).unwrap();
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        let mut requests: Vec<_> = request::tiled_requests(&shape).unwrap().collect();
        requests.extend([
            BatchRequest::linear(66, 3000).unwrap(),
            BatchRequest::linear(2000, 2690).unwrap(),
            BatchRequest::rectangles(vec![
                Cartesian::new(vec![
                    Axis::Selected(vec![1, 0, 1]),
                    Axis::Selected(vec![34, 0, 34, 1]),
                    Axis::Selected(vec![66, 0, 31, 32]),
                ])
                .unwrap(),
            ])
            .unwrap(),
        ]);

        for request in requests {
            let expected = request.coordinates(&shape).unwrap();
            let direct = plan_request(&mapping, &shape, &strides, &request).unwrap();
            assert!(
                direct
                    .iter()
                    .all(|p| !matches!(p.rectangle.scatter, Scatter::Explicit(_)))
            );
            let explicit = plan_request(
                &mapping,
                &shape,
                &strides,
                &BatchRequest::explicit(expected.clone()).unwrap(),
            )
            .unwrap();
            assert_eq!(planned_coordinates(&direct, request.len()), expected);
            assert_eq!(planned_coordinates(&explicit, request.len()), expected);
        }

        let flipped = mapping.flip(2).unwrap();
        let request = BatchRequest::linear(60, 100).unwrap();
        let expected: Vec<_> = request
            .coordinates(&shape)
            .unwrap()
            .iter()
            .map(|c| flipped.resolve(c, &shape, &strides).unwrap())
            .collect();
        let plans = plan_request(&flipped, &shape, &strides, &request).unwrap();
        assert!(
            plans
                .iter()
                .all(|p| matches!(p.rectangle.scatter, Scatter::Explicit(_)))
        );
        assert_eq!(planned_coordinates(&plans, request.len()), expected);
    }

    #[test]
    fn compact_planning_representation_counts() {
        let shape = ha_ndarray::shape![32, 4097];
        let strides = crate::schema::contiguous_strides(&shape).unwrap();
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        let mut descriptors = 0;
        let mut logical = 0;
        let mut scatter_slots = 0;
        let mut calls = 0;

        for request in request::tiled_requests(&shape).unwrap() {
            let RequestKind::Rectangles(rectangles) = request.kind() else {
                panic!("expanded regular traversal")
            };
            descriptors += rectangles.len();
            logical += request.len();

            for plan in plan_request(&mapping, &shape, &strides, &request).unwrap() {
                let Scatter::Cartesian { rows, columns, .. } = &plan.rectangle.scatter else {
                    panic!("expanded regular scatter")
                };
                scatter_slots += rows.len() + columns.len();
                calls += 17usize.div_ceil(contraction_step(
                    plan.rectangle.rows.len(),
                    plan.rectangle.columns.len(),
                ));
            }
        }
        assert_eq!(
            (logical, descriptors, scatter_slots, calls),
            (131104, 129, 8225, 129)
        );
        println!(
            "COMPACT,wide,logical={logical},rectangles={descriptors},scatter_axis_slots={scatter_slots},matmul_calls={calls},pre_evaluation_coordinate_payloads=0"
        );
    }
}
