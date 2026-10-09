//! Bounded matrix tiles over filesystem-backed expressions.

use std::collections::BTreeMap;

use ha_ndarray::{MatrixDual, NDArrayRead, NDArrayTransform, Number};

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::{self, Axis, BatchRequest, Cartesian, RequestKind};
use crate::schema::Coord;
use crate::{
    BoxFuture, Layout, Result, Shape, TensorElement, TensorGeometry, TensorMatMul, TensorRead,
    TensorTransform, TensorViewSemantics,
};

/// Maximum rows and columns per spatial tile; result size is TILE_SIDE squared.
pub(crate) const TILE_SIDE: usize = 32;

/// Maximum computed rectangle area per unique requested output.
const MAX_RECTANGLE_AMPLIFICATION: usize = 2;

/// A lazy, read-only matrix product with independent source adapters.
/// Missing sparse inputs are numerical zeros. Transforms address the product's
/// output, not its operands.
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
    output_strides: crate::Strides,
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
            let output_strides = crate::schema::contiguous_strides(&output_shape)?;
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

// At most MAX_BATCH_ELEMENTS requests across all tiles; each key is rank-sized.
// Values retain request order and duplicates for scattering after computation.
type TileRequests = BTreeMap<Vec<u64>, Vec<(usize, u64, u64)>>;

fn plan_tiles(
    mapping: &CoordinateMap,
    shape: &[u64],
    strides: &[u64],
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
    mapping.visit_mapped(coords, shape, strides, |position, mapped| {
        let column = mapped[mapped.len() - 1];
        let row = mapped[mapped.len() - 2];
        let mut coord = mapped[..mapped.len() - 2].to_vec();
        coord.push(row / TILE_SIDE as u64);
        coord.push(column / TILE_SIDE as u64);
        tiles
            .entry(coord)
            .or_default()
            .push((position, row, column));
        Ok(())
    })?;

    Ok(tiles)
}

// All rectangle metadata is bounded by the current request batch. Scatter entries
// retain duplicates; arithmetic uses each unique requested pair only once.
struct Rectangle {
    rows: Vec<u64>,
    columns: Vec<u64>,
    scatter: Scatter,
}

fn plan_rectangles(requests: Vec<(usize, u64, u64)>) -> Vec<Rectangle> {
    // Sort one request vector rather than allocating a list for every unique pair.
    let mut requests = requests;
    requests.sort_unstable_by_key(|(_, row, column)| (*row, *column));
    let mut by_row: BTreeMap<u64, Vec<u64>> = BTreeMap::new();

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
        let mut groups: BTreeMap<Vec<u64>, Vec<u64>> = BTreeMap::new();

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

type AxisTiles = BTreeMap<u64, Vec<(usize, u64)>>;

fn axis_tiles(axis: &Axis) -> AxisTiles {
    let mut tiles = BTreeMap::<u64, Vec<_>>::new();

    for position in 0..axis.len() {
        let value = axis.at(position);
        tiles
            .entry(value / TILE_SIDE as u64)
            .or_default()
            .push((position as usize, value));
    }
    tiles
}

fn selected_axis(positions: &[(usize, u64)]) -> (Vec<u64>, Vec<(usize, usize)>) {
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

fn cartesian_plans(rectangles: &[Cartesian], shape: &[u64]) -> Result<Vec<MatrixPlan>> {
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
        let mut prefix = Coord::new();

        while prefixes.next_into(&mut prefix) {
            for row_positions in row_tiles.values() {
                let (rows, row_scatter) = selected_axis(row_positions);

                for column_positions in column_tiles.values() {
                    let (columns, column_scatter) = selected_axis(column_positions);
                    plans.push(MatrixPlan {
                        prefix: prefix.to_vec(),
                        rectangle: Rectangle {
                            rows: rows.clone(),
                            columns,
                            scatter: Scatter::Cartesian {
                                start,
                                width: axes[rank - 1].len() as usize,
                                rows: row_scatter.clone(),
                                columns: column_scatter,
                            },
                        },
                    });
                }
            }
            start += (axes[rank - 2].len() * axes[rank - 1].len()) as usize;
        }
    }

    Ok(plans)
}

fn linear_plans(start: u64, len: usize, shape: &[u64]) -> Result<Vec<MatrixPlan>> {
    // A span intersects at most len row segments. No coordinate/pair expansion.
    let rank = shape.len();
    let width = shape[rank - 1];
    let mut tiles = BTreeMap::<Vec<u64>, Vec<(usize, u64, u64, usize)>>::new();
    let mut coord = crate::schema::Coord::new();

    for (flat, output, length) in crate::request::linear_segments(start, len, width)? {
        crate::request::decode_flat(flat, shape, &mut coord)?;
        let row = coord[rank - 2];
        let mut done = 0;

        while done < length {
            let column = coord[rank - 1] + done as u64;
            let mut key = coord.to_vec();
            key[rank - 2] = row / TILE_SIDE as u64;
            key[rank - 1] = column / TILE_SIDE as u64;
            let count = (length - done).min(TILE_SIDE - (column % TILE_SIDE as u64) as usize);
            tiles
                .entry(key)
                .or_default()
                .push((output + done, row, column, count));
            done += count;
        }
    }

    Ok(tiles
        .into_iter()
        .map(|(mut prefix, segments)| {
            let row_origin = prefix[rank - 2] * TILE_SIDE as u64;
            let column_origin = prefix[rank - 1] * TILE_SIDE as u64;
            prefix.truncate(rank - 2);
            let mut row_mask = 0u64;
            let mut column_mask = 0u64;

            for &(_, row, column, len) in &segments {
                row_mask |= 1 << (row - row_origin);
                column_mask |= ((1u64 << len) - 1) << (column - column_origin);
            }

            let rows: Vec<_> = (0..TILE_SIDE)
                .filter(|i| row_mask & (1 << i) != 0)
                .map(|i| row_origin + i as u64)
                .collect();
            let columns: Vec<_> = (0..TILE_SIDE)
                .filter(|i| column_mask & (1 << i) != 0)
                .map(|i| column_origin + i as u64)
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
        .collect())
}

fn plan_request(
    mapping: &CoordinateMap,
    shape: &[u64],
    strides: &[u64],
    request: &BatchRequest,
) -> Result<Vec<MatrixPlan>> {
    request.validate(&mapping.shape)?;
    if mapping.is_identity(shape, strides) {
        match request.kind() {
            RequestKind::Rectangles(rectangles) => return cartesian_plans(rectangles, shape),
            RequestKind::Linear { start } => return linear_plans(*start, request.len(), shape),
            RequestKind::Explicit(_) | RequestKind::FlatRuns(_) => {}
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
    shape: &[u64],
    prefix: &[u64],
    fixed: &[u64],
    start: u64,
    count: usize,
    left: bool,
) -> Result<BatchRequest> {
    let mut axes: Vec<_> = prefix.iter().map(|v| Axis::range(*v, 1)).collect();
    let selection = Axis::Selected(fixed.to_vec());
    let contraction = Axis::range(start, count as u64);
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
    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([
            self.left.expression_nodes()?,
            self.right.expression_nodes()?,
        ])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.left.detach_sources(pending);
        self.right.detach_sources(pending);
    }

    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        let requests: expression::RequestIterator = if shape.len() < 2 {
            Box::new(request::linear_requests(shape)?)
        } else {
            Box::new(request::tiled_requests(shape)?)
        };

        Ok(expression::traversal::Preferred::Ready(Some(requests)))
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let tiles = plan_request(
                &self.mapping,
                &self.output_shape,
                &self.output_strides,
                &coords,
            )?;

            let mut values = vec![L::DType::ZERO; coords.len()];
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
                // Only one running result tile, at most TILE_SIDE * TILE_SIDE values.
                let mut accumulated = None;
                let step = contraction_step(rows.len(), columns.len());

                for start in (0..inner).step_by(step) {
                    let count = (inner - start).min(step as u64) as usize;
                    let left_coords =
                        operand_request(self.left.shape(), &key, &rows, start, count, true)?;
                    let right_coords =
                        operand_request(self.right.shape(), &key, &columns, start, count, false)?;

                    // Direct evaluation: nested products/reductions start no buffered streams.
                    let left = context
                        .evaluate(&self.left, std::sync::Arc::new(left_coords))
                        .await?;
                    let right = context
                        .evaluate(&self.right, std::sync::Arc::new(right_coords))
                        .await?;

                    let left = expression::batch_array(left.values)?
                        .reshape(ha_ndarray::shape![rows.len(), count])?;
                    let right = expression::batch_array(right.values)?
                        .reshape(ha_ndarray::shape![count, columns.len()])?;
                    let partial = left.matmul(right)?.buffer()?.to_slice()?.into_vec();

                    #[cfg(test)]
                    crate::read_metrics::record(|m| m.backend_calls += 1);

                    combine_partial(&mut accumulated, partial, rows.len() * columns.len())?;
                }

                let accumulated = accumulated.expect("nonzero contraction dimension");
                scatter.each(columns.len(), |position, offset| {
                    values[position] = accumulated[offset];
                });
            }

            Batch::from_values(values)
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

    fn shape(&self) -> &[u64] {
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
}

impl<L, R> TensorRead for MatMulView<L, R>
where
    L: Expression,
    R: Expression<DType = L::DType>,
    L::DType: TensorElement,
{
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<L, R> TensorTransform for MatMulView<L, R>
where
    L: TensorGeometry,
    R: TensorGeometry<DType = L::DType>,
    L::DType: TensorElement,
{
    crate::mapping::transform_methods!();
}

#[cfg(test)]
#[path = "../tests/unit/matmul/tests.rs"]
mod tests;
