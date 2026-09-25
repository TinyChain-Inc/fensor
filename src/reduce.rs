//! Bounded reductions over original expression support.

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{NDArrayReduceAll, Number, Real};

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::{self, BatchRequest};
use crate::schema::Coord;
use crate::{
    Axes, BoxFuture, Error, Layout, Range, Result, Shape, SparseElementStream, TensorElement,
    TensorGeometry, TensorRead, TensorReduce, TensorReduceAll, TensorReduceBoolean,
    TensorTransform, TensorViewSemantics, ValueBlockStream,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed reduction delegating numerical rules to ha-ndarray.
pub trait ReduceOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    fn partial(&self, values: Vec<T>) -> Result<T>;

    fn combine(left: T, right: T) -> T;

    fn empty() -> Result<T>;
}

/// Sum over supported values.
#[derive(Clone, Copy, Debug)]
pub struct Sum;

impl sealed::Sealed for Sum {}

impl<T: TensorElement + Real> ReduceOp<T> for Sum {
    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.sum_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Number>::add(left, right)
    }

    fn empty() -> Result<T> {
        Ok(T::ZERO)
    }
}

/// Product over supported values.
#[derive(Clone, Copy, Debug)]
pub struct Product;

impl sealed::Sealed for Product {}

impl<T: TensorElement + Real> ReduceOp<T> for Product {
    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.product_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Number>::mul(left, right)
    }

    fn empty() -> Result<T> {
        Ok(T::ONE)
    }
}

/// Min over supported values.
#[derive(Clone, Copy, Debug)]
pub struct Min;

impl sealed::Sealed for Min {}

impl<T: TensorElement + Real> ReduceOp<T> for Min {
    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.min_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Real>::min(left, right)
    }

    fn empty() -> Result<T> {
        Err(Error::Unsupported("min_all has empty support".into()))
    }
}

/// Max over supported values.
#[derive(Clone, Copy, Debug)]
pub struct Max;

impl sealed::Sealed for Max {}

impl<T: TensorElement + Real> ReduceOp<T> for Max {
    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.max_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Real>::max(left, right)
    }

    fn empty() -> Result<T> {
        Err(Error::Unsupported("max_all has empty support".into()))
    }
}

// The accumulator holds one partial result, never a list of batch results.
fn accumulate<T: TensorElement, O: ReduceOp<T>>(
    op: &O,
    state: &mut Option<T>,
    batch: expression::EvaluatedBatch<T>,
) -> Result<()> {
    let values = batch.populated()?;

    if !values.is_empty() {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.reduction_calls += 1);
        let partial = op.partial(values)?;
        *state = Some(match *state {
            Some(previous) => O::combine(previous, partial),
            None => partial,
        });
    }

    Ok(())
}

async fn terminal<E, O>(source: &E, op: O) -> Result<E::DType>
where
    E: Expression,
    E::DType: TensorElement,
    O: ReduceOp<E::DType>,
{
    let requests = source.slice_requests(crate::slice::Slice::full(source.shape())?)?;
    // Numeric aggregates allow evaluation-order differences. Consume completed
    // batches immediately; boolean terminals retain logical ordered delivery.
    let batches =
        expression::evaluation_futures(source, requests).buffer_unordered(num_cpus::get().max(1));
    futures::pin_mut!(batches);
    let mut state = None;

    while let Some((_, batch)) = batches.try_next().await? {
        accumulate::<_, O>(&op, &mut state, batch)?;
    }

    state.map(Ok).unwrap_or_else(O::empty)
}

impl<E> TensorReduceAll for E
where
    E: Expression + TensorRead,
    E::DType: TensorElement + Real,
{
    fn sum_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Sum>(self, Sum))
    }

    fn product_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Product>(self, Product))
    }

    fn min_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Min>(self, Min))
    }

    fn max_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Max>(self, Max))
    }
}

impl<E> TensorReduceBoolean for E
where
    E: Expression + TensorRead,
    E::DType: TensorElement,
{
    fn all(&self) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async move {
            // Logical batches preserve which errors precede a decisive value;
            // occupied-region traversal could change those short-circuit boundaries.
            let coords = request::linear_requests(self.shape())?;
            let mut batches = expression::ordered_batches(self, coords);

            while let Some((_, batch)) = batches.try_next().await? {
                if batch.populated()?.into_iter().any(|v| v == E::DType::ZERO) {
                    return Ok(false);
                }
            }

            Ok(true)
        })
    }

    fn any(&self) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async move {
            // Logical batches preserve which errors precede a decisive value;
            // occupied-region traversal could change those short-circuit boundaries.
            let coords = request::linear_requests(self.shape())?;
            let mut batches = expression::ordered_batches(self, coords);

            while let Some((_, batch)) = batches.try_next().await? {
                if batch.populated()?.into_iter().any(|v| v != E::DType::ZERO) {
                    return Ok(true);
                }
            }

            Ok(false)
        })
    }
}

/// A lazy, read-only reduction. Transforms map reduced outputs, not source axes.
///
/// Descriptions clone without requiring a clonable filesystem adapter.
///
/// ```
/// use fensor::{Tensor, TensorFileEntry, TensorReduce, TensorReduceAll, TensorTransform, TensorUnary};
/// use ha_ndarray::axes;
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) -> fensor::Result<()> {
///     let reduced = a.view().sum(axes![0], false).await?.clone();
///     let _ = reduced.exp().await?.transpose(None)?.sum_all().await?;
///     Ok(())
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorReduce, TensorTransform, TensorWrite};
/// use ha_ndarray::axes;
/// fn writable<T: TensorWrite>(_: &T) {}
/// async fn example<F: TensorFileEntry<f32>>(a: &Tensor<F, f32>) {
///     writable(&a.view().sum(axes![0], false).await.unwrap().flip(0).unwrap());
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorReduce, TensorArray};
/// use ha_ndarray::axes;
/// fn stored<T: TensorArray>(_: &T) {}
/// async fn example<F: TensorFileEntry<u8>>(a: &Tensor<F, u8>) {
///     stored(&a.view().product(axes![0], false).await.unwrap());
/// }
/// ```
#[derive(Clone)]
pub struct ReduceView<Source, Op> {
    source: Source,
    axes: Axes,
    keepdims: bool,
    output_shape: Shape,
    // One stride per output axis, not per output value.
    output_strides: crate::Strides,
    mapping: CoordinateMap,
    op: Op,
}

impl<S: TensorGeometry, O> ReduceView<S, O> {
    fn new(source: S, mut axes: Axes, keepdims: bool, op: O) -> Result<Self> {
        crate::schema::validate_shape_dims(source.shape())?;
        axes.sort_unstable();
        axes.dedup();

        if axes.iter().any(|axis| *axis >= source.ndim()) {
            return Err(Error::InvalidLayout("reduction axis out of bounds".into()));
        }

        let mut output_shape: Shape = source.shape().into();

        for &axis in axes.iter().rev() {
            if keepdims {
                output_shape[axis] = 1;
            } else {
                output_shape.remove(axis);
            }
        }

        if output_shape.is_empty() {
            output_shape.push(1);
        }

        let output_strides = crate::schema::contiguous_strides(&output_shape)?;
        let mapping = CoordinateMap::identity(output_shape.clone(), &output_strides);

        Ok(Self {
            source,
            axes,
            keepdims,
            output_shape,
            output_strides,
            mapping,
            op,
        })
    }

    // One descriptor per source axis; even a huge group is enumerated lazily.
    fn source_group_axes(&self, coord: &[u64]) -> Result<Vec<request::Axis>> {
        let coord = self
            .mapping
            .resolve(coord, &self.output_shape, &self.output_strides)?;
        let mut next = 0;
        let axes = self
            .source
            .shape()
            .iter()
            .enumerate()
            .map(|(axis, dim)| {
                if self.axes.contains(&axis) {
                    request::Axis::Span {
                        start: 0,
                        step: 1,
                        len: *dim,
                    }
                } else {
                    let i = if self.keepdims {
                        axis
                    } else {
                        let i = next;
                        next += 1;
                        i
                    };
                    request::Axis::Span {
                        start: coord[i],
                        step: 1,
                        len: 1,
                    }
                }
            })
            .collect();

        Ok(axes)
    }
}

impl<E> TensorReduce for E
where
    E: Expression + Clone,
    E::DType: TensorElement + Real,
{
    type SumOutput = ReduceView<Self, Sum>;

    type ProductOutput = ReduceView<Self, Product>;

    type MinOutput = ReduceView<Self, Min>;

    type MaxOutput = ReduceView<Self, Max>;

    fn sum(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::SumOutput>> {
        Box::pin(async move { ReduceView::new(self.clone(), axes, keepdims, Sum) })
    }

    fn product(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::ProductOutput>> {
        Box::pin(async move { ReduceView::new(self.clone(), axes, keepdims, Product) })
    }

    fn min(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::MinOutput>> {
        Box::pin(async move { ReduceView::new(self.clone(), axes, keepdims, Min) })
    }

    fn max(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::MaxOutput>> {
        Box::pin(async move { ReduceView::new(self.clone(), axes, keepdims, Max) })
    }
}

impl<S, O> TensorGeometry for ReduceView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    type DType = S::DType;

    fn dtype(&self) -> crate::NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.mapping.shape
    }

    fn layout(&self) -> Layout {
        match self.source.layout() {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: None },
        }
    }
}

impl<S, O> TensorViewSemantics for ReduceView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    fn is_base_tensor(&self) -> bool {
        false
    }

    fn supports_write_through(&self) -> bool {
        false
    }
}

impl<S, O> Expression for ReduceView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    fn build<'a>(&'a self, coords: &'a BatchRequest) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            // Output buffers are bounded by the current evaluation batch.
            let mut values = Vec::with_capacity(coords.len());
            let mut support = matches!(self.layout(), Layout::Sparse { .. })
                .then(|| Vec::with_capacity(coords.len()));

            let mut cursor = coords.cursor(self.shape())?;
            let mut coord = Coord::new();
            let mut pending = None;

            loop {
                let slice = if let Some(slice) = pending.take() {
                    slice
                } else if cursor.next_into(&mut coord) {
                    crate::slice::Slice::new(self.source.shape(), self.source_group_axes(&coord)?)?
                } else {
                    break;
                };
                if slice.len() > expression::MAX_BATCH_ELEMENTS as u64 {
                    let mut state = None;
                    let mut requests = self.source.slice_requests(slice)?;
                    // Inner consumers never start buffered streams.
                    while let Some(request) = requests.try_next().await? {
                        let batch = expression::evaluate_batch(&self.source, &request).await?;
                        accumulate::<_, O>(&self.op, &mut state, batch)?;
                    }

                    if let Some(support) = &mut support {
                        support.push(u8::from(state.is_some()));
                    }
                    values.push(state.unwrap_or(S::DType::ZERO));
                    continue;
                }

                // Complete small groups share a storage read; lengths and rectangles
                // contain at most MAX_BATCH_ELEMENTS entries, independent of the output size.
                let mut total = slice.len();
                let mut lengths = vec![total];
                let mut rectangles = vec![slice.rectangle()?];

                while total < expression::MAX_BATCH_ELEMENTS as u64 && cursor.next_into(&mut coord)
                {
                    let next = crate::slice::Slice::new(
                        self.source.shape(),
                        self.source_group_axes(&coord)?,
                    )?;
                    if next.len() > expression::MAX_BATCH_ELEMENTS as u64 - total {
                        pending = Some(next);
                        break;
                    }
                    total += next.len();
                    lengths.push(next.len());
                    rectangles.push(next.rectangle()?);
                }

                let batch = expression::evaluate_batch(
                    &self.source,
                    &BatchRequest::rectangles(rectangles)?,
                )
                .await?;
                let mut input = batch.values.into_iter();
                let mut masks = batch.support.map(Vec::into_iter);

                for len in lengths {
                    let mut state = None;
                    accumulate::<_, O>(
                        &self.op,
                        &mut state,
                        expression::EvaluatedBatch {
                            values: input.by_ref().take(len as usize).collect(),
                            support: masks
                                .as_mut()
                                .map(|m| m.by_ref().take(len as usize).collect()),
                        },
                    )?;
                    if let Some(support) = &mut support {
                        support.push(u8::from(state.is_some()));
                    }
                    values.push(state.unwrap_or(S::DType::ZERO));
                }
            }

            Ok(Batch {
                array: expression::batch_array(values)?,
                support,
            })
        })
    }
}

impl<S, O> TensorRead for ReduceView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
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

impl<S, O> TensorTransform for ReduceView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
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
