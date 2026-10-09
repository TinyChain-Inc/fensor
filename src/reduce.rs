//! Bounded reductions over logical values, including implicit sparse zeros.

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{NDArrayReduceAll, Number, Real};

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::{self, BatchRequest};
use crate::schema::Coord;
use crate::{
    Axes, BoxFuture, Error, Layout, Result, Shape, TensorElement, TensorGeometry, TensorRead,
    TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorTransform, TensorViewSemantics,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed reduction delegating numerical rules to ha-ndarray.
pub trait ReduceOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type State: Copy + Send + Sync;
    type Output: TensorElement;

    fn partial(&self, values: Vec<T>) -> Result<Self::State>;

    fn combine(left: Self::State, right: Self::State) -> Self::State;

    fn finish(state: Self::State) -> Result<Self::Output>;

    fn empty() -> Result<Self::Output>;

    fn finish_axis(state: Option<Self::State>) -> Result<Self::Output> {
        state.map(Self::finish).unwrap_or(Ok(Self::Output::ZERO))
    }
}

/// Sum over logical values.
#[derive(Clone, Copy, Debug)]
pub struct Sum;

impl sealed::Sealed for Sum {}

impl<T: TensorElement> ReduceOp<T> for Sum {
    type State = T;
    type Output = T;

    fn finish(state: T) -> Result<T> {
        Ok(state)
    }

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

/// Product over logical values.
#[derive(Clone, Copy, Debug)]
pub struct Product;

impl sealed::Sealed for Product {}

impl<T: TensorElement> ReduceOp<T> for Product {
    type State = T;
    type Output = T;

    fn finish(state: T) -> Result<T> {
        Ok(state)
    }

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

/// Min over logical values.
#[derive(Clone, Copy, Debug)]
pub struct Min;

impl sealed::Sealed for Min {}

impl<T: TensorElement + Real> ReduceOp<T> for Min {
    type State = T;
    type Output = T;

    fn finish(state: T) -> Result<T> {
        Ok(state)
    }

    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.min_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Real>::min(left, right)
    }

    fn empty() -> Result<T> {
        Err(Error::Unsupported("min_all has empty input".into()))
    }
}

/// Max over logical values.
#[derive(Clone, Copy, Debug)]
pub struct Max;

impl sealed::Sealed for Max {}

impl<T: TensorElement + Real> ReduceOp<T> for Max {
    type State = T;
    type Output = T;

    fn finish(state: T) -> Result<T> {
        Ok(state)
    }

    fn partial(&self, values: Vec<T>) -> Result<T> {
        Ok(expression::batch_array(values)?.max_all()?)
    }

    fn combine(left: T, right: T) -> T {
        <T as Real>::max(left, right)
    }

    fn empty() -> Result<T> {
        Err(Error::Unsupported("max_all has empty input".into()))
    }
}

/// Numeric conversion owned by the statistics reductions.
pub trait StatisticsElement: TensorElement + sealed::Sealed {
    type Mean: TensorElement;

    fn components(self) -> [f64; 2];

    fn mean_value(value: [f64; 2]) -> Self::Mean;
}

macro_rules! real_statistics {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}

            impl StatisticsElement for $ty {
                type Mean = f64;

                fn components(self) -> [f64; 2] {
                    [self as f64, 0.0]
                }

                fn mean_value(value: [f64; 2]) -> f64 {
                    value[0]
                }
            }
        )+
    };
}

real_statistics!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cfg(feature = "complex")]
macro_rules! complex_statistics {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}

            impl StatisticsElement for $ty {
                type Mean = crate::complex::Complex64;

                fn components(self) -> [f64; 2] {
                    [self.re as f64, self.im as f64]
                }

                fn mean_value(value: [f64; 2]) -> Self::Mean {
                    Self::Mean::new(value[0], value[1])
                }
            }
        )+
    };
}

#[cfg(feature = "complex")]
complex_statistics!(crate::complex::Complex32, crate::complex::Complex64);

/// Mean counts every logical value, including implicit sparse zeros.
#[derive(Clone, Copy, Debug)]
pub struct Mean;

impl sealed::Sealed for Mean {}

impl<T: StatisticsElement> ReduceOp<T> for Mean {
    type State = (u64, [f64; 2]);
    type Output = T::Mean;

    fn partial(&self, values: Vec<T>) -> Result<Self::State> {
        let count = values.len() as u64;
        let sum = values.into_iter().fold([0.0; 2], |sum, value| {
            let value = value.components();
            [sum[0] + value[0], sum[1] + value[1]]
        });
        Ok((count, sum))
    }

    fn combine((n, l): Self::State, (m, r): Self::State) -> Self::State {
        (n + m, [l[0] + r[0], l[1] + r[1]])
    }

    fn finish((count, sum): Self::State) -> Result<Self::Output> {
        Ok(T::mean_value([
            sum[0] / count as f64,
            sum[1] / count as f64,
        ]))
    }

    fn empty() -> Result<Self::Output> {
        Ok(T::mean_value([f64::NAN; 2]))
    }

    fn finish_axis(state: Option<Self::State>) -> Result<Self::Output> {
        state
            .map(<Self as ReduceOp<T>>::finish)
            .unwrap_or_else(<Self as ReduceOp<T>>::empty)
    }
}

/// Population standard deviation using squared complex magnitude.
#[derive(Clone, Copy, Debug)]
pub struct StandardDeviation;

impl sealed::Sealed for StandardDeviation {}

impl<T: StatisticsElement> ReduceOp<T> for StandardDeviation {
    type State = (u64, [f64; 2], f64);
    type Output = f64;

    fn partial(&self, values: Vec<T>) -> Result<Self::State> {
        let mut state = (0, [0.0; 2], 0.0);

        for value in values {
            let value = value.components();
            let count = state.0 + 1;
            let delta = [value[0] - state.1[0], value[1] - state.1[1]];
            let mean = [
                state.1[0] + delta[0] / count as f64,
                state.1[1] + delta[1] / count as f64,
            ];
            let variance =
                state.2 + delta[0] * (value[0] - mean[0]) + delta[1] * (value[1] - mean[1]);
            state = (count, mean, variance);
        }

        Ok(state)
    }

    fn combine((n, l, lv): Self::State, (m, r, rv): Self::State) -> Self::State {
        let count = n + m;
        let delta = [r[0] - l[0], r[1] - l[1]];
        let weight = m as f64 / count as f64;
        let mean = [l[0] + delta[0] * weight, l[1] + delta[1] * weight];
        let variance = lv + rv + (delta[0] * delta[0] + delta[1] * delta[1]) * n as f64 * weight;
        (count, mean, variance)
    }

    fn finish((count, _, variance): Self::State) -> Result<f64> {
        Ok((variance / count as f64).sqrt())
    }

    fn empty() -> Result<f64> {
        Ok(f64::NAN)
    }

    fn finish_axis(state: Option<Self::State>) -> Result<f64> {
        state
            .map(<Self as ReduceOp<T>>::finish)
            .unwrap_or_else(<Self as ReduceOp<T>>::empty)
    }
}

/// Euclidean norm over logical values.
#[derive(Clone, Copy, Debug)]
pub struct Norm;

impl sealed::Sealed for Norm {}

impl<T: StatisticsElement> ReduceOp<T> for Norm {
    type State = f64;
    type Output = f64;

    fn partial(&self, values: Vec<T>) -> Result<f64> {
        Ok(values
            .into_iter()
            .map(|value| {
                let value = value.components();
                value[0] * value[0] + value[1] * value[1]
            })
            .sum())
    }

    fn combine(left: f64, right: f64) -> f64 {
        left + right
    }

    fn finish(state: f64) -> Result<f64> {
        Ok(state.sqrt())
    }

    fn empty() -> Result<f64> {
        Ok(0.0)
    }
}

/// Statistics include implicit zeros and share native batching and axis traversal.
pub trait TensorStatistics: TensorGeometry
where
    Self::DType: StatisticsElement,
{
    type MeanOutput: TensorRead<DType = <Self::DType as StatisticsElement>::Mean>;
    type StdOutput: TensorRead<DType = f64>;
    type NormOutput: TensorRead<DType = f64>;

    fn mean_all(&self) -> BoxFuture<'_, Result<<Self::DType as StatisticsElement>::Mean>>;

    fn std_all(&self) -> BoxFuture<'_, Result<f64>>;

    fn norm_all(&self) -> BoxFuture<'_, Result<f64>>;

    fn mean(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::MeanOutput>>;

    fn std(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::StdOutput>>;

    fn norm(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::NormOutput>>;
}

macro_rules! axis_constructor {
    ($output:ident, $method:ident, $op:ident $(; where [$($bounds:tt)+])?) => {
        type $output = ReduceView<Self, $op> $(where $($bounds)+)?;
        fn $method(&self, axes: Axes, keepdims: bool) -> BoxFuture<'_, Result<Self::$output>>
        $(where $($bounds)+)?
        {
            Box::pin(async move { ReduceView::new(self.clone(), axes, keepdims, $op) })
        }
    };
}

impl<E> TensorStatistics for E
where
    E: Expression + Clone,
    E::DType: StatisticsElement,
{
    axis_constructor!(MeanOutput, mean, Mean);
    axis_constructor!(StdOutput, std, StandardDeviation);
    axis_constructor!(NormOutput, norm, Norm);

    fn mean_all(&self) -> BoxFuture<'_, Result<<Self::DType as StatisticsElement>::Mean>> {
        Box::pin(terminal(self, Mean))
    }

    fn std_all(&self) -> BoxFuture<'_, Result<f64>> {
        Box::pin(terminal(self, StandardDeviation))
    }

    fn norm_all(&self) -> BoxFuture<'_, Result<f64>> {
        Box::pin(terminal(self, Norm))
    }
}

// The accumulator holds one partial result, never a list of batch results.
fn accumulate<T: TensorElement, O: ReduceOp<T>>(
    op: &O,
    state: &mut Option<O::State>,
    remaining: &mut u64,
    values: Vec<T>,
) -> Result<()> {
    *remaining = remaining.checked_sub(values.len() as u64).ok_or_else(|| {
        Error::InvalidLayout("reduction requests exceed selected cardinality".into())
    })?;
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

// Candidate requests cover distinct selected coordinates and may include numerical
// zeros. Their complement is known zero; repeated partial states account for that
// cardinality without constructing its values or changing the numerical combiner.
fn accumulate_zeros<T: TensorElement, O: ReduceOp<T>>(
    op: &O,
    state: &mut Option<O::State>,
    mut count: u64,
    zero: T,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    #[cfg(test)]
    crate::read_metrics::record(|m| m.reduction_calls += 1);
    let mut partial = op.partial(vec![zero])?;
    loop {
        if count & 1 != 0 {
            *state = Some(match *state {
                Some(previous) => O::combine(previous, partial),
                None => partial,
            });
        }
        count >>= 1;
        if count == 0 {
            return Ok(());
        }
        partial = O::combine(partial, partial);
    }
}

async fn terminal<E, O>(source: &E, op: O) -> Result<O::Output>
where
    E: Expression,
    E::DType: TensorElement,
    O: ReduceOp<E::DType>,
{
    let slice = crate::slice::Slice::full(source.shape())?;
    let mut remaining = slice.len();
    let requests = expression::traversal::selection(source, slice)?;
    // Numeric aggregates allow evaluation-order differences. Consume completed
    // batches immediately; boolean terminals retain logical ordered delivery.
    let batches =
        expression::evaluation_futures(source, requests).buffer_unordered(num_cpus::get().max(1));
    futures::pin_mut!(batches);
    let mut state = None;

    while let Some((_, batch)) = batches.try_next().await? {
        accumulate::<_, O>(&op, &mut state, &mut remaining, batch.values)?;
    }

    accumulate_zeros::<_, O>(&op, &mut state, remaining, source.implicit_zero())?;
    state.map(O::finish).unwrap_or_else(O::empty)
}

impl<E> TensorReduceAll for E
where
    E: Expression + TensorRead,
    E::DType: TensorElement,
{
    fn sum_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Sum>(self, Sum))
    }

    fn product_all(&self) -> BoxFuture<'_, Result<Self::DType>> {
        Box::pin(terminal::<_, Product>(self, Product))
    }

    fn min_all(&self) -> BoxFuture<'_, Result<Self::DType>>
    where
        E::DType: Real,
    {
        Box::pin(terminal::<_, Min>(self, Min))
    }

    fn max_all(&self) -> BoxFuture<'_, Result<Self::DType>>
    where
        E::DType: Real,
    {
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
            // indexed candidate traversal could change those short-circuit boundaries.
            let coords = request::linear_requests(self.shape())?;
            let mut batches = expression::ordered_batches(self, coords);

            while let Some((_, batch)) = batches.try_next().await? {
                if batch.values.into_iter().any(|v| v == E::DType::ZERO) {
                    return Ok(false);
                }
            }

            Ok(true)
        })
    }

    fn any(&self) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async move {
            // Logical batches preserve which errors precede a decisive value;
            // indexed candidate traversal could change those short-circuit boundaries.
            let coords = request::linear_requests(self.shape())?;
            let mut batches = expression::ordered_batches(self, coords);

            while let Some((_, batch)) = batches.try_next().await? {
                if batch.values.into_iter().any(|v| v != E::DType::ZERO) {
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
    fn new(source: S, axes: Axes, keepdims: bool, op: O) -> Result<Self> {
        crate::schema::validate_shape_dims(source.shape())?;
        let axes = crate::reduction_axes(source.ndim(), axes)?;

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
                    request::Axis::range(0, *dim)
                } else {
                    let i = if self.keepdims {
                        axis
                    } else {
                        let i = next;
                        next += 1;
                        i
                    };
                    request::Axis::range(coord[i], 1)
                }
            })
            .collect();

        Ok(axes)
    }
}

impl<E> TensorReduce for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
{
    axis_constructor!(SumOutput, sum, Sum);

    axis_constructor!(ProductOutput, product, Product);

    axis_constructor!(MinOutput, min, Min; where [E::DType: Real]);

    axis_constructor!(MaxOutput, max, Max; where [E::DType: Real]);
}

impl<S, O> TensorGeometry for ReduceView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    type DType = O::Output;

    fn dtype(&self) -> crate::NumberType {
        <O::Output as number_general::DType>::dtype()
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
}

impl<S, O> crate::expression::traversal::Plan for ReduceView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
}

impl<S, O> Expression for ReduceView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([self.source.expression_nodes()?])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.source.detach_sources(pending);
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        coords: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            // Output buffers are bounded by the current evaluation batch.
            let mut values = Vec::with_capacity(coords.len());
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
                    let mut remaining = slice.len();
                    let mut requests = expression::traversal::selection(&self.source, slice)?;
                    // Inner consumers never start buffered streams.
                    while let Some(request) = requests.try_next().await? {
                        let batch = context
                            .evaluate(&self.source, std::sync::Arc::new(request))
                            .await?;
                        accumulate::<_, O>(&self.op, &mut state, &mut remaining, batch.values)?;
                    }

                    accumulate_zeros::<_, O>(
                        &self.op,
                        &mut state,
                        remaining,
                        self.source.implicit_zero(),
                    )?;
                    values.push(O::finish_axis(state)?);
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

                let batch = context
                    .evaluate(
                        &self.source,
                        std::sync::Arc::new(BatchRequest::rectangles(rectangles)?),
                    )
                    .await?;
                let mut input = batch.values.into_iter();

                // The validated batch contains every nonempty group in full.
                for len in lengths {
                    #[cfg(test)]
                    crate::read_metrics::record(|m| m.reduction_calls += 1);
                    let group = input.by_ref().take(len as usize).collect();
                    values.push(O::finish(self.op.partial(group)?)?);
                }
            }

            Batch::from_values(values)
        })
    }
}

impl<S, O> TensorRead for ReduceView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    crate::expression::reader_members!(read_value, read_blocks, read_sparse_elements_in_order);
}

impl<S, O> TensorTransform for ReduceView<S, O>
where
    S: TensorGeometry,
    S::DType: TensorElement,
    O: ReduceOp<S::DType>,
{
    crate::mapping::transform_methods!();
}

#[cfg(test)]
#[path = "../tests/unit/reduce/tests.rs"]
mod tests;
