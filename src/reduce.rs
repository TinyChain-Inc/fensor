//! Bounded reductions over logical values, including implicit sparse zeros.

use std::future::Future;

use futures::TryStreamExt;
use ha_ndarray::host::Host;
use ha_ndarray::statistics::BlockReduce;
pub use ha_ndarray::statistics::{
    MeanOp as Mean, NormOp as Norm, StandardDeviationOp as StandardDeviation,
};
use ha_ndarray::{NDArrayReduceAll, Number, Real};

use crate::expression::{Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::BatchRequest;
use crate::schema::Coord;
use crate::{
    Axes, BoxFuture, Error, Layout, Result, Shape, TensorElement, TensorGeometry, TensorRead,
    TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorTransform, expression, request,
};

mod sealed {
    pub trait Sealed {}
}

/// A sealed reduction delegating numerical rules to ha-ndarray.
pub trait ReduceOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    type State: Copy + Send + Sync;

    type Output: TensorElement;

    fn partial(&self, values: Vec<T>) -> Result<Self::State>;

    fn combine(left: Self::State, right: Self::State) -> Result<Self::State>;

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

    fn combine(left: T, right: T) -> Result<T> {
        Ok(<T as Number>::add(left, right))
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

    fn combine(left: T, right: T) -> Result<T> {
        Ok(<T as Number>::mul(left, right))
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

    fn combine(left: T, right: T) -> Result<T> {
        Ok(<T as Real>::min(left, right))
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

    fn combine(left: T, right: T) -> Result<T> {
        Ok(<T as Real>::max(left, right))
    }

    fn empty() -> Result<T> {
        Err(Error::Unsupported("max_all has empty input".into()))
    }
}

macro_rules! block_reduction {
    ($(#[$attr:meta])* $op:ident) => {
        impl sealed::Sealed for $op {}

        impl<T: TensorElement> ReduceOp<T> for $op
        where
            Self: BlockReduce<Host, T, Block = [T]>,
            <Self as BlockReduce<Host, T>>::Output: TensorElement,
        {
            type State = <Self as BlockReduce<Host, T>>::State;

            type Output = <Self as BlockReduce<Host, T>>::Output;

            $(#[$attr])*
            fn partial(&self, values: Vec<T>) -> Result<Self::State> {
                Ok(<Self as BlockReduce<Host, T>>::partial(self, &values)?)
            }

            fn combine(left: Self::State, right: Self::State) -> Result<Self::State> {
                Ok(<Self as BlockReduce<Host, T>>::merge(left, right)?)
            }

            fn finish(state: Self::State) -> Result<Self::Output> {
                Ok(<Self as BlockReduce<Host, T>>::finish(state))
            }

            fn empty() -> Result<Self::Output> {
                <Self as ReduceOp<T>>::finish(Self::State::default())
            }

            fn finish_axis(state: Option<Self::State>) -> Result<Self::Output> {
                <Self as ReduceOp<T>>::finish(state.unwrap_or_default())
            }
        }
    };
}

// Inlining the mean kernel slowed unrelated reductions in paired consumer runs.
block_reduction!(
    #[inline(never)]
    Mean
);

block_reduction!(StandardDeviation);
block_reduction!(Norm);

/// Statistics include implicit zeros and share native batching and axis traversal.
pub trait TensorStatistics: TensorGeometry<DType: TensorElement> {
    type MeanOutput: TensorRead<DType: TensorElement>;

    type StdOutput: TensorRead<DType = f64>;

    type NormOutput: TensorRead<DType = f64>;

    fn mean_all(&self) -> BoxFuture<'_, Result<<Self::MeanOutput as TensorGeometry>::DType>>;

    fn std_all(&self) -> BoxFuture<'_, Result<f64>>;

    fn norm_all(&self) -> BoxFuture<'_, Result<f64>>;

    fn mean(
        &self,
        axes: Axes,
        keepdims: bool,
    ) -> impl Future<Output = Result<Self::MeanOutput>> + Send + '_;

    fn std(
        &self,
        axes: Axes,
        keepdims: bool,
    ) -> impl Future<Output = Result<Self::StdOutput>> + Send + '_;

    fn norm(
        &self,
        axes: Axes,
        keepdims: bool,
    ) -> impl Future<Output = Result<Self::NormOutput>> + Send + '_;
}

macro_rules! axis_constructor {
    ($output:ident, $method:ident, $op:ident $(; where [$($bounds:tt)+])?) => {
        type $output = ReduceView<Self, $op> $(where $($bounds)+)?;

        async fn $method(&self, axes: Axes, keepdims: bool) -> Result<Self::$output>
        $(where $($bounds)+)?
        {
            ReduceView::new(self.clone(), axes, keepdims, $op)
        }
    };
}

impl<E> TensorStatistics for E
where
    E: Expression + Clone,
    E::DType: TensorElement,
    Mean: ReduceOp<E::DType>,
    StandardDeviation: ReduceOp<E::DType, Output = f64>,
    Norm: ReduceOp<E::DType, Output = f64>,
{
    axis_constructor!(MeanOutput, mean, Mean);

    axis_constructor!(StdOutput, std, StandardDeviation);

    axis_constructor!(NormOutput, norm, Norm);

    fn mean_all(&self) -> BoxFuture<'_, Result<<Self::MeanOutput as TensorGeometry>::DType>> {
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
            Some(previous) => O::combine(previous, partial)?,
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
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    #[cfg(test)]
    crate::read_metrics::record(|m| m.reduction_calls += 1);

    let mut partial = op.partial(vec![T::ZERO])?;
    loop {
        if count & 1 != 0 {
            *state = Some(match *state {
                Some(previous) => O::combine(previous, partial)?,
                None => partial,
            });
        }

        count >>= 1;
        if count == 0 {
            return Ok(());
        }

        partial = O::combine(partial, partial)?;
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
    let batches = expression::evaluate_requests(source, requests);
    futures::pin_mut!(batches);
    let mut state = None;

    while let Some((_, batch)) = batches.try_next().await? {
        accumulate::<_, O>(&op, &mut state, &mut remaining, batch.values)?;
    }

    accumulate_zeros::<_, O>(&op, &mut state, remaining)?;
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
        self.mapping.shape()
    }

    fn layout(&self) -> Layout {
        match self.source.layout() {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: None },
        }
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

                    accumulate_zeros::<_, O>(&self.op, &mut state, remaining)?;

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
    crate::expression::reader_members!(read_value, read_blocks, read_sparse_elements);
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
