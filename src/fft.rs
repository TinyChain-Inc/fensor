//! Bounded last-axis Fourier evaluation. Each pack contains complete axis groups;
//! backend workspace is bounded by the capped transform size. Nested transforms
//! may recompute groups; there is no result cache or implicit persistence.

use std::collections::BTreeMap;

use ha_ndarray::{ArrayAccess, NDArrayFourier, NDArrayRead, NDArrayTransform, Number};

use crate::expression::{self, Batch, Expression, MAX_BATCH_ELEMENTS};
use crate::mapping::CoordinateMap;
use crate::request::{Axis, BatchRequest};
use crate::schema::Coord;
use crate::slice::Slice;
use crate::{
    Axes, BoxFuture, Error, Layout, Result, Shape, Strides, TensorElement, TensorFourier,
    TensorGeometry, TensorRead, TensorTransform, TensorViewSemantics,
};

mod sealed {
    pub trait Sealed {}
}

/// Sealed backend Fourier operation; numerical work belongs to ha-ndarray.
pub trait FourierOp<T: TensorElement>: sealed::Sealed + Clone + Send + Sync {
    const NAME: &'static str;

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>>;
}

/// Unnormalized forward Fourier transform.
#[derive(Clone, Copy, Debug)]
pub struct Fft;

/// Unnormalized inverse Fourier transform.
#[derive(Clone, Copy, Debug)]
pub struct Ifft;

impl sealed::Sealed for Fft {}

impl sealed::Sealed for Ifft {}

impl<T: TensorElement> FourierOp<T> for Fft
where
    ArrayAccess<'static, T>: NDArrayFourier<DType = T, Platform = ha_ndarray::Platform>,
    <ArrayAccess<'static, T> as NDArrayFourier>::Output: Into<ha_ndarray::Accessor<'static, T>>,
    T: ha_ndarray::Complex,
{
    const NAME: &'static str = "fft";

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.fft()?))
    }
}

impl<T: TensorElement> FourierOp<T> for Ifft
where
    ArrayAccess<'static, T>: NDArrayFourier<DType = T, Platform = ha_ndarray::Platform>,
    <ArrayAccess<'static, T> as NDArrayFourier>::Output: Into<ha_ndarray::Accessor<'static, T>>,
    T: ha_ndarray::Complex,
{
    const NAME: &'static str = "ifft";

    fn apply(&self, array: ArrayAccess<'static, T>) -> Result<ArrayAccess<'static, T>> {
        Ok(ArrayAccess::from(array.ifft()?))
    }
}

/// Lazy, read-only Fourier view with bounded complete-group evaluation.
///
/// Output transforms change coordinate mapping, never the source transform axis.
/// Even one requested frequency reads its complete input group. Missing sparse
/// inputs are numerical zeros.
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorFourier};
/// async fn real<F: TensorFileEntry<f32>>(t: &Tensor<F, f32>) {
///     t.view().fft().await.unwrap();
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex32, Tensor, TensorFileEntry, TensorFourier, TensorWrite};
/// fn writable<V: TensorWrite>(_: &V) {}
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) {
///     writable(&t.view().fft().await.unwrap());
/// }
/// ```
///
/// ```
/// use fensor::{complex::Complex32, Result, Tensor, TensorFileEntry, TensorFourier,
///     TensorTransform, TensorMatrixUnaryComplex};
/// async fn example<F: TensorFileEntry<Complex32>>(t: &Tensor<F, Complex32>) -> Result<()> {
///     let view = t.view().fft().await?;
///     let _ = view.clone().flip(0)?.mh().await?;
///     Ok(())
/// }
/// ```
///
/// ```compile_fail
/// use fensor::{complex::Complex64, Tensor, TensorArray, TensorFileEntry, TensorFourier};
/// fn storage<V: TensorArray>(_: &V) {}
/// async fn example<F: TensorFileEntry<Complex64>>(t: &Tensor<F, Complex64>) {
///     storage(&t.view().ifft().await.unwrap());
/// }
/// ```
#[derive(Clone)]
pub struct FourierView<Source, Op> {
    source: Source,
    op: Op,
    // Rank-sized original geometry and output mapping.
    shape: Shape,
    strides: Strides,
    mapping: CoordinateMap,
}

fn validate_axes(shape: &[u64], count: usize, operation: &str) -> Result<()> {
    if shape.len() < count || shape.contains(&0) {
        return Err(Error::InvalidLayout(format!(
            "{operation} requires rank >= {count} and nonempty dimensions: {shape:?}"
        )));
    }
    crate::schema::contiguous_strides(shape)?;

    for &len in &shape[shape.len() - count..] {
        if len > MAX_BATCH_ELEMENTS as u64 {
            return Err(Error::Unsupported(format!(
                "{operation} axis length {len} exceeds limit {MAX_BATCH_ELEMENTS}"
            )));
        }
    }

    Ok(())
}

impl<S: Expression + Clone, O: FourierOp<S::DType>> FourierView<S, O>
where
    S::DType: TensorElement,
{
    fn new(source: &S, op: O) -> Result<Self> {
        validate_axes(source.shape(), 1, O::NAME)?;
        let shape = Shape::from_slice(source.shape());
        let strides = crate::schema::contiguous_strides(&shape)?;
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        Ok(Self {
            source: source.clone(),
            op,
            shape,
            strides,
            mapping,
        })
    }
}

impl<E: Expression + Clone> TensorFourier for E
where
    E::DType: TensorElement,
    Fft: FourierOp<E::DType>,
    Ifft: FourierOp<E::DType>,
{
    type FftOutput = FourierView<Self, Fft>;
    type IfftOutput = FourierView<Self, Ifft>;

    fn fft(&self) -> BoxFuture<'_, Result<Self::FftOutput>> {
        Box::pin(async move { FourierView::new(self, Fft) })
    }

    fn ifft(&self) -> BoxFuture<'_, Result<Self::IfftOutput>> {
        Box::pin(async move { FourierView::new(self, Ifft) })
    }
}

/// Unnormalized forward transform of the final two axes, preserving batch axes.
/// Both axis lengths must fit the execution limit; nested groups may be recomputed.
pub async fn fft2<E>(source: &E) -> Result<FourierView<FourierView<E, Fft>, Fft>>
where
    E: Expression + Clone,
    E::DType: TensorElement,
    Fft: FourierOp<E::DType>,
{
    validate_axes(source.shape(), 2, "fft2")?;
    let mut axes: Axes = (0..source.ndim()).collect();
    axes.swap(source.ndim() - 2, source.ndim() - 1);
    let first = FourierView::new(source, Fft)?.transpose(Some(axes.clone()))?;
    FourierView::new(&first, Fft)?.transpose(Some(axes))
}

/// Unnormalized inverse transform of the final two axes.
/// Inverse-after-forward scales by the product of the final two dimensions.
pub async fn ifft2<E>(source: &E) -> Result<FourierView<FourierView<E, Ifft>, Ifft>>
where
    E: Expression + TensorTransform + Clone,
    E::DType: TensorElement,
    Ifft: FourierOp<E::DType>,
{
    validate_axes(source.shape(), 2, "ifft2")?;
    let mut axes: Axes = (0..source.ndim()).collect();
    axes.swap(source.ndim() - 2, source.ndim() - 1);
    let transposed = source.clone().transpose(Some(axes.clone()))?;
    let first = FourierView::new(&transposed, Ifft)?.transpose(Some(axes))?;
    FourierView::new(&first, Ifft)
}

impl<S: TensorGeometry, O: Send + Sync> TensorGeometry for FourierView<S, O> {
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

impl<S: TensorGeometry, O: Send + Sync> TensorViewSemantics for FourierView<S, O> {
    fn is_base_tensor(&self) -> bool {
        false
    }
}

// Group keys and scatter entries contain at most one entry per requested output.
type Groups = BTreeMap<Coord, Vec<(usize, usize)>>;

impl<S, O> FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
{
    fn groups(&self, request: &BatchRequest) -> Result<Groups> {
        let mut groups = Groups::new();
        self.mapping.visit_mapped(
            request,
            &self.shape,
            &self.strides,
            |destination, mapped| {
                let frequency = mapped.pop().expect("validated Fourier rank") as usize;
                groups
                    .entry(mapped.clone())
                    .or_default()
                    .push((destination, frequency));
                Ok(())
            },
        )?;
        Ok(groups)
    }
}

impl<S, O> Expression for FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
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
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            let groups = self.groups(&request)?;
            // Axis length is checked before narrowing. Each pack and output vector
            // holds at most MAX_BATCH_ELEMENTS elements, independently of tensor size.
            let width = *self.shape.last().expect("validated Fourier rank") as usize;
            let mut groups = groups.into_iter().peekable();
            let mut values = vec![S::DType::ZERO; request.len()];

            while groups.peek().is_some() {
                let pack: Vec<_> = groups.by_ref().take(MAX_BATCH_ELEMENTS / width).collect();
                let rectangles = pack
                    .iter()
                    .map(|(prefix, _)| {
                        let axes = prefix
                            .iter()
                            .map(|&i| Axis::range(i, 1))
                            .chain(std::iter::once(Axis::range(0, width as u64)))
                            .collect();
                        Slice::new(self.source.shape(), axes)?.rectangle()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let input = BatchRequest::rectangles(rectangles)?;
                let batch = context
                    .evaluate(&self.source, std::sync::Arc::new(input.clone()))
                    .await?;
                let array = expression::batch_array(batch.values)?
                    .reshape(ha_ndarray::shape![pack.len(), width])?;
                let transformed = self
                    .op
                    .apply(ArrayAccess::from(array))?
                    .buffer()?
                    .to_slice()?
                    .into_vec();
                if transformed.len() != input.len() {
                    return Err(Error::InvalidLayout(format!(
                        "{} output: expected {} elements, got {}",
                        O::NAME,
                        input.len(),
                        transformed.len()
                    )));
                }

                for (group, (_, destinations)) in pack.into_iter().enumerate() {
                    let start = group * width;
                    for (destination, frequency) in destinations {
                        values[destination] = transformed[start + frequency];
                    }
                }
            }

            Batch::from_values(values)
        })
    }
}

impl<S, O> TensorRead for FourierView<S, O>
where
    S: Expression,
    S::DType: TensorElement,
    O: FourierOp<S::DType>,
{
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<S, O: Send + Sync> TensorTransform for FourierView<S, O>
where
    S: TensorGeometry,
{
    crate::mapping::transform_methods!();
}

#[cfg(test)]
#[path = "../tests/unit/fft/tests.rs"]
mod tests;
