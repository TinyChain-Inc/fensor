//! Matrix-unary views delegate numerical expression construction to their source.

use crate::expression::{Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::BatchRequest;
use crate::{
    Axes, BoxFuture, Error, Layout, Result, Shape, Strides, TensorElement, TensorGeometry,
    TensorMatrixUnary, TensorRead, TensorTransform, TensorViewSemantics,
};

/// A lazy, read-only diagonal projection. Transforms address its output geometry.
/// Sparse support is inherited only from diagonal coordinates, including supported
/// intermediate zeros. Complete sparse scans traverse the logical diagonal length.
///
/// Diagonals cannot be written, even after transforms:
///
/// ```compile_fail
/// use fensor::{Tensor, TensorFileEntry, TensorMatrixUnary, TensorTransform, TensorWrite};
/// fn writable<V: TensorWrite>(_: &V) {}
/// async fn example<F: TensorFileEntry<f32>>(tensor: &Tensor<F, f32>) {
///     writable(&tensor.view().diag().await.unwrap().flip(0).unwrap());
/// }
/// ```
///
/// They do not expose a storage schema:
///
/// ```compile_fail
/// use fensor::{Tensor, TensorArray, TensorFileEntry, TensorMatrixUnary};
/// fn storage<V: TensorArray>(_: &V) {}
/// async fn example<F: TensorFileEntry<u8>>(tensor: &Tensor<F, u8>) {
///     storage(&tensor.view().diag().await.unwrap());
/// }
/// ```
///
/// View descriptions clone without cloning filesystem adapters:
///
/// ```
/// use fensor::{Result, Tensor, TensorFileEntry, TensorMatrixUnary, TensorTransform};
/// async fn example<F: TensorFileEntry<f64>>(tensor: &Tensor<F, f64>) -> Result<()> {
///     let diagonal = tensor.view().mt().await?.diag().await?;
///     let _ = diagonal.clone().flip(0)?;
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct DiagView<Source> {
    source: Source,
    // Original output geometry and transform metadata are rank-sized.
    shape: Shape,
    strides: Strides,
    mapping: CoordinateMap,
}

impl<E> TensorMatrixUnary for E
where
    E: Expression + TensorRead + TensorTransform + Clone,
    E::DType: TensorElement,
{
    type TransposeOutput = Self;

    type DiagOutput = DiagView<Self>;

    fn mt(&self) -> BoxFuture<'_, Result<Self::TransposeOutput>> {
        Box::pin(async move {
            let rank = self.ndim();
            if rank < 2 {
                return Err(Error::InvalidLayout(format!(
                    "matrix transpose requires rank >= 2: {:?}",
                    self.shape()
                )));
            }
            let mut axes: Axes = (0..rank).collect();
            axes.swap(rank - 2, rank - 1);
            self.clone().transpose(Some(axes))
        })
    }

    fn diag(&self) -> BoxFuture<'_, Result<Self::DiagOutput>> {
        Box::pin(async move {
            let rank = self.ndim();
            if rank < 2 || self.shape()[rank - 2] != self.shape()[rank - 1] {
                return Err(Error::InvalidLayout(format!(
                    "diagonal requires rank >= 2 and square final dimensions: {:?}",
                    self.shape()
                )));
            }
            let shape = Shape::from_slice(&self.shape()[..rank - 1]);
            let strides = crate::schema::contiguous_strides(&shape)?;
            let mapping = CoordinateMap::identity(shape.clone(), &strides);
            Ok(DiagView {
                source: self.clone(),
                shape,
                strides,
                mapping,
            })
        })
    }
}

#[cfg(feature = "complex")]
impl<E> crate::TensorMatrixUnaryComplex for E
where
    E: Expression + TensorRead + TensorTransform + Clone,
    E::DType: TensorElement + ha_ndarray::Complex,
    <E::DType as ha_ndarray::Complex>::Real: TensorElement,
{
    type HermitianOutput = crate::UnaryView<Self, crate::unary::Conj>;

    fn mh(&self) -> BoxFuture<'_, Result<Self::HermitianOutput>> {
        use crate::TensorComplex;

        Box::pin(async move { self.mt().await?.conj().await })
    }
}

impl<S: TensorGeometry> TensorGeometry for DiagView<S> {
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

impl<S: TensorGeometry> TensorViewSemantics for DiagView<S> {
    fn is_base_tensor(&self) -> bool {
        false
    }
}

impl<S> Expression for DiagView<S>
where
    S: Expression,
    S::DType: TensorElement,
{
    fn expression_nodes(&self) -> Result<usize> {
        crate::expression::traversal::node_count([self.source.expression_nodes()?])
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn crate::owned::Drain>>) {
        self.source.detach_sources(pending);
    }

    fn build<'a>(
        &'a self,
        context: crate::expression::Context<'a>,
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<Self::DType>>> {
        Box::pin(async move {
            // Correlated diagonal axes require bounded explicit source coordinates.
            let mut coordinates = Vec::with_capacity(request.len());
            self.mapping
                .visit_mapped(&request, &self.shape, &self.strides, |_, mapped| {
                    let index = mapped[mapped.len() - 1];
                    mapped.push(index);
                    coordinates.push(mapped.to_vec());
                    Ok(())
                })?;
            let request = BatchRequest::explicit(coordinates)?;
            // Preserve the source's lazy ndarray expression and support. This
            // projection adds no evaluation or buffering boundary.
            context
                .batch(&self.source, std::sync::Arc::new(request))
                .await
        })
    }
}

impl<S> TensorRead for DiagView<S>
where
    S: Expression,
    S::DType: TensorElement,
{
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<S> TensorTransform for DiagView<S>
where
    S: TensorGeometry,
{
    crate::mapping::transform_methods!();
}

#[cfg(test)]
mod tests {
    use crate::expression::{self, MAX_BATCH_ELEMENTS};
    use crate::request::BatchRequest;
    use crate::test_support::{cleanup, fixture};
    use crate::{Layout, TensorGeometry, TensorMatMul, TensorMatrixUnary, TensorUnary};

    #[tokio::test]
    async fn projection_preserves_support_in_one_bounded_source_request() {
        let (root, tensor) = fixture::source(
            "diag_batch",
            smallvec::smallvec![2, 2],
            Layout::Sparse { axis: None },
            2,
            4096,
            [0.2f32, 7., 0., 0.],
        )
        .await;
        let product = tensor.view().matmul(&tensor.view()).await.unwrap();
        assert!(
            crate::expression::traversal::preferred(&product, product.shape())
                .unwrap()
                .is_some()
        );
        let diagonal = product.diag().await.unwrap();
        assert!(
            crate::expression::traversal::preferred(&diagonal, diagonal.shape())
                .unwrap()
                .is_none()
        );
        let view = tensor
            .view()
            .round()
            .await
            .unwrap()
            .diag()
            .await
            .unwrap()
            .exp()
            .await
            .unwrap();
        let request = BatchRequest::explicit(
            (0..MAX_BATCH_ELEMENTS)
                .map(|i| vec![(i % 2) as u64])
                .collect(),
        )
        .unwrap();
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                let batch = expression::evaluate_batch(&view, &request).await.unwrap();
                let support = batch.support.unwrap();
                for (i, value) in batch.values.into_iter().enumerate() {
                    let expected = u8::from(i % 2 == 0);
                    assert_eq!(value, expected as f32);
                    assert_eq!(support[i], expected);
                }
                crate::read_metrics::CURRENT.with(|metrics| {
                    let metrics = metrics.borrow();
                    assert_eq!(metrics.requested, MAX_BATCH_ELEMENTS);
                    assert_eq!(metrics.logical_payload_reads, 2);
                });
            })
            .await;
        cleanup(&root).await;
    }
}
