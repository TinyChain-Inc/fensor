#[cfg(test)]
#[path = "../tests/unit/view/tests.rs"]
mod tests;

use crate::mapping::{AxisContrib, CoordinateMap};
use crate::schema::Layout;
use crate::{
    BoxFuture, Error, Result, TensorArray, TensorElement, TensorGeometry, TensorRead, TensorSource,
    TensorTransform, TensorViewSemantics, TensorWrite,
};

/// A geometric view of filesystem-backed tensor storage.
#[derive(Clone)]
pub struct TensorView<S> {
    tensor: S,
    mapping: CoordinateMap,
}

impl<S: TensorArray> TensorView<S>
where
    S::DType: TensorElement,
{
    pub fn new(tensor: S) -> Self {
        let mapping = CoordinateMap::identity(tensor.shape().into(), tensor.strides());
        Self { tensor, mapping }
    }

    pub fn source(&self) -> &S {
        &self.tensor
    }

    /// Replace a source with identical base geometry, retaining this view's mapping.
    pub fn with_source<R: TensorArray<DType = S::DType>>(self, source: R) -> Result<TensorView<R>> {
        if self.tensor.schema() != source.schema() || self.tensor.layout() != source.layout() {
            return Err(Error::InvalidLayout(
                "replacement source geometry differs".into(),
            ));
        }

        Ok(TensorView {
            tensor: source,
            mapping: self.mapping,
        })
    }

    fn validate_geometry(&self, geometry: &crate::StorageGeometry) -> Result<()> {
        if geometry.schema() != self.tensor.schema() || geometry.layout() != self.tensor.layout() {
            return Err(Error::InvalidLayout(
                "storage geometry differs from view source".into(),
            ));
        }

        Ok(())
    }

    /// Map one bounded row-major value batch to logical-block updates.
    /// This plans addresses only; callers enforce write-through and reserve storage.
    pub fn plan_updates(
        &self,
        geometry: &crate::StorageGeometry,
        start: u64,
        values: Vec<S::DType>,
    ) -> Result<crate::BlockUpdates<S::DType>> {
        self.validate_geometry(geometry)?;
        let request = crate::request::BatchRequest::linear(start, values.len())?;
        crate::storage::plan_updates(geometry, Some(&self.mapping), &request, values)
    }

    /// Consume an owned expression into bounded destination updates in completion order.
    /// Dropping the stream releases its source handles. This acquires no write permission.
    pub fn updates_from(
        &self,
        geometry: &crate::StorageGeometry,
        source: crate::TensorExpression<S::DType>,
    ) -> Result<futures::stream::BoxStream<'static, Result<crate::BlockUpdates<S::DType>>>> {
        use futures::StreamExt;
        self.validate_geometry(geometry)?;
        if self.shape() != source.shape() {
            return Err(Error::InvalidLayout(
                "update source and destination shapes differ".into(),
            ));
        }

        let geometry = geometry.clone();
        let mapping = self.mapping.clone();
        Ok(
            crate::expression::completion_batches(std::sync::Arc::new(source))?
                .map(move |result| {
                    let (request, batch) = result?;
                    crate::storage::plan_updates(&geometry, Some(&mapping), &request, batch.values)
                })
                .boxed(),
        )
    }

    /// Conservative logical storage-block interval without enumerating the selection.
    /// Irregular mappings may reserve gaps; explicit gather metadata is inspected once.
    pub fn logical_block_range(
        &self,
        geometry: &crate::StorageGeometry,
    ) -> Result<std::ops::Range<u64>> {
        self.validate_geometry(geometry)?;
        let Some((lo, hi)) = self.mapping.flat_bounds()? else {
            return Ok(0..0);
        };

        let mut lower = Vec::with_capacity(self.tensor.shape().len());
        let mut upper = Vec::with_capacity(self.tensor.shape().len());

        for (&len, &stride) in self.tensor.shape().iter().zip(self.tensor.strides()) {
            let (start, end) = (lo / stride, hi / stride);
            if start / len == end / len {
                lower.push(start % len);
                upper.push(end % len);
            } else {
                lower.push(0);
                upper.push(len - 1);
            }
        }

        Ok(geometry.block_position(&lower)?.0..geometry.block_position(&upper)?.0 + 1)
    }

    pub fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        self.mapping
            .resolve(coord, self.tensor.shape(), self.tensor.strides())
    }
}

impl<S: TensorArray> TensorGeometry for TensorView<S>
where
    S::DType: TensorElement,
{
    type DType = S::DType;

    fn dtype(&self) -> number_general::NumberType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        match self.tensor.layout() {
            Layout::Sparse { .. }
                if !self
                    .mapping
                    .is_identity(self.tensor.shape(), self.tensor.strides()) =>
            {
                Layout::Sparse { axis: None }
            }
            layout => layout,
        }
    }

    fn shape(&self) -> &[u64] {
        &self.mapping.shape
    }
}

impl<S: TensorArray> TensorViewSemantics for TensorView<S>
where
    S::DType: TensorElement,
{
    fn is_base_tensor(&self) -> bool {
        self.mapping.base_offset == 0
            && self.mapping.shape.as_slice() == self.tensor.shape()
            && self.mapping.is_c_contiguous()
    }

    fn supports_write_through(&self) -> bool {
        !self
            .mapping
            .axes
            .iter()
            .any(|a| matches!(a, AxisContrib::Gather(_) | AxisContrib::Broadcast(_)))
    }
}

impl<S: TensorArray> TensorTransform for TensorView<S>
where
    S::DType: TensorElement,
{
    crate::mapping::transform_methods!();
}

impl<S: TensorSource> TensorRead for TensorView<S>
where
    S::DType: TensorElement,
{
    crate::expression::reader_members!(
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let request = crate::request::BatchRequest::point(coord);
            Ok(self.read_batch(&request).await?[0])
        })
    }
}

impl<S: TensorSource + TensorWrite> TensorWrite for TensorView<S>
where
    S::DType: TensorElement,
{
    fn write_value<'a>(
        &'a self,
        coord: &'a [u64],
        value: Self::DType,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if !self.supports_write_through() {
                return Err(Error::Unsupported(
                    "this view does not support write-through to the base tensor".to_string(),
                ));
            }

            let base_coord = self.resolve_base_coord(coord)?;
            self.tensor.write_value(&base_coord, value).await
        })
    }
}

impl<S: TensorSource> TensorView<S>
where
    S::DType: TensorElement,
{
    pub(crate) async fn read_batch(
        &self,
        request: &crate::request::BatchRequest,
    ) -> Result<Vec<S::DType>> {
        self.tensor
            .read_storage(crate::StorageRead {
                request,
                mapping: Some(&self.mapping),
            })
            .await
    }

    pub(crate) fn ordered_storage_requests(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<crate::slice::Requests<'static>> {
        match self
            .mapping
            .storage_slice(self.tensor.shape(), self.tensor.strides())?
        {
            Some(mapping) => crate::storage::ordered_requests(&self.tensor, slice, mapping),
            None => Ok(slice.stream()),
        }
    }

    pub(crate) fn slice_requests_for_storage(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<crate::slice::Requests<'_>> {
        if slice.len() <= crate::expression::MAX_BATCH_ELEMENTS as u64
            || matches!(self.layout(), Layout::Dense)
        {
            return Ok(slice.stream());
        }

        match self
            .mapping
            .storage_slice(self.tensor.shape(), self.tensor.strides())?
        {
            Some(mapping) => crate::storage::slice_requests(&self.tensor, slice, Some(mapping)),
            None => Ok(slice.stream()),
        }
    }
}
