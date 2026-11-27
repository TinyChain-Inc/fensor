#[cfg(test)]
#[path = "../tests/unit/view/tests.rs"]
mod tests;

use crate::mapping::CoordinateMap;
use crate::schema::Layout;
use crate::{
    BoxFuture, Error, Result, TensorArray, TensorElement, TensorGeometry, TensorRead, TensorSource,
    TensorTransform,
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
        let mapping = CoordinateMap::identity(tensor.shape().into(), tensor.schema().strides());
        Self { tensor, mapping }
    }

    fn is_identity(&self) -> bool {
        let schema = self.tensor.schema();
        self.mapping.is_identity(schema.shape(), schema.strides())
    }

    pub fn source(&self) -> &S {
        &self.tensor
    }

    /// Replace a source with identical base geometry, retaining this view's mapping.
    pub fn with_source<R: TensorArray<DType = S::DType>>(
        &self,
        source: R,
    ) -> Result<TensorView<R>> {
        if self.tensor.schema() != source.schema() || self.tensor.layout() != source.layout() {
            return Err(Error::InvalidLayout(
                "replacement source geometry differs".into(),
            ));
        }

        Ok(TensorView {
            tensor: source,
            mapping: self.mapping.clone(),
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

        let lower = geometry.storage.blocks.position(lo).block_id;
        let upper = geometry.storage.blocks.position(hi).block_id;
        Ok(lower..upper + 1)
    }

    pub fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        self.mapping
            .resolve(coord, self.tensor.shape(), self.tensor.schema().strides())
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
            Layout::Sparse { .. } if !self.is_identity() => Layout::Sparse { axis: None },
            layout => layout,
        }
    }

    fn shape(&self) -> &[u64] {
        self.mapping.shape()
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
    crate::expression::reader_members!(read_blocks, read_coordinate_blocks, read_sparse_elements);

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            crate::validate::validate_coord(self.shape(), coord)?;
            let flat = self.mapping.flat_offset(coord)?;
            let blocks = &self.tensor.storage_geometry().storage.blocks;
            let flat = u64::try_from(flat)
                .ok()
                .filter(|flat| *flat < blocks.size)
                .ok_or_else(|| {
                    crate::Error::InvalidCoord("mapped coordinate out of bounds".into())
                })?;

            let position = blocks.position(flat);
            self.tensor
                .read_block_value(position.block_id, position.offset_in_block)
                .await
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
        crate::storage::read_batch(&self.tensor, request, Some(&self.mapping)).await
    }
}

impl<S: TensorSource> crate::expression::traversal::Plan for TensorView<S>
where
    S::DType: TensorElement,
{
    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
        mapping: Option<crate::mapping::StorageSlice>,
    ) -> Result<crate::expression::traversal::Preferred<'a>> {
        let mapping = if self.is_identity() {
            mapping
        } else {
            let Some(local) = self
                .mapping
                .storage_slice(self.tensor.shape(), self.tensor.schema().strides())?
            else {
                return Ok(crate::expression::traversal::Preferred::Ready(None));
            };

            Some(local.compose(mapping)?)
        };

        crate::storage::preferred_requests(&self.tensor, shape, mapping)
            .map(crate::expression::traversal::Preferred::Ready)
    }

    fn selection_step(
        &self,
        slice: crate::slice::Slice,
        mapping: Option<crate::mapping::StorageSlice>,
    ) -> Result<crate::expression::traversal::Selection<'_>> {
        if slice.len() <= crate::MAX_BATCH_ELEMENTS as u64 || matches!(self.layout(), Layout::Dense)
        {
            return Ok(crate::expression::traversal::Selection::Ready(
                slice.stream(),
            ));
        }

        let mapping = if self.is_identity() {
            mapping
        } else {
            let Some(local) = self
                .mapping
                .storage_slice(self.tensor.shape(), self.tensor.schema().strides())?
            else {
                return Ok(crate::expression::traversal::Selection::Ready(
                    slice.stream(),
                ));
            };

            Some(local.compose(mapping)?)
        };

        crate::storage::slice_requests(&self.tensor, slice, mapping)
            .map(crate::expression::traversal::Selection::Ready)
    }

    fn ordered_step(
        &self,
        slice: crate::slice::Slice,
        mapping: Option<crate::mapping::StorageSlice>,
    ) -> Result<crate::expression::traversal::Ordered<'_>> {
        if slice.len() <= crate::MAX_BATCH_ELEMENTS as u64 || matches!(self.layout(), Layout::Dense)
        {
            return Ok(crate::expression::traversal::Ordered::Ready(slice.stream()));
        }

        let mapping = if self.is_identity() {
            mapping
        } else {
            let Some(local) = self
                .mapping
                .storage_slice(self.tensor.shape(), self.tensor.schema().strides())?
            else {
                return Ok(crate::expression::traversal::Ordered::Ready(slice.stream()));
            };

            Some(local.compose(mapping)?)
        };

        crate::storage::ordered_requests(&self.tensor, slice, mapping)
            .map(crate::expression::traversal::Ordered::Ready)
    }
}
