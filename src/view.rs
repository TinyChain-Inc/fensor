#[cfg(test)]
mod tests;

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{Axes, Range, Shape};

use crate::mapping::{AxisContrib, CoordinateMap};
use crate::schema::Layout;
use crate::{
    BoxFuture, Error, Result, Tensor, TensorArray, TensorElement, TensorFileEntry, TensorGeometry,
    TensorRead, TensorTransform, TensorViewSemantics, TensorWrite,
};

/// A geometric view of filesystem-backed tensor storage.
pub struct TensorView<'t, FE, T: TensorElement> {
    tensor: &'t Tensor<FE, T>,
    mapping: CoordinateMap,
}

impl<FE, T: TensorElement> Clone for TensorView<'_, FE, T> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor,
            mapping: self.mapping.clone(),
        }
    }
}

impl<'t, FE, T: TensorElement> TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
{
    pub(crate) fn new_identity(tensor: &'t Tensor<FE, T>) -> Self {
        Self {
            tensor,
            mapping: CoordinateMap::identity(tensor.shape().into(), tensor.strides()),
        }
    }

    pub fn flat_offset(&self, coord: &[u64]) -> Result<i64> {
        self.mapping.flat_offset(coord)
    }

    pub(crate) async fn read_batch(
        &self,
        request: &crate::request::BatchRequest,
    ) -> Result<Vec<T>> {
        self.tensor.read_batch(request, Some(&self.mapping)).await
    }

    pub(crate) fn slice_requests_for_storage(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<crate::slice::Requests<'_>> {
        if slice.len() <= crate::expression::MAX_BATCH_ELEMENTS
            || matches!(self.layout(), Layout::Dense)
        {
            return Ok(slice.stream());
        }
        match self
            .mapping
            .storage_slice(self.tensor.shape(), self.tensor.strides())?
        {
            Some(mapping) => self.tensor.storage_slice_requests(slice, Some(mapping)),
            None => Ok(slice.stream()),
        }
    }

    fn resolve_base_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        self.mapping
            .resolve(coord, self.tensor.shape(), self.tensor.strides())
    }
}

impl<'t, FE, T> TensorGeometry for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type DType = T;

    fn dtype(&self) -> number_general::NumberType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }

    fn shape(&self) -> &[usize] {
        &self.mapping.shape
    }
}

impl<'t, FE, T> TensorViewSemantics for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
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

impl<'t, FE, T> TensorTransform for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn reshape(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.reshape(shape)?,
        })
    }

    fn broadcast(self, shape: Shape) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.broadcast(shape)?,
        })
    }

    fn slice(self, range: Range) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.slice(range)?,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.transpose(permutation)?,
        })
    }

    fn flip(self, axis: usize) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.flip(axis)?,
        })
    }

    fn squeeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.squeeze(axes)?,
        })
    }

    fn unsqueeze(self, axes: Axes) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor,
            mapping: self.mapping.unsqueeze(axes)?,
        })
    }
}

impl<'t, FE, T> TensorRead for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn read_blocks(&self) -> Result<crate::ValueBlockStream<'_, Self::DType>> {
        let requests = crate::request::linear_requests(self.shape())?;
        Ok(crate::expression::ordered_batches(self, requests)
            .map_ok(|(_, batch)| batch.values)
            .boxed())
    }

    fn read_coordinate_blocks(&self) -> Result<crate::CoordinateBlockStream<'_, Self::DType>> {
        crate::expression::coordinate_blocks(self)
    }

    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, Result<Self::DType>> {
        Box::pin(async move {
            let base_coord = self.resolve_base_coord(coord)?;
            self.tensor.read_value(&base_coord).await
        })
    }
}

impl<'t, FE, T> TensorWrite for TensorView<'t, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
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
