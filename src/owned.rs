//! Owned read-only expressions for dynamically composed application values.
//!
//! Only expression descriptions are erased here. Their storage sources retain
//! their concrete types, native guards, and caller-owned lifetimes.

use std::sync::Arc;

use futures::stream::BoxStream;

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::BatchRequest;
use crate::{
    BoxFuture, Layout, Result, TensorElement, TensorGeometry, TensorRead, TensorTransform,
    TensorViewSemantics,
};

/// An owned, read-only tensor value for runtime composition.
///
/// Concrete operations retain their operands and delegate numerical work; shared
/// consumption bounds stack use. Clones retain the same sources and their leases.
#[derive(Clone)]
pub struct TensorExpression<T: TensorElement> {
    // Detach before destruction: ordinary Arc/Box drop would recurse through the
    // expression even with stack-safe evaluation. None makes detached drop inert.
    source: Option<Arc<Owned<T>>>,
    mapping: CoordinateMap,
    dense: bool,
}

/// One admitted runtime description. Geometry and size never recurse through operands.
struct Owned<T: TensorElement> {
    expression: Box<dyn Expression<DType = T>>,
    nodes: usize,
    zero: T,
    layout: Layout,
}

pub trait Drain: Send {
    fn drain(self: Box<Self>, pending: &mut Vec<Box<dyn Drain>>);
}

impl<T: TensorElement> Drain for Arc<Owned<T>> {
    fn drain(self: Box<Self>, pending: &mut Vec<Box<dyn Drain>>) {
        // Unlike a failed try_unwrap followed by drop, into_inner remains safe when
        // different threads release the final shared operands concurrently.
        if let Some(mut owned) = Arc::into_inner(*self) {
            owned.expression.detach_sources(pending);
        }
    }
}

impl<T: TensorElement> Drop for TensorExpression<T> {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        self.detach_sources(&mut pending);

        while let Some(source) = pending.pop() {
            source.drain(&mut pending);
        }
    }
}

impl<T: TensorElement> TensorExpression<T> {
    pub fn new<E: Expression<DType = T> + 'static>(source: E) -> Result<Self> {
        let nodes = expression::traversal::node_count([source.expression_nodes()?])?;
        let layout = source.layout();
        let zero = source.implicit_zero();
        let strides = crate::contiguous_strides(source.shape())?;
        let mapping = CoordinateMap::identity(source.shape().into(), &strides);

        Ok(Self {
            source: Some(Arc::new(Owned {
                expression: Box::new(source),
                nodes,
                zero,
                layout,
            })),
            mapping,
            dense: false,
        })
    }

    /// Treat implicit sparse zeros as dense values without evaluating or copying storage.
    /// This expression retains its operands; previously cloned expressions are unchanged.
    pub fn into_dense(mut self) -> Self {
        self.dense = true;
        self
    }

    fn owned(&self) -> &Owned<T> {
        self.source.as_deref().expect("live expression operand")
    }

    fn source(&self) -> &dyn Expression<DType = T> {
        &*self.owned().expression
    }

    fn is_identity(&self, strides: &[u64]) -> bool {
        self.mapping.is_identity(self.source().shape(), strides)
    }

    /// A pull-driven owned stream. Dropping it releases all source handles.
    pub fn into_blocks(self) -> Result<BoxStream<'static, Result<Vec<T>>>> {
        expression::read_blocks(Arc::new(self))
    }

    pub fn into_sparse_elements(self) -> Result<crate::SparseElementStream<'static, T>> {
        let requests =
            expression::traversal::ordered(&self, crate::slice::Slice::full(self.shape())?)?;
        Ok(expression::sparse_stream(Arc::new(self), requests))
    }
}

impl<T: TensorElement> TensorGeometry for TensorExpression<T> {
    type DType = T;

    fn dtype(&self) -> crate::NumberType {
        <T as number_general::DType>::dtype()
    }

    fn layout(&self) -> Layout {
        if self.dense {
            return Layout::Dense;
        }

        match self.owned().layout {
            Layout::Sparse { axis: Some(_) }
                if self.mapping.base_offset != 0
                    || self.mapping.shape.as_slice() != self.source().shape()
                    || !self.mapping.is_c_contiguous() =>
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

impl<T: TensorElement> TensorViewSemantics for TensorExpression<T> {
    fn is_base_tensor(&self) -> bool {
        false
    }
}

impl<T: TensorElement> crate::expression::traversal::Plan for TensorExpression<T> {
    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        Ok(expression::traversal::Preferred::Sources(vec![(
            self.source(),
            shape,
        )]))
    }

    fn selection_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Selection<'_>> {
        if matches!(self.layout(), Layout::Dense) {
            return Ok(expression::traversal::Selection::Ready(slice.stream()));
        }
        let strides = crate::contiguous_strides(self.source().shape())?;

        Ok(if self.is_identity(&strides) {
            expression::traversal::Selection::Source(self.source(), slice)
        } else {
            expression::traversal::Selection::Ready(slice.stream())
        })
    }

    fn ordered_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Ordered<'_>> {
        if matches!(self.layout(), Layout::Dense) {
            return Ok(expression::traversal::Ordered::Ready(slice.stream()));
        }
        let strides = crate::contiguous_strides(self.source().shape())?;

        Ok(if self.is_identity(&strides) {
            expression::traversal::Ordered::Sources(vec![(self.source(), slice)])
        } else {
            expression::traversal::Ordered::Ready(slice.stream())
        })
    }
}

impl<T: TensorElement> Expression for TensorExpression<T> {
    fn implicit_zero(&self) -> Self::DType {
        self.owned().zero
    }

    fn expression_nodes(&self) -> Result<usize> {
        Ok(self.owned().nodes)
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn Drain>>) {
        if let Some(source) = self.source.take() {
            pending.push(Box::new(source));
        }
    }

    fn build<'a>(
        &'a self,
        context: expression::Context<'a>,
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<T>>> {
        Box::pin(async move {
            let strides = crate::contiguous_strides(self.source().shape())?;
            if self.is_identity(&strides) {
                return context.batch(self.source(), request).await;
            }

            let mapped = request.mapped(&self.mapping, self.source().shape(), &strides)?;
            context
                .batch(self.source(), std::sync::Arc::new(mapped))
                .await
        })
    }
}

impl<T: TensorElement> TensorRead for TensorExpression<T> {
    crate::expression::reader_members!(
        read_value,
        read_blocks,
        read_coordinate_blocks,
        read_sparse_elements_in_order
    );
}

impl<T: TensorElement> TensorTransform for TensorExpression<T> {
    crate::mapping::transform_methods!(clone_mapping);
}
