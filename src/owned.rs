//! Owned read-only expressions for dynamically composed application values.
//!
//! Only expression descriptions are erased here. Their storage sources retain
//! their concrete types, native guards, and caller-owned lifetimes.

use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, stream::BoxStream};

use crate::expression::{self, Batch, Expression};
use crate::mapping::CoordinateMap;
use crate::request::{self, BatchRequest};
use crate::{
    BoxFuture, Layout, Result, TensorElement, TensorGeometry, TensorRead, TensorTransform,
    TensorViewSemantics,
};

#[derive(Clone)]
pub struct TensorExpression<T: TensorElement> {
    source: Handle<T>,
    mapping: CoordinateMap,
}

/// One admitted runtime description. Geometry and size never recurse through operands.
struct Owned<T: TensorElement> {
    expression: Box<dyn Expression<DType = T>>,
    layout: Layout,
    nodes: usize,
}

#[derive(Clone)]
struct Handle<T: TensorElement>(Option<Arc<Owned<T>>>);

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

impl<T: TensorElement> Handle<T> {
    fn owned(&self) -> &Owned<T> {
        self.0.as_deref().expect("live expression operand")
    }

    fn detach(&mut self, pending: &mut Vec<Box<dyn Drain>>) {
        if let Some(source) = self.0.take() {
            pending.push(Box::new(source));
        }
    }
}

impl<T: TensorElement> Drop for Handle<T> {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        self.detach(&mut pending);

        while let Some(source) = pending.pop() {
            source.drain(&mut pending);
        }
    }
}

impl<T: TensorElement> TensorExpression<T> {
    pub fn new<E: Expression<DType = T> + 'static>(source: E) -> Result<Self> {
        let nodes = expression::traversal::node_count([source.expression_nodes()?])?;
        let layout = source.layout();
        let strides = crate::contiguous_strides(source.shape())?;
        let mapping = CoordinateMap::identity(source.shape().into(), &strides);
        Ok(Self {
            source: Handle(Some(Arc::new(Owned {
                expression: Box::new(source),
                layout,
                nodes,
            }))),
            mapping,
        })
    }

    fn source(&self) -> &dyn Expression<DType = T> {
        &*self.source.owned().expression
    }

    fn is_identity(&self, strides: &[u64]) -> bool {
        self.mapping.is_identity(self.source().shape(), strides)
    }

    /// A pull-driven owned stream. Dropping it releases all source handles.
    pub fn into_blocks(self) -> Result<BoxStream<'static, Result<Vec<T>>>> {
        let requests = request::linear_requests(self.shape())?;
        Ok(
            expression::ordered_requests(Arc::new(self), futures::stream::iter(requests.map(Ok)))
                .map_ok(|(_, batch)| batch.values)
                .boxed(),
        )
    }

    pub fn into_sparse_elements(self) -> Result<crate::SparseElementStream<'static, T>> {
        let requests =
            expression::traversal::support(&self, crate::slice::Slice::full(self.shape())?)?;
        Ok(expression::sparse_stream(Arc::new(self), requests))
    }
}

impl<T: TensorElement> TensorGeometry for TensorExpression<T> {
    type DType = T;

    fn dtype(&self) -> crate::NumberType {
        <T as number_general::DType>::dtype()
    }

    fn layout(&self) -> Layout {
        self.source.owned().layout
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

impl<T: TensorElement> Expression for TensorExpression<T> {
    fn expression_nodes(&self) -> Result<usize> {
        Ok(self.source.owned().nodes)
    }

    fn detach_sources(&mut self, pending: &mut Vec<Box<dyn Drain>>) {
        self.source.detach(pending);
    }

    fn preferred_step<'a>(
        &'a self,
        shape: &'a [u64],
    ) -> Result<expression::traversal::Preferred<'a>> {
        Ok(expression::traversal::Preferred::Sources(vec![Box::new(
            move || self.source().preferred_step(shape),
        )]))
    }

    fn selection_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Selection<'_>> {
        let strides = crate::contiguous_strides(self.source().shape())?;
        Ok(if self.is_identity(&strides) {
            expression::traversal::Selection::Source(Box::new(move || {
                self.source().selection_step(slice)
            }))
        } else {
            expression::traversal::Selection::Ready(slice.stream())
        })
    }

    fn support_step(
        &self,
        slice: crate::slice::Slice,
    ) -> Result<expression::traversal::Support<'_>> {
        let strides = crate::contiguous_strides(self.source().shape())?;
        Ok(if self.is_identity(&strides) {
            expression::traversal::Support::Sources(vec![(
                0,
                Box::new(move || self.source().support_step(slice)),
            )])
        } else {
            expression::traversal::Support::Ready(slice.stream())
        })
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
    crate::mapping::transform_methods!();
}
