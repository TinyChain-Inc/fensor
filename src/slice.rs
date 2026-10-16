//! Validated, rank-preserving selections. Only emitted requests are batch-sized.

use futures::{StreamExt, stream::BoxStream};

use crate::expression::MAX_BATCH_ELEMENTS;
use crate::request::{Axis, BatchRequest, Cartesian};
use crate::schema::Coord;
use crate::{Error, Result};

pub(crate) type Requests<'a> = BoxStream<'a, Result<BatchRequest>>;

#[derive(Clone)]
pub struct Slice {
    // Rank-sized metadata plus caller-supplied explicit selections.
    shape: crate::Shape,
    pub(crate) axes: Vec<Axis>,
    len: u64,
}

impl Slice {
    pub(crate) fn new(shape: &[u64], axes: Vec<Axis>) -> Result<Self> {
        if axes.len() != shape.len() {
            return Err(Error::InvalidCoord("slice rank mismatch".into()));
        }

        for (axis, &dim) in axes.iter().zip(shape) {
            axis.validate(dim)?;
        }

        let len = if axes.iter().any(|axis| axis.len() == 0) {
            0
        } else {
            axes.iter()
                .try_fold(1u64, |n, axis| n.checked_mul(axis.len()))
                .ok_or_else(|| Error::InvalidLayout("slice cardinality overflow".into()))?
        };

        Ok(Self {
            shape: shape.iter().copied().collect(),
            axes,
            len,
        })
    }

    pub(crate) fn full(shape: &[u64]) -> Result<Self> {
        crate::schema::validate_shape_dims(shape)?;
        Self::new(
            shape,
            shape.iter().map(|&dim| Axis::range(0, dim)).collect(),
        )
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn rectangle(self) -> Result<Cartesian> {
        Cartesian::new(self.axes)
    }

    pub(crate) fn requests(self) -> SliceRequests {
        let mut capacity = 1;
        let mut chunks = crate::Shape::from_elem(1, self.axes.len());

        for (axis, chunk) in self.axes.iter().zip(&mut chunks).rev() {
            *chunk = axis.len().min(MAX_BATCH_ELEMENTS as u64 / capacity);
            capacity *= (*chunk).max(1);
        }
        SliceRequests {
            origins: Coord::from_elem(0, self.axes.len()),
            chunks,
            done: self.len == 0,
            slice: self,
        }
    }

    pub(crate) fn stream<'a>(self) -> Requests<'a> {
        futures::stream::iter(self.requests().map(Ok)).boxed()
    }

    /// Clip each axis without changing explicit-selection order or multiplicity.
    pub(crate) fn intersect(&self, bounds: &[(u64, u64)]) -> Result<Self> {
        if bounds.len() != self.axes.len() || bounds.iter().any(|(lo, hi)| lo > hi) {
            return Err(Error::InvalidCoord(
                "invalid slice intersection bounds".into(),
            ));
        }

        let axes = self
            .axes
            .iter()
            .zip(bounds)
            .map(|(axis, &(lo, hi))| match axis {
                Axis::Span { start, step, len } => {
                    let first = lo.saturating_sub(*start).div_ceil(*step).min(*len);
                    let end = hi.saturating_sub(*start).div_ceil(*step).min(*len);
                    let count = end.saturating_sub(first);
                    Axis::Span {
                        start: if count == 0 {
                            *start
                        } else {
                            start + step * first
                        },
                        step: *step,
                        len: count,
                    }
                }
                Axis::Selected(indices) => Axis::Selected(
                    indices
                        .iter()
                        .copied()
                        .filter(|&i| i >= lo && i < hi)
                        .collect(),
                ),
            })
            .collect();
        Self::new(&self.shape, axes)
    }
}

pub(crate) struct SliceRequests {
    slice: Slice,
    chunks: crate::Shape,
    origins: Coord,
    done: bool,
}

impl SliceRequests {
    pub(crate) fn next_rectangle(&mut self) -> Option<Cartesian> {
        if self.done {
            return None;
        }

        let axes = self
            .slice
            .axes
            .iter()
            .zip(&self.origins)
            .zip(&self.chunks)
            .map(|((axis, &origin), &chunk)| {
                let len = chunk.min(axis.len() - origin);
                match axis {
                    Axis::Span { start, step, .. } => Axis::Span {
                        start: start + step * origin,
                        step: *step,
                        len,
                    },
                    Axis::Selected(values) => {
                        Axis::Selected(values[origin as usize..(origin + len) as usize].to_vec())
                    }
                }
            })
            .collect();
        // Constructor validation proves endpoints, cardinality, and every advance.
        let rectangle = Cartesian::new(axes).expect("validated slice chunk");
        self.done = true;

        for axis in (0..self.origins.len()).rev() {
            let remaining = self.slice.axes[axis].len() - self.origins[axis];
            if self.chunks[axis] < remaining {
                self.origins[axis] += self.chunks[axis];
                self.done = false;
                break;
            }

            self.origins[axis] = 0;
        }

        Some(rectangle)
    }
}

impl Iterator for SliceRequests {
    type Item = BatchRequest;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_rectangle().map(|rectangle| {
            BatchRequest::rectangles(vec![rectangle]).expect("one bounded rectangle")
        })
    }
}

impl std::iter::FusedIterator for SliceRequests {}

#[cfg(test)]
#[path = "../tests/unit/slice/tests.rs"]
mod tests;
