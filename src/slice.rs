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
mod tests {
    use futures::TryStreamExt;
    use ha_ndarray::{axes, shape};

    use crate::AxisRange;
    use number_general::DType;

    use super::*;
    use crate::test_support::{FsEntry, new_dir};
    use crate::{
        Layout, Tensor, TensorRead, TensorReduce, TensorReduceAll, TensorSchema, TensorSparseIndex,
        TensorTransform, TensorUnary, TensorWrite,
    };

    #[test]
    fn ordered_coverage_and_permanent_exhaustion() {
        let slice = Slice::new(
            &[3, 6000],
            vec![
                Axis::Selected(vec![2, 0, 2]),
                Axis::Span {
                    start: 1,
                    step: 2,
                    len: 2999,
                },
            ],
        )
        .unwrap();
        let mut requests = slice.requests();
        let mut actual = Vec::new();

        for request in requests.by_ref() {
            assert!(request.len() <= MAX_BATCH_ELEMENTS);
            actual.extend(request.coordinates(&[3, 6000]).unwrap());
        }

        let expected: Vec<_> = [2, 0, 2]
            .into_iter()
            .flat_map(|row| (0..2999).map(move |i| vec![row, 1 + i * 2]))
            .collect();
        assert_eq!(actual, expected);
        assert!(requests.next().is_none());
        assert!(requests.next().is_none());
        assert!(
            Slice::new(&[4], vec![Axis::range(4, 0)])
                .unwrap()
                .requests()
                .next()
                .is_none()
        );
        assert!(Slice::full(&[]).is_err());
        let mut scalar = Slice::new(&[], vec![]).unwrap().requests();
        assert_eq!(scalar.next().unwrap().len(), 1);
        assert!(scalar.next().is_none());
    }

    #[test]
    fn validation_and_huge_bounded_prefix() {
        assert!(Slice::new(&[2], vec![]).is_err());
        let slice = Slice::full(&[3]).unwrap();
        assert!(slice.intersect(&[]).is_err());
        assert!(slice.intersect(&[(2, 1)]).is_err());
        assert!(
            Slice::new(
                &[10],
                vec![Axis::Span {
                    start: 0,
                    step: 0,
                    len: 1
                }]
            )
            .is_err()
        );
        assert!(
            Slice::new(
                &[u64::MAX],
                vec![Axis::Span {
                    start: 1,
                    step: u64::MAX,
                    len: 2
                }]
            )
            .is_err()
        );
        assert!(Slice::new(&[3], vec![Axis::Selected(vec![3])]).is_err());
        assert!(Slice::full(&[u64::MAX, 2]).is_err());
        let slice = Slice::full(&[1_000_000_000, 1_000_000_000]).unwrap();
        let request = slice.requests().next().unwrap();
        assert_eq!(request.len(), MAX_BATCH_ELEMENTS);
        assert!(matches!(
            request.kind(),
            crate::request::RequestKind::Rectangles(_)
        ));
    }

    #[tokio::test]
    async fn huge_sparse_slices_keep_intermediate_support_and_release_guards() {
        let (root, dir) = new_dir("indexed_reduction").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir,
            TensorSchema::new(f32::dtype(), shape![2, 1_000_000_000]).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        tensor.write_value(&[0, 999_999_998], 0.2).await.unwrap();
        tensor.write_value(&[1, 3], 2.).await.unwrap();
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                assert_eq!(tensor.sum_all().await.unwrap(), 2.2);
                let round = tensor.view().round().await.unwrap();
                let result = round
                    .exp()
                    .await
                    .unwrap()
                    .sum(axes![1], false)
                    .await
                    .unwrap();
                assert_eq!(result.read_value(&[0]).await.unwrap(), 1.);
                let slice = tensor
                    .view()
                    .slice(vec![AxisRange::At(0), AxisRange::In(2, 1_000_000_000, 2)].into())
                    .unwrap();
                assert_eq!(slice.sum_all().await.unwrap(), 0.2);
                crate::read_metrics::CURRENT.with(|m| {
                    assert!(m.borrow().index_entries <= 4);
                    assert!(m.borrow().requested <= 32);
                });
            })
            .await;
        let mut requests = crate::expression::Expression::slice_requests(
            &tensor,
            Slice::full(&[2, 1_000_000_000]).unwrap(),
        )
        .unwrap();
        assert!(requests.try_next().await.unwrap().is_some());
        tensor.write_value(&[1, 4], 3.).await.unwrap();
        drop(requests);
        assert_eq!(tensor.sum_all().await.unwrap(), 5.2);
        crate::test_support::cleanup(&root).await;
    }

    #[tokio::test]
    async fn packed_dense_groups_share_one_block_and_no_output_mask() {
        let (root, dir) = new_dir("packed_reduction").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir,
            TensorSchema::new(f32::dtype(), shape![64, 4]).unwrap(),
            Layout::Dense,
            crate::schema::MAX_BLOCK_CAPACITY,
        )
        .await
        .unwrap();

        for row in 0..64 {
            tensor.write_value(&[row, 0], row as f32).await.unwrap();
        }

        let reduced = tensor.view().sum(axes![1], false).await.unwrap();
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                let request = BatchRequest::linear(0, 64).unwrap();
                let batch = crate::expression::evaluate_batch(&reduced, &request)
                    .await
                    .unwrap();
                assert!(batch.support.is_none());
                assert_eq!(batch.values, (0..64).map(|i| i as f32).collect::<Vec<_>>());
                crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().borrows, 1));
            })
            .await;
        crate::test_support::cleanup(&root).await;
    }

    #[tokio::test]
    async fn sparse_regions_filter_corruption_and_validate_grid_keys() {
        let (root, dir) = new_dir("slice_corruption").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(f32::dtype(), shape![2, 10_000]).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        tensor.write_value(&[0, 0], 1.).await.unwrap();
        tensor.write_value(&[0, 9_999], 2.).await.unwrap();
        let id = tensor.lookup_block_id(&[0, 0]).await.unwrap().unwrap();
        dir.read()
            .await
            .get_dir("blocks")
            .unwrap()
            .write()
            .await
            .delete(&id.to_string())
            .await;
        let slice = tensor
            .view()
            .slice(vec![AxisRange::At(0), AxisRange::In(5000, 10000, 1)].into())
            .unwrap();
        assert_eq!(slice.sum_all().await.unwrap(), 2.);
        assert!(tensor.sum_all().await.is_err());
        tensor.delete_row(vec![0, 0]).await.unwrap();
        tensor.upsert_block_id(vec![1, 0], id).await.unwrap();
        assert!(matches!(
            tensor.sum_all().await,
            Err(Error::InvalidLayout(_))
        ));
        crate::test_support::cleanup(&root).await;
    }

    #[tokio::test]
    async fn occupied_slices_respect_physical_aliases_and_live_sources() {
        let (root, dir) = new_dir("slice_alias").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir,
            TensorSchema::new(f32::dtype(), shape![2, 10000]).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        tensor.write_value(&[0, 0], 2.).await.unwrap();
        tensor.write_value(&[0, 1], 3.).await.unwrap();
        let id = tensor.lookup_block_id(&[0, 0]).await.unwrap().unwrap();
        tensor.upsert_block_id(vec![1, 1429], id).await.unwrap();
        assert_eq!(tensor.sum_all().await.unwrap(), 10.);
        let transposed = tensor.view().transpose(None).unwrap();
        assert_eq!(transposed.product_all().await.unwrap(), 36.);
        tensor.delete_row(vec![1, 1429]).await.unwrap();
        assert_eq!(transposed.sum_all().await.unwrap(), 5.);
        crate::test_support::cleanup(&root).await;
    }
}
