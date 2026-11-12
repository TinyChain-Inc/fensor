use number_general::DType;

use super::*;

#[test]
fn block_consumers_share_order_padding_and_rank_scratch() {
    assert!(TensorSchema::new(f64::dtype(), Shape::new()).is_err());
    let mut high = Shape::from_elem(1, crate::PORTABLE_INLINE_RANK + 2);
    high[0] = 3;
    let mut high_block = high.clone();
    high_block[0] = 2;

    for (shape, block) in [
        (smallvec::smallvec![1], smallvec::smallvec![1]),
        (smallvec::smallvec![5], smallvec::smallvec![3]),
        (smallvec::smallvec![6], smallvec::smallvec![3]),
        (smallvec::smallvec![2, 3], smallvec::smallvec![3, 4]),
        (smallvec::smallvec![3, 5], smallvec::smallvec![2, 3]),
        (high, high_block),
    ] {
        let layouts = [Layout::Dense, Layout::Sparse { axis: None }]
            .into_iter()
            .chain((0..shape.len()).map(|axis| Layout::Sparse { axis: Some(axis) }));
        for layout in layouts {
            let mut block = block.clone();
            if let Layout::Sparse { axis } = layout {
                let prefix = axis.map_or(shape.len(), |axis| axis + 1);
                block[..prefix].fill(1);
            }

            let geometry = StorageGeometry::new(
                TensorSchema::new(f64::dtype(), shape.clone()).unwrap(),
                layout,
                block,
            )
            .unwrap();

            crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
                geometry.validate_id(0).unwrap();
                geometry.validate_id(geometry.block_count() - 1).unwrap();
                assert!(matches!(
                    geometry.validate_id(geometry.block_count()),
                    Err(Error::InvalidCoord(_))
                ));
                crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
            });

            for id in 0..geometry.block_count() {
                let bounds = geometry.block_bounds(id).unwrap();
                let expected: Vec<_> = crate::schema::row_major_coords(&shape)
                    .unwrap()
                    .filter(|coord| {
                        coord
                            .iter()
                            .zip(&bounds)
                            .all(|(c, (lo, hi))| c >= lo && c < hi)
                    })
                    .map(|coord| {
                        let offset = coord
                            .iter()
                            .zip(&bounds)
                            .zip(&geometry.storage.block_schema.strides)
                            .map(|((c, (lo, _)), s)| (c - lo) * s)
                            .sum::<u64>() as usize;
                        (offset, coord)
                    })
                    .collect();
                assert_eq!(
                    geometry.block_offsets(id).unwrap().collect::<Vec<_>>(),
                    expected.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                );
                let mut values = vec![-0.0; geometry.block_len()];
                for (i, (offset, _)) in expected.iter().enumerate() {
                    values[*offset] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][i % 3];
                }
                crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
                    geometry.validate_block(id, &values).unwrap();
                    crate::read_metrics::CURRENT.with(|m| {
                        let metrics = m.borrow();
                        let walks = usize::from(expected.len() != geometry.block_len());
                        assert_eq!(metrics.block_bounds, walks);
                        assert_eq!(metrics.padding_walks, walks);
                    });
                });
                for offset in 0..values.len() {
                    if expected.iter().any(|(valid, _)| *valid == offset) {
                        continue;
                    }
                    for invalid in [1.0, f64::NAN, f64::INFINITY] {
                        values[offset] = invalid;
                        assert!(matches!(
                            geometry.validate_block(id, &values),
                            Err(Error::InvalidLayout(_))
                        ));
                    }
                    values[offset] = -0.0;
                }
                let mut cursor = BlockCursor::new(&geometry, id).unwrap();
                let address = cursor.coord.as_ptr();

                for (offset, coord) in &expected {
                    let (actual, scratch) = cursor.current().unwrap();
                    assert_eq!(actual, *offset);
                    assert_eq!(scratch, coord);
                    assert_eq!(scratch.as_ptr(), address);
                    cursor.advance();
                }
                assert!(cursor.current().is_none());
                assert!(cursor.current().is_none());
            }
            assert!(geometry.block_offsets(geometry.block_count()).is_err());
            assert!(geometry.block_offsets(u64::MAX).is_err());
            crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
                for id in [geometry.block_count(), u64::MAX] {
                    assert!(matches!(
                        geometry.validate_block::<f64>(id, &[]),
                        Err(Error::InvalidCoord(_))
                    ));
                }
                for id in [0, geometry.block_count() - 1] {
                    for len in [geometry.block_len() - 1, geometry.block_len() + 1] {
                        assert!(matches!(
                            geometry.validate_block(id, &vec![1.0; len]),
                            Err(Error::InvalidLayout(message))
                                if message == "invalid logical block length"
                        ));
                    }
                }
                crate::read_metrics::CURRENT.with(|m| {
                    let metrics = m.borrow();
                    assert_eq!(metrics.block_bounds, 0);
                    assert_eq!(metrics.padding_walks, 0);
                });
            });
        }
    }
}

#[test]
fn block_validation_handles_u64_edge_without_overflow() {
    let geometry = StorageGeometry::new(
        TensorSchema::new(f64::dtype(), smallvec::smallvec![u64::MAX]).unwrap(),
        Layout::Dense,
        smallvec::smallvec![2],
    )
    .unwrap();
    let last = geometry.block_count() - 1;

    for (id, values, walks) in [
        (last - 1, [f64::NAN, f64::INFINITY], 0),
        (last, [f64::NAN, -0.0], 1),
    ] {
        crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
            geometry.validate_block(id, &values).unwrap();
            crate::read_metrics::CURRENT.with(|m| {
                let metrics = m.borrow();
                assert_eq!(metrics.block_bounds, walks);
                assert_eq!(metrics.padding_walks, walks);
            });
        });
    }

    for invalid in [1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            geometry.validate_block(last, &[f64::NAN, invalid]),
            Err(Error::InvalidLayout(message)) if message == "nonzero logical block padding"
        ));
    }
}

#[tokio::test]
async fn preferred_regions_skip_gaps_and_include_sparse_zeros() {
    use futures::TryStreamExt;

    use crate::test_support::{FsEntry, new_dir};
    use crate::{Tensor, TensorExpression, TensorGeometry, TensorRead, TensorTransform};

    let (_root, dir) = new_dir("preferred_regions").await;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), smallvec::smallvec![1_000_000_001, 3]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        2,
    )
    .await
    .unwrap();
    let view = tensor
        .view()
        .slice(smallvec::smallvec![
            crate::AxisRange::In(0, 1_000_000_001, 1_000_000_000),
            crate::AxisRange::In(0, 3, 1),
        ])
        .unwrap()
        .transpose(None)
        .unwrap();

    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let requests = crate::expression::traversal::preferred(&view, view.shape())
                .unwrap()
                .unwrap();
            let mut actual = Vec::new();
            for request in requests {
                actual.extend(request.into_coordinates(view.shape()).unwrap());
            }
            let mut sorted = actual.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                crate::schema::row_major_coords(view.shape())
                    .unwrap()
                    .collect::<Vec<_>>()
            );
            assert_ne!(actual, sorted);
            assert_eq!(
                actual,
                vec![
                    vec![0, 0],
                    vec![1, 0],
                    vec![2, 0],
                    vec![0, 1],
                    vec![1, 1],
                    vec![2, 1]
                ]
            );
            // Adjacent column blocks share one region per selected storage row.
            crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 2));
        })
        .await;

    let mut blocks = view.read_coordinate_blocks().unwrap();
    let mut count = 0;
    while let Some((coordinates, values)) = blocks.try_next().await.unwrap() {
        assert_eq!(coordinates.len(), values.len());
        assert!(values.iter().all(|&value| value == 0.));
        count += values.len();
    }
    assert_eq!(count, 6);

    let identity = TensorExpression::new(tensor.view().transpose(None).unwrap())
        .unwrap()
        .transpose(None)
        .unwrap();
    crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
        assert!(
            crate::expression::traversal::preferred(&identity, identity.shape())
                .unwrap()
                .is_none()
        );
        crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 0));
    });

    for consume in [false, true] {
        let lease = std::sync::Arc::new(());
        let released = std::sync::Arc::downgrade(&lease);
        let source = PlanningSource {
            tensor: tensor.clone(),
            lease,
        };
        let slice = crate::slice::Slice::full(source.shape()).unwrap();
        let mut requests = slice_requests(&source, slice, None).unwrap();
        drop(source);
        assert!(released.upgrade().is_some());
        if consume {
            assert!(requests.try_next().await.unwrap().is_none());
            assert!(released.upgrade().is_none());
        }
        drop(requests);
        assert!(released.upgrade().is_none());
    }

    // Matching axis positions alone do not make a smaller selection an identity.
    let first_row = tensor
        .view()
        .slice(smallvec::smallvec![
            crate::AxisRange::In(0, 1, 1),
            crate::AxisRange::In(0, 3, 1),
        ])
        .unwrap();
    crate::read_metrics::CURRENT.sync_scope(Default::default(), || {
        let requests = crate::expression::traversal::preferred(&first_row, first_row.shape())
            .unwrap()
            .unwrap();
        let actual: Vec<_> = requests
            .flat_map(|request| request.into_coordinates(first_row.shape()).unwrap())
            .collect();
        assert_eq!(actual, vec![vec![0, 0], vec![0, 1], vec![0, 2]]);
        crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().block_bounds, 1));
    });
}

// A native setup borrow must not duplicate the source; the occupied stream owns
// its independent lease, including before its first poll.
struct PlanningSource {
    tensor: crate::Tensor<crate::test_support::FsEntry, f64>,
    lease: std::sync::Arc<()>,
}

impl Clone for PlanningSource {
    fn clone(&self) -> Self {
        panic!("sparse planning cloned its borrowed source")
    }
}

impl crate::TensorGeometry for PlanningSource {
    type DType = f64;

    fn dtype(&self) -> crate::NumberType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }

    fn shape(&self) -> &[u64] {
        self.tensor.shape()
    }
}

impl crate::TensorArray for PlanningSource {
    fn schema(&self) -> &TensorSchema {
        self.tensor.schema()
    }

    fn strides(&self) -> &[u64] {
        self.tensor.strides()
    }
}

impl TensorSource for PlanningSource {
    fn storage_geometry(&self) -> StorageGeometry {
        self.tensor.storage_geometry()
    }

    fn read_logical_block(&self, id: u64) -> crate::BoxFuture<'_, Result<Vec<f64>>> {
        self.tensor.read_logical_block(id)
    }

    fn occupied_blocks(&self, range: std::ops::Range<u64>) -> BoxStream<'static, Result<u64>> {
        let lease = self.lease.clone();
        let keys = self.tensor.occupied_blocks(range);
        Box::pin(keys.map(move |key| {
            let _ = &lease;
            key
        }))
    }
}
