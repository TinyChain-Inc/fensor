use super::*;
use crate::request::{Axis, BatchRequest, Cartesian};

#[test]
fn mapped_visits_preserve_request_order_and_reset_scratch() {
    let shape = smallvec::smallvec![2, 3];
    let strides = schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides)
        .transpose(None)
        .unwrap()
        .flip(0)
        .unwrap();
    let requests = [
        BatchRequest::linear(0, 6).unwrap(),
        BatchRequest::rectangles(vec![
            Cartesian::new(vec![Axis::Selected(vec![2, 0, 2]), Axis::range(0, 2)]).unwrap(),
        ])
        .unwrap(),
        BatchRequest::explicit(vec![vec![1, 0], vec![0, 1], vec![1, 0]]).unwrap(),
        BatchRequest::explicit(vec![]).unwrap(),
        BatchRequest::linear(0, 0).unwrap(),
        BatchRequest::rectangles(vec![]).unwrap(),
    ];
    for request in requests {
        let expected: Vec<_> = request
            .coordinates(&mapping.shape)
            .unwrap()
            .iter()
            .map(|coord| mapping.resolve(coord, &shape, &strides).unwrap())
            .collect();
        let mut actual = Vec::new();
        mapping
            .visit_mapped(&request, &shape, &strides, |position, coord| {
                assert_eq!(position, actual.len());
                actual.push(coord.to_vec());
                coord.clear();
                coord.push(u64::MAX);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn mapped_visits_validate_first_and_stop_on_errors() {
    let shape = smallvec::smallvec![3];
    let mapping = CoordinateMap::identity(shape, &[1]);
    let invalid = BatchRequest::explicit(vec![vec![0], vec![3]]).unwrap();
    assert!(matches!(
        mapping.visit_mapped(&invalid, &[3], &[1], |_, _| {
            panic!("invalid request must fail before visiting")
        }),
        Err(Error::InvalidCoord(_))
    ));

    let request = BatchRequest::linear(0, 3).unwrap();
    let mut visited = Vec::new();
    let error = mapping.visit_mapped(&request, &[3], &[1], |position, _| {
        visited.push(position);
        if position == 1 {
            Err(Error::Unsupported("callback".into()))
        } else {
            Ok(())
        }
    });
    assert!(matches!(error, Err(Error::Unsupported(message)) if message == "callback"));
    assert_eq!(visited, [0, 1]);

    visited.clear();
    let error = mapping.visit_mapped(&request, &[1], &[1], |position, _| {
        visited.push(position);
        Ok(())
    });
    assert!(matches!(error, Err(Error::InvalidCoord(_))));
    assert_eq!(visited, [0]);
}

#[test]
fn mapped_visits_reuse_rank_sized_scratch_beyond_inline_capacity() {
    let mut shape = Shape::from_elem(1, PORTABLE_INLINE_RANK + 2);
    shape[0] = crate::expression::MAX_BATCH_ELEMENTS as u64;
    let strides = schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides);
    let request = BatchRequest::linear(0, crate::expression::MAX_BATCH_ELEMENTS).unwrap();
    let mut allocation = None;
    let mut count = 0;
    mapping
        .visit_mapped(&request, &shape, &strides, |position, coord| {
            assert_eq!(coord.len(), shape.len());
            assert_eq!(coord[0], position as u64);
            assert!(coord[1..].iter().all(|c| *c == 0));
            let current = (coord.as_ptr() as usize, coord.capacity());
            assert_eq!(*allocation.get_or_insert(current), current);
            count += 1;
            coord.clear();
            Ok(())
        })
        .unwrap();
    assert_eq!(count, request.len());
}

#[test]
fn compact_mapped_requests_match_coordinate_reference() {
    let shape = smallvec::smallvec![2, 3, 4];
    let strides = schema::contiguous_strides(&shape).unwrap();
    let base = CoordinateMap::identity(shape.clone(), &strides);
    for mapping in [
        base.clone(),
        base.clone().transpose(None).unwrap().flip(0).unwrap(),
        base.clone().reshape(smallvec::smallvec![6, 4]).unwrap(),
        base.clone()
            .slice(smallvec::smallvec![
                AxisRange::At(0),
                AxisRange::In(0, 3, 1),
                AxisRange::In(0, 4, 2)
            ])
            .unwrap()
            .broadcast(smallvec::smallvec![5, 3, 2])
            .unwrap(),
        base.slice(smallvec::smallvec![
            AxisRange::Of(vec![1, 0, 1]),
            AxisRange::In(0, 3, 1),
            AxisRange::Of(vec![3, 1, 3])
        ])
        .unwrap(),
    ] {
        let total = schema::checked_product(&mapping.shape).unwrap();
        let requests = [
            BatchRequest::linear(0, total as usize).unwrap(),
            BatchRequest::linear(1, (total - 2) as usize).unwrap(),
            BatchRequest::linear(total, 0).unwrap(),
            BatchRequest::explicit(vec![vec![0; mapping.shape.len()]; 3]).unwrap(),
            BatchRequest::rectangles(vec![
                Cartesian::new(
                    mapping
                        .shape
                        .iter()
                        .map(|&d| Axis::Selected(vec![d - 1, 0, d - 1]))
                        .collect(),
                )
                .unwrap(),
            ])
            .unwrap(),
        ];
        for request in requests {
            let expected: Vec<_> = request
                .coordinates(&mapping.shape)
                .unwrap()
                .iter()
                .map(|coord| mapping.resolve(coord, &shape, &strides).unwrap())
                .collect();
            let mapped = request.mapped(&mapping, &shape, &strides).unwrap();
            assert_eq!(mapped.coordinates(&shape).unwrap(), expected);
            // A nested wrapper consumes runs without allocating an explicit request.
            let identity = CoordinateMap::identity(shape.clone(), &strides);
            let nested = mapped.mapped(&identity, &shape, &strides).unwrap();
            assert_eq!(nested.coordinates(&shape).unwrap(), expected);
            assert!(matches!(
                nested.kind(),
                crate::request::RequestKind::FlatRuns(_)
            ));
        }
        assert!(
            BatchRequest::linear(total, 1)
                .unwrap()
                .mapped(&mapping, &shape, &strides)
                .is_err()
        );
    }
}

#[tokio::test]
async fn affine_mapping_retains_runs_across_batch_boundaries() {
    let shape = smallvec::smallvec![3, 4097];
    let strides = schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides)
        .flip(1)
        .unwrap();
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            for request in crate::request::linear_requests(&shape).unwrap() {
                let mapped = request.mapped(&mapping, &shape, &strides).unwrap();
                let crate::request::RequestKind::FlatRuns(runs) = mapped.kind() else {
                    panic!("compact runs")
                };
                assert!(runs.len() <= request.len().div_ceil(4097) + 1);
                assert_eq!(runs.iter().map(|r| r.len).sum::<usize>(), request.len());
            }
            crate::read_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!(m.expanded_coordinates, 0);
                assert_eq!(m.coordinate_resolutions, 0);
                assert!(m.mapped_runs > 0);
            });
        })
        .await;
}

#[test]
fn dense_base_flat_offset_diagnostic() {
    let shape = smallvec::smallvec![2, 3, 4];
    let strides = schema::contiguous_strides(&shape).unwrap();
    let map = CoordinateMap::identity(shape, &strides);
    assert_eq!(map.flat_offset(&[1, 2, 3]).unwrap(), 23);
}
