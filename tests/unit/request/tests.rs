use super::*;

#[test]
fn flat_runs_validate_before_cursor_use() {
    for (start, step, len, total) in [
        (-1, 1, 2, 2),
        (0, -1, 2, 2),
        (2, 1, 2, 2),
        (0, i128::MAX, 3, 3),
        (0, 1, 0, 0),
        (0, 1, 2, 3),
    ] {
        let request = BatchRequest {
            kind: RequestKind::FlatRuns(vec![Progression {
                output: 0,
                start,
                step,
                len,
            }]),
            len: total,
        };
        assert!(request.cursor(&[3]).is_err());
    }

    let request = BatchRequest {
        kind: RequestKind::FlatRuns(vec![Progression {
            output: 0,
            start: 0,
            step: 0,
            len: 3,
        }]),
        len: 3,
    };
    assert_eq!(request.coordinates(&[1]).unwrap(), vec![vec![0]; 3]);
    let empty = BatchRequest {
        kind: RequestKind::FlatRuns(vec![]),
        len: 0,
    };
    assert!(empty.coordinates(&[1]).unwrap().is_empty());
}

#[test]
fn shared_segments_preserve_order_bounds_and_exhaustion() {
    for axis in [
        Axis::range(8, 0),
        Axis::range(8, 7),
        Axis::Selected(vec![]),
        Axis::Selected(vec![9, 7, 5, 5, 5, 8, 2]),
        Axis::Selected(vec![u64::MAX, 0, u64::MAX]),
    ] {
        let values: Vec<_> = axis
            .segments()
            .flat_map(|(start, step, len)| {
                (0..len).map(move |i| (start as i128 + step * i as i128) as u64)
            })
            .collect();
        assert_eq!(
            values,
            (0..axis.len()).map(|i| axis.at(i)).collect::<Vec<_>>()
        );
    }

    let mut scratch = Coord::new();

    for (flat, offset, len) in linear_segments(5, 17, 7).unwrap() {
        decode_flat(flat, &[4, 7], &mut scratch).unwrap();
        assert_eq!(flat, 5 + offset as u64);
        assert!(scratch[1] + len as u64 <= 7);
    }

    let mut huge = linear_segments(0, usize::MAX, 1).unwrap();
    assert_eq!(
        huge.by_ref().take(3).collect::<Vec<_>>(),
        vec![(0, 0, 1), (1, 1, 1), (2, 2, 1)]
    );
    let mut empty = linear_segments(0, 0, 7).unwrap();
    assert_eq!(empty.next(), None);
    assert_eq!(empty.next(), None);
    assert!(linear_segments(u64::MAX, 1, 7).is_err());
    assert!(linear_segments(0, 1, 0).is_err());
    assert!(decode_flat(0, &[0], &mut scratch).is_err());
    assert!(decode_flat(6, &[2, 3], &mut scratch).is_err());
    decode_flat(u64::MAX - 1, &[u64::MAX], &mut scratch).unwrap();
    assert_eq!(scratch.as_slice(), &[u64::MAX - 1]);
}

#[test]
fn linear_batches_preserve_boundaries_and_validate_shape() {
    for len in [
        1,
        MAX_BATCH_ELEMENTS - 1,
        MAX_BATCH_ELEMENTS,
        MAX_BATCH_ELEMENTS + 1,
    ] {
        let batches = linear_requests(&[len as u64]).unwrap().collect::<Vec<_>>();
        assert_eq!(batches.len(), len.div_ceil(MAX_BATCH_ELEMENTS));
        let mut next = 0;

        for batch in batches {
            assert!(matches!(batch.kind(), RequestKind::Linear { start } if *start == next as u64));
            assert_eq!(batch.len(), (len - next).min(MAX_BATCH_ELEMENTS));
            next += batch.len();
        }
        assert_eq!(next, len);
    }

    for shape in [vec![], vec![0], vec![2, 0], vec![u64::MAX, 2]] {
        assert!(linear_requests(&shape).is_err());
    }

    // Taking a prefix retains compact descriptors, not the logical tensor.
    let prefix = linear_requests(&[u64::MAX])
        .unwrap()
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(prefix.len(), 2);
    assert!(prefix.iter().all(|batch| batch.len() == MAX_BATCH_ELEMENTS));
    assert!(matches!(
        prefix[1].kind(),
        RequestKind::Linear {
            start
        } if *start == MAX_BATCH_ELEMENTS as u64
    ));
}

#[test]
fn request_validation_and_cursor_reuse() {
    assert!(BatchRequest::linear(u64::MAX, 1).is_err());
    assert!(BatchRequest::linear(0, MAX_BATCH_ELEMENTS + 1).is_err());
    assert!(
        BatchRequest::linear(5, 2)
            .unwrap()
            .validate(&[2, 3])
            .is_err()
    );
    assert!(Cartesian::new(vec![Axis::range(0, u64::MAX), Axis::range(0, 2)]).is_err());
    // Reject an unallocatable extent before narrowing it to a buffer length.
    assert!(matches!(
        Cartesian::new(vec![Axis::range(0, u64::MAX)]),
        Err(Error::InvalidLayout(_))
    ));
    let bad = BatchRequest::rectangles(vec![
        Cartesian::new(vec![Axis::Span {
            start: u64::MAX,
            step: 2,
            len: 2,
        }])
        .unwrap(),
    ])
    .unwrap();
    assert!(bad.validate(&[10]).is_err());
    assert!(BatchRequest::point(&[0, 0]).validate(&[1]).is_err());
    let request = BatchRequest::rectangles(vec![
        Cartesian::new(vec![
            Axis::Selected(vec![2, 0, 2]),
            Axis::Span {
                start: 1,
                step: 2,
                len: 2,
            },
        ])
        .unwrap(),
    ])
    .unwrap();
    let expected = vec![
        vec![2, 1],
        vec![2, 3],
        vec![0, 1],
        vec![0, 3],
        vec![2, 1],
        vec![2, 3],
    ];
    assert_eq!(request.coordinates(&[3, 4]).unwrap(), expected);
    let mut cursor = request.cursor(&[3, 4]).unwrap();
    let mut scratch = Coord::with_capacity(2);
    let ptr = scratch.as_ptr();

    for expected in expected {
        assert!(cursor.next_into(&mut scratch));
        assert_eq!(scratch.as_slice(), expected.as_slice());
        assert_eq!(scratch.as_ptr(), ptr);
    }
    assert!(!cursor.next_into(&mut scratch));
}

#[test]
fn spans_and_packed_tiles_preserve_coverage() {
    assert_eq!(
        BatchRequest::linear(2, 5)
            .unwrap()
            .coordinates(&[3, 3])
            .unwrap(),
        vec![vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2], vec![2, 0]]
    );

    for shape in [vec![2, 33, 35], vec![3, 1, 1], vec![1, 1, 129]] {
        let mut all = Vec::new();

        for request in tiled_requests(&shape).unwrap() {
            assert!(request.len() <= MAX_BATCH_ELEMENTS);
            assert!(matches!(request.kind(), RequestKind::Rectangles(_)));
            all.extend(request.coordinates(&shape).unwrap());
        }
        all.sort();
        assert_eq!(
            all,
            schema::row_major_coords(&shape)
                .unwrap()
                .collect::<Vec<_>>()
        );
    }

    let first = tiled_requests(&[1_000_000_000, 1_000_000_000])
        .unwrap()
        .next()
        .unwrap();
    let RequestKind::Rectangles(rectangles) = first.kind() else {
        panic!("expected rectangles")
    };
    assert_eq!(rectangles.len(), 4);
    assert_eq!(first.len(), MAX_BATCH_ELEMENTS);
}

#[test]
fn range_requests_tile_without_expanding_coordinates() {
    let axes = vec![Axis::range(3, 2), Axis::range(0, 10_000)];
    let requests: Vec<_> = crate::slice::Slice::new(&[10, 10_000], axes)
        .unwrap()
        .requests()
        .collect();
    assert!(
        requests
            .iter()
            .all(|request| request.len() <= MAX_BATCH_ELEMENTS)
    );
    let mut coordinates = Vec::new();

    for request in requests {
        coordinates.extend(request.coordinates(&[10, 10_000]).unwrap());
    }
    assert_eq!(coordinates.len(), 20_000);
    assert_eq!(coordinates.first(), Some(&vec![3, 0]));
    assert_eq!(coordinates.last(), Some(&vec![4, 9_999]));
}
