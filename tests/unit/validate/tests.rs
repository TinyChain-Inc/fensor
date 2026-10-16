use super::*;

#[test]
fn broadcast_geometry_preserves_logical_width_and_explicit_matrix_contract() {
    for (left, right, expected) in [
        (vec![], vec![2, 3], vec![2, 3]),
        (vec![2, 1, 4], vec![3, 4], vec![2, 3, 4]),
        (vec![u64::MAX, 1], vec![1, 2], vec![u64::MAX, 2]),
    ] {
        assert_eq!(broadcast_shape(&left, &right).unwrap().as_slice(), expected);
        assert_eq!(broadcast_shape(&right, &left).unwrap().as_slice(), expected);
    }
    assert!(broadcast_shape(&[2, 3], &[4, 3]).is_err());
    let (left, right) = matmul_broadcast_shapes(&[2, 1, 3, 4], &[5, 4, 6]).unwrap();
    assert_eq!(left.as_slice(), &[2, 5, 3, 4]);
    assert_eq!(right.as_slice(), &[2, 5, 4, 6]);
    assert_eq!(
        matmul_output_shape(&left, &right).unwrap().as_slice(),
        &[2, 5, 3, 6]
    );
    assert!(matmul_output_shape(&[2, 1, 3, 4], &[5, 4, 6]).is_err());
    assert!(matmul_broadcast_shapes(&[3], &[3, 4]).is_err());
    assert!(matmul_broadcast_shapes(&[2, 3, 4], &[5, 4, 6]).is_err());
}

#[test]
fn reduction_geometry_normalizes_axes_and_reverses_broadcasting() {
    for (rank, axes, expected) in [
        (0, vec![], vec![]),
        (3, vec![2, 0, 2], vec![0, 2]),
        (3, vec![], vec![]),
    ] {
        assert_eq!(
            reduction_axes(rank, axes.into()).unwrap().as_slice(),
            expected
        );
    }
    assert!(reduction_axes(3, vec![3, 0, 3].into()).is_err());

    for (source, target, expected) in [
        (vec![2, 3, 4], vec![1, 4], vec![0, 1]),
        (vec![2, 3], vec![2, 3], vec![]),
        (vec![2, 3], vec![], vec![0, 1]),
        (vec![u64::MAX, 3], vec![1, 3], vec![0]),
    ] {
        assert_eq!(
            broadcast_reduce_axes(&source, &target).unwrap().as_slice(),
            expected
        );
    }
    assert!(broadcast_reduce_axes(&[2, 3], &[2, 2]).is_err());
    assert!(broadcast_reduce_axes(&[2, 3], &[1, 2, 3]).is_err());
}

#[test]
fn range_cardinality_handles_empty_selections_and_overflow() {
    assert!(matches!(
        validate_range(&[3], &Range::new()),
        Err(Error::InvalidLayout(error)) if error == "range rank must match tensor rank"
    ));
    for (name, bound) in [
        ("point", AxisRange::At(3)),
        ("step", AxisRange::In(0, 3, 0)),
        ("reversed", AxisRange::In(2, 1, 1)),
        ("endpoint", AxisRange::In(0, 4, 1)),
        ("explicit", AxisRange::Of(vec![0, 3])),
    ] {
        assert!(
            matches!(validate_range(&[3], &smallvec::smallvec![bound]),
                Err(Error::InvalidLayout(error)) if error == "range bound at axis 0 is out of bounds"),
            "{name}"
        );
    }
    for (range, expected) in [
        (smallvec::smallvec![AxisRange::In(1, 3, 2)], 1),
        (smallvec::smallvec![AxisRange::In(3, 3, 1)], 0),
        (smallvec::smallvec![AxisRange::Of(vec![2, 0, 2])], 3),
    ] {
        let (lengths, count) = validate_range(&[3], &range).unwrap();
        assert_eq!(lengths.as_slice(), &[expected]);
        assert_eq!(count, expected);
        assert_eq!(
            iter_range_coords(&[3], &range).unwrap().count() as u64,
            expected
        );
    }

    let long = u64::MAX / 2;
    let selection: Range =
        smallvec::smallvec![AxisRange::In(0, long, 1), AxisRange::Of(vec![0, 1, 0])];
    assert!(matches!(
        iter_range_coords(&[long, 2], &selection),
        Err(Error::InvalidLayout(_))
    ));

    let mut empty = selection;
    empty.push(AxisRange::Of(Vec::new()));
    let mut coords = iter_range_coords(&[long, 2, 1], &empty).unwrap();
    assert_eq!(coords.remaining(), 0);
    assert_eq!(coords.next(), None);
    assert_eq!(coords.next(), None);
}
