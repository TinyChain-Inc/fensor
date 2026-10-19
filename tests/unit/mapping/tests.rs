use super::*;

#[test]
fn signed_mapping_overflow_is_a_structured_error() {
    let map = CoordinateMap {
        base_offset: i128::MAX,
        axes: vec![AxisContrib::Stride(i128::MAX)],
        shape: smallvec::smallvec![3],
    };
    assert!(matches!(map.flat_offset(&[2]), Err(Error::InvalidCoord(_))));
    assert!(matches!(map.clone().flip(0), Err(Error::InvalidLayout(_))));
    assert!(matches!(
        map.clone()
            .slice(smallvec::smallvec![AxisRange::In(0, 3, 2)]),
        Err(Error::InvalidLayout(_))
    ));
    assert!(matches!(
        map.slice(smallvec::smallvec![AxisRange::Of(vec![2])]),
        Err(Error::InvalidLayout(_))
    ));
}

#[test]
fn gather_transforms_share_the_original_table() {
    let original = GatherOffsets::from(vec![9, 2, 9, 4, 7, 3]);
    let sliced = original.slice(1, 2, 3).unwrap();
    let flipped = sliced.flipped();
    let nested = flipped.slice(1, 1, 2).unwrap();

    for view in [&sliced, &flipped, &nested] {
        assert!(Arc::ptr_eq(&original.offsets, &view.offsets));
    }
    assert_eq!(
        (0..3).map(|i| *flipped.get(i).unwrap()).collect::<Vec<_>>(),
        [3, 4, 2]
    );
    assert_eq!(
        (0..2).map(|i| *nested.get(i).unwrap()).collect::<Vec<_>>(),
        [4, 2]
    );
    assert_eq!(*original.get(0).unwrap(), *original.get(2).unwrap());
    assert_eq!(
        *nested
            .slice(1, usize::MAX, 1)
            .unwrap()
            .flipped()
            .get(0)
            .unwrap(),
        2
    );
}
