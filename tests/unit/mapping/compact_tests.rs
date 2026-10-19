use super::*;

#[test]
fn identity_is_structural_and_mapping_reuses_scratch() {
    let shape = ha_ndarray::shape![3, 4];
    let strides = schema::contiguous_strides(&shape).unwrap();
    let map = CoordinateMap::identity(shape.clone(), &strides);
    assert!(map.is_identity(&shape, &strides));
    let flipped = map.clone().flip(1).unwrap();
    assert!(!flipped.is_identity(&shape, &strides));
    assert!(
        flipped
            .clone()
            .flip(1)
            .unwrap()
            .is_identity(&shape, &strides)
    );
    let mut out = Coord::with_capacity(2);
    let ptr = out.as_ptr();

    for row in 0..3 {
        for col in 0..4 {
            flipped
                .resolve_into(&[row, col], &shape, &strides, &mut out)
                .unwrap();
            assert_eq!(out.as_slice(), &[row, 3 - col]);
            assert_eq!(out.as_ptr(), ptr);
        }
    }

    let mut bad = map;
    bad.base_offset = 12;
    assert!(
        bad.resolve_into(&[0, 0], &shape, &strides, &mut out)
            .is_err()
    );
    bad.base_offset = i128::MAX;
    assert!(
        bad.resolve_into(&[2, 3], &shape, &strides, &mut out)
            .is_err()
    );
}
