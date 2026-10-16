use super::*;

#[test]
fn contiguous_strides_valid_shapes() {
    let cases: &[(&str, &[u64], &[u64])] = &[
        ("rank_three", &[4, 5, 6], &[30, 6, 1]),
        ("rank_two", &[2, 3], &[3, 1]),
        ("single_dim", &[7], &[1]),
        // Unit dimensions preserve trailing strides; a trailing unit collapses them.
        ("all_ones", &[1, 1, 1], &[1, 1, 1]),
        ("leading_one", &[1, 5, 6], &[30, 6, 1]),
        ("trailing_one", &[4, 5, 1], &[5, 1, 1]),
    ];

    for &(name, shape, expected) in cases {
        assert_eq!(
            contiguous_strides(shape).expect(name).as_slice(),
            expected,
            "{name}"
        );
    }
}

#[test]
fn contiguous_strides_reject_invalid_shapes() {
    let cases: &[(&str, &[u64])] = &[
        ("empty_shape", &[]),
        ("zero_dim", &[3, 0, 4]),
        // 1 * MAX fits, but the next stride multiplies MAX by 2.
        ("overflow", &[3, 2, u64::MAX]),
    ];

    for &(name, shape) in cases {
        let err = contiguous_strides(shape).expect_err(name);
        assert!(matches!(err, Error::InvalidSchema(_)), "{name}: {err}");
    }
}

#[test]
fn greedy_block_shape_valid_capacities() {
    let cases: &[(&str, &[u64], usize, &[u64])] = &[
        ("capacity_covers_whole_tensor", &[4, 5, 6], 1000, &[4, 5, 6]),
        ("capacity_equals_total", &[2, 3], 6, &[2, 3]),
        // Fill trailing axes first: 50 / 6 / 5 leaves room for one outer element.
        ("partial_fit_limits_outer_axis", &[4, 5, 6], 50, &[1, 5, 6]),
        ("rank_two_partial", &[3, 7], 5, &[1, 5]),
        // Capacity need not evenly divide the axis.
        ("single_dim", &[10], 3, &[3]),
        ("capacity_one", &[4, 5, 6], 1, &[1, 1, 1]),
    ];

    for &(name, shape, capacity, expected) in cases {
        assert_eq!(
            greedy_block_shape(shape, capacity).expect(name).as_slice(),
            expected,
            "{name}"
        );
    }
}

#[test]
fn greedy_block_shape_reject_invalid_inputs() {
    let cases: &[(&str, &[u64], usize)] = &[
        ("zero_capacity", &[4, 5, 6], 0),
        ("empty_shape", &[], 10),
        ("zero_dim", &[3, 0, 4], 10),
    ];

    for &(name, shape, capacity) in cases {
        let err = greedy_block_shape(shape, capacity).expect_err(name);
        assert!(matches!(err, Error::InvalidSchema(_)), "{name}: {err}");
    }
}

#[test]
fn tensor_schema_new_computes_strides() {
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), vec![4u64, 5, 6].into())
        .expect("schema");
    assert_eq!(schema.shape().as_slice(), &[4, 5, 6]);
    assert_eq!(schema.strides().as_slice(), &[30, 6, 1]);
}

#[test]
fn tensor_schema_new_rejects_empty_shape() {
    let err = TensorSchema::new(NumberType::Float(FloatType::F32), Shape::new())
        .expect_err("empty shape rejected");
    assert!(matches!(err, crate::Error::InvalidSchema(_)));
}

#[test]
fn block_schema_new_covers_whole_tensor() {
    let block = BlockSchema::new(&[4, 5, 6], 1000).expect("block schema");
    assert_eq!(block.shape.as_slice(), &[4, 5, 6]);
    assert_eq!(block.strides.as_slice(), &[30, 6, 1]);
}

#[test]
fn storage_schema_new_evenly_dividing() {
    // tensor [4, 6], capacity large enough for block [4, 6] -> grid [1, 1]
    let storage = StorageSchema::new(&[4, 6], Layout::Dense, 1000).expect("storage schema");
    assert_eq!(storage.shape.as_slice(), &[1, 1]);
    assert_eq!(storage.block_schema.shape.as_slice(), &[4, 6]);
}

#[test]
fn storage_schema_new_ceiling_remainder() {
    // tensor [10], capacity 3 -> block [3], grid ceil(10/3) = 4
    let storage = StorageSchema::new(&[10], Layout::Dense, 3).expect("storage schema");
    assert_eq!(storage.block_schema.shape.as_slice(), &[3]);
    assert_eq!(storage.shape.as_slice(), &[4]);
    assert_eq!(storage.strides.as_slice(), &[1]);
}

#[test]
fn storage_schema_from_block_shape_matches_new() {
    let via_new = StorageSchema::new(&[10], Layout::Dense, 3).expect("via new");
    let via_block_shape = StorageSchema::from_block_shape(&[10], Layout::Dense, vec![3u64].into())
        .expect("via from_block_shape");
    assert_eq!(via_new, via_block_shape);
}

#[test]
fn sparse_storage_has_scalar_or_dense_suffix_payloads() {
    let shape = [3, 5, 7];

    for (axis, capacity, expected) in [
        (None, MAX_BLOCK_CAPACITY, [1, 1, 1]),
        (Some(0), MAX_BLOCK_CAPACITY, [1, 5, 7]),
        (Some(1), MAX_BLOCK_CAPACITY, [1, 1, 7]),
        (Some(2), MAX_BLOCK_CAPACITY, [1, 1, 1]),
        (Some(0), 4, [1, 1, 4]),
        (Some(1), 4, [1, 1, 4]),
    ] {
        let layout = Layout::Sparse { axis };
        let storage = StorageSchema::new(&shape, layout, capacity).unwrap();
        assert_eq!(storage.block_schema.shape.as_slice(), expected);
        assert_eq!(
            StorageSchema::from_block_shape(&shape, layout, expected.to_vec().into()).unwrap(),
            storage
        );

        for prefix in 0..axis.map_or(shape.len(), |axis| axis + 1) {
            let mut invalid = expected;
            invalid[prefix] = 2;
            assert!(matches!(
                StorageSchema::from_block_shape(&shape, layout, invalid.to_vec().into()),
                Err(Error::InvalidLayout(_))
            ));
        }
    }

    for axis in [shape.len(), usize::MAX] {
        let layout = Layout::Sparse { axis: Some(axis) };
        assert!(matches!(
            StorageSchema::new(&shape, layout, MAX_BLOCK_CAPACITY),
            Err(Error::InvalidSchema(_))
        ));
        assert!(matches!(
            StorageSchema::from_block_shape(&shape, layout, vec![1, 1, 1].into()),
            Err(Error::InvalidSchema(_))
        ));
    }
}

#[test]
fn storage_schema_new_rejects_zero_capacity() {
    let err = StorageSchema::new(&[4, 5, 6], Layout::Dense, 0).expect_err("zero capacity rejected");
    assert!(matches!(err, crate::Error::InvalidSchema(_)));
}

#[test]
fn storage_schema_new_rejects_empty_shape() {
    let err = StorageSchema::new(&[], Layout::Dense, 10).expect_err("empty shape rejected");
    assert!(matches!(err, crate::Error::InvalidSchema(_)));
}
