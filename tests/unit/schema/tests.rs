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
fn storage_schema_new_evenly_dividing() {
    let storage = StorageSchema::new(&[4, 6], Layout::Dense, 1000).unwrap();
    assert_eq!(storage.blocks.capacity, 24);
    assert_eq!(storage.blocks.count(), 1);
}

#[test]
fn storage_schema_new_ceiling_remainder() {
    let storage = StorageSchema::new(&[10], Layout::Dense, 3).unwrap();
    assert_eq!(storage.blocks.count(), 4);
    assert_eq!(storage.blocks.range(3).unwrap(), 9..10);
}

#[test]
fn sparse_storage_has_scalar_or_dense_suffix_payloads() {
    let shape = [3, 5, 7];
    for (axis, length) in [(None, 1), (Some(0), 35), (Some(1), 7), (Some(2), 1)] {
        let storage =
            StorageSchema::new(&shape, Layout::Sparse { axis }, MAX_BLOCK_CAPACITY).unwrap();

        assert_eq!(storage.blocks.region_len, length);
        assert_eq!(storage.blocks.capacity as u64, length);
        for id in 0..storage.blocks.count() {
            assert_eq!(
                storage.blocks.range(id).unwrap().end - storage.blocks.range(id).unwrap().start,
                length
            );
        }
    }

    for axis in [shape.len(), usize::MAX] {
        assert!(StorageSchema::new(&shape, Layout::Sparse { axis: Some(axis) }, 4).is_err());
    }
}

#[test]
fn storage_schema_new_rejects_zero_capacity() {
    let err = StorageSchema::new(&[4, 5, 6], Layout::Dense, 0).expect_err("zero capacity rejected");
    assert!(matches!(err, crate::Error::InvalidLayout(_)));
}

#[test]
fn storage_schema_new_rejects_empty_shape() {
    let err = StorageSchema::new(&[], Layout::Dense, 10).expect_err("empty shape rejected");
    assert!(matches!(err, crate::Error::InvalidSchema(_)));
}

#[test]
fn compact_linear_blocks_cover_regions_and_progressions() {
    use crate::schema::LinearBlocks;
    for shape in [vec![1], vec![3, 5], vec![5, 7, 11], vec![2, 3, 4]] {
        for prefix in 0..=shape.len() {
            let layout = if prefix == 0 {
                crate::Layout::Dense
            } else {
                crate::Layout::Sparse {
                    axis: Some(prefix - 1),
                }
            };

            for capacity in [1, 2, 3, 4, 7, 32, 4096] {
                let blocks = LinearBlocks::new(&shape, layout, capacity).unwrap();
                let size: u64 = shape.iter().product();
                // Independent oracle: chunk each dense suffix separately.
                let region_len: u64 = shape[prefix..].iter().product();
                let mut expected_ranges = Vec::new();
                for region in (0..size).step_by(region_len as usize) {
                    let positions: Vec<_> = (region..region + region_len).collect();
                    for chunk in positions.chunks(capacity) {
                        expected_ranges.push(chunk[0]..chunk[chunk.len() - 1] + 1);
                    }
                }

                assert_eq!(blocks.count() as usize, expected_ranges.len());
                for (id, expected) in expected_ranges.iter().enumerate() {
                    assert_eq!(&blocks.range(id as u64).unwrap(), expected);
                }

                let mut expected_start = 0;
                for id in 0..blocks.count() {
                    let range = blocks.range(id).unwrap();
                    assert_eq!(range.start, expected_start);
                    assert!(!range.is_empty());
                    assert!(range.end - range.start <= capacity as u64);
                    for flat in range.clone() {
                        let actual = blocks.position(flat);
                        assert_eq!(actual.block_id, id);
                        assert_eq!(actual.offset_in_block as u64, flat - range.start);
                    }

                    expected_start = range.end;
                }

                assert_eq!(expected_start, size);
                assert!(blocks.range(blocks.count()).is_err());
                for step in [-7_i128, -1, 0, 1, 7] {
                    for start in 0..size {
                        let len = if step == 0 {
                            17
                        } else if step < 0 {
                            (start as i128 / -step + 1) as usize
                        } else {
                            ((size - 1 - start) as i128 / step + 1) as usize
                        };

                        let mut output = 0;
                        blocks
                            .split(
                                start as i128,
                                step,
                                len,
                                |id, offset, stride, count, done| {
                                    assert_eq!(done, output);
                                    let range = blocks.range(id).unwrap();
                                    for i in 0..count {
                                        assert_eq!(
                                            range.start as i128
                                                + offset as i128
                                                + i as i128 * stride as i128,
                                            start as i128 + (done + i) as i128 * step
                                        );
                                    }

                                    output += count;
                                    Ok(())
                                },
                            )
                            .unwrap();

                        assert_eq!(output, len);
                    }
                }
            }
        }
    }

    for capacity in [0, 4097] {
        assert!(LinearBlocks::new(&[3, 5], crate::Layout::Dense, capacity).is_err());
    }

    assert!(LinearBlocks::new(&[u64::MAX, 2], crate::Layout::Dense, 4096).is_err());
}

#[test]
fn compact_flat_rectangles_preserve_exact_coverage() {
    use crate::schema::flat_rectangles;
    for shape in [vec![1], vec![3, 5], vec![2, 3, 4], vec![3, 2, 4, 2]] {
        let size: u64 = shape.iter().product();
        let strides = crate::schema::contiguous_strides(&shape).unwrap();
        let schema =
            TensorSchema::new(NumberType::Float(FloatType::F64), shape.clone().into()).unwrap();

        for start in 0..=size {
            for end in start..=size {
                let mut next = start;
                let mut count = 0;
                for bounds in flat_rectangles(&schema, start..end).unwrap() {
                    count += 1;
                    let lengths: Vec<_> = bounds.iter().map(|&(lo, hi)| hi - lo).collect();
                    let mut local = vec![0; shape.len()];
                    for _ in 0..lengths.iter().product::<u64>() {
                        let flat: u64 = local
                            .iter()
                            .zip(&bounds)
                            .zip(&strides)
                            .map(|((&c, &(lo, _)), &stride)| (lo + c) * stride)
                            .sum();

                        assert_eq!(flat, next);
                        next += 1;
                        for axis in (0..local.len()).rev() {
                            local[axis] += 1;
                            if local[axis] < lengths[axis] {
                                break;
                            }

                            local[axis] = 0;
                        }
                    }
                }

                assert_eq!(next, end);
                assert!(count < 2 * shape.len());
            }
        }

        assert!(flat_rectangles(&schema, 0..size + 1).is_err());
    }
}
