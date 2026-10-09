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
                geometry.validate_block(id, &values).unwrap();
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
            for id in [geometry.block_count(), u64::MAX] {
                assert!(matches!(
                    geometry.validate_block::<f64>(id, &[]),
                    Err(Error::InvalidCoord(_))
                ));
            }
            for len in [geometry.block_len() - 1, geometry.block_len() + 1] {
                assert!(matches!(
                    geometry.validate_block(0, &vec![0.0; len]),
                    Err(Error::InvalidLayout(_))
                ));
            }
        }
    }
}
