use ha_ndarray::{axes, range, shape};

use super::*;
use crate::AxisRange;
use crate::mapping::test_support::affine;
use crate::request::{Axis, Cartesian};
use crate::schema::contiguous_strides;

fn compare(storage: &StorageShape<'_>, map: &CoordinateMap, request: BatchRequest) {
    let planned = storage.plan(&request, Some(map)).unwrap();
    let mut actual = vec![(0, 0); request.len()];

    for (key, runs) in planned {
        for run in runs {
            for i in 0..run.len {
                actual[run.output + i] = (
                    key,
                    (run.offset as i128 + i as i128 * run.stride as i128) as usize,
                );
            }
        }
    }

    let expected: Vec<_> = request
        .coordinates(map.shape())
        .unwrap()
        .iter()
        .map(|coord| {
            let base = map
                .resolve(
                    coord,
                    storage.geometry.schema().shape(),
                    storage.geometry.schema().strides(),
                )
                .unwrap();

            let base: Vec<_> = base.into_iter().collect();
            let (id, offset) = storage.position(&base);
            (id, offset)
        })
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn affine_plans_match_scalar_mapping_across_blocks() {
    let shape = shape![3, 5, 7];
    let strides = contiguous_strides(&shape).unwrap();
    let identity = CoordinateMap::identity(shape.clone(), &strides);
    let maps = [
        identity.clone(),
        identity.clone().transpose(Some(axes![2, 0, 1])).unwrap(),
        identity.clone().flip(0).unwrap().flip(2).unwrap(),
        identity
            .clone()
            .slice(range![
                AxisRange::In(0, 3, 2),
                AxisRange::In(1, 5, 2),
                AxisRange::In(0, 7, 3)
            ])
            .unwrap(),
        identity.clone().reshape(shape![15, 7]).unwrap(),
        identity
            .clone()
            .reshape(shape![105])
            .unwrap()
            .slice(range![AxisRange::In(0, 105, 11)])
            .unwrap(),
        identity
            .clone()
            .slice(range![
                AxisRange::At(1),
                AxisRange::In(0, 5, 1),
                AxisRange::At(3)
            ])
            .unwrap()
            .broadcast(shape![4, 5])
            .unwrap(),
        identity
            .clone()
            .slice(range![
                AxisRange::Of(vec![2, 0, 2]),
                AxisRange::In(0, 5, 1),
                AxisRange::In(0, 7, 1)
            ])
            .unwrap()
            .flip(1)
            .unwrap(),
    ];

    for block_shape in [vec![1, 1, 1], vec![2, 3, 4], vec![3, 5, 7]] {
        let geometry = crate::StorageGeometry::new(
            crate::TensorSchema::new(
                number_general::NumberType::Float(number_general::FloatType::F64),
                shape.to_vec().into(),
            )
            .unwrap(),
            crate::Layout::Dense,
            block_shape.iter().copied().product::<u64>() as usize,
        )
        .unwrap();

        let storage = StorageShape {
            geometry: &geometry,
        };

        for map in &maps {
            let size = map.shape().iter().product::<u64>();

            for start in 0..size {
                compare(
                    &storage,
                    map,
                    BatchRequest::linear(start, (size - start).min(13) as usize).unwrap(),
                );
            }

            let axes = map
                .shape()
                .iter()
                .map(|d| Axis::selected(vec![(*d - 1), 0, 0]))
                .collect();

            compare(
                &storage,
                map,
                BatchRequest::rectangles(vec![Cartesian::new(axes).unwrap()]).unwrap(),
            );
        }
    }
}

#[test]
fn progressions_handle_arbitrary_signed_steps_and_carries() {
    let shape = [3, 5, 7];
    let strides = [35, 7, 1];
    let geometry = crate::StorageGeometry::new(
        crate::TensorSchema::new(
            number_general::NumberType::Float(number_general::FloatType::F64),
            shape.to_vec().into(),
        )
        .unwrap(),
        crate::Layout::Dense,
        [2, 3, 4].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let storage = StorageShape {
        geometry: &geometry,
    };

    for start in 0..105 {
        for step in -104..=104 {
            let mut len = 1;

            while len < 16 && (0..105).contains(&(start + len * step)) {
                len += 1;
            }

            let map = affine(start, step, shape![len as usize as u64]);
            let request = BatchRequest::linear(0, len as usize).unwrap();
            let mapped = request.mapped(&map, &shape, &strides).unwrap();
            compare(&storage, &map, request);

            let nested = CoordinateMap::identity(shape![3, 5, 7], &strides)
                .transpose(Some(axes![2, 0, 1]))
                .unwrap()
                .flip(1)
                .unwrap();

            compare(&storage, &nested, mapped);
        }
    }
}

#[tokio::test]
async fn compact_runs_reduce_mapping_work_and_preserve_broadcasts() {
    let geometry = crate::StorageGeometry::new(
        crate::TensorSchema::new(
            number_general::NumberType::Float(number_general::FloatType::F64),
            [8, 8].to_vec().into(),
        )
        .unwrap(),
        crate::Layout::Dense,
        [8, 8].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let storage = StorageShape {
        geometry: &geometry,
    };

    let identity = CoordinateMap::identity(shape![8, 8], &[8, 1]);

    for (map, expected_runs) in [
        (identity.clone(), 1),
        (identity.clone().transpose(None).unwrap(), 8),
        (identity.clone().flip(1).unwrap(), 8),
        (
            identity
                .clone()
                .slice(range![AxisRange::In(0, 8, 1), AxisRange::In(0, 1, 1)])
                .unwrap()
                .broadcast(shape![8, 512])
                .unwrap(),
            8,
        ),
    ] {
        let request =
            BatchRequest::linear(0, map.shape().iter().product::<u64>() as usize).unwrap();

        let strides = contiguous_strides(map.shape()).unwrap();
        let identity = CoordinateMap::identity(map.shape().into(), &strides);
        let mapped = request.mapped(&identity, map.shape(), &strides).unwrap();

        for request in [&request, &mapped] {
            crate::read_metrics::CURRENT
                .scope(Default::default(), async {
                    let plan = storage.plan(request, Some(&map)).unwrap();
                    assert_eq!(plan.values().map(Vec::len).sum::<usize>(), expected_runs);
                    crate::read_metrics::CURRENT.with(|m| {
                        let m = m.borrow();
                        assert_eq!(m.coordinate_resolutions, 0);
                        assert_eq!(m.boundary_decodes, 0);
                    });
                })
                .await;
        }
    }

    let huge = StorageShape {
        geometry: &crate::StorageGeometry::new(
            crate::TensorSchema::new(
                number_general::NumberType::Float(number_general::FloatType::F64),
                [1_000_000_000, 1_000_000_000].to_vec().into(),
            )
            .unwrap(),
            crate::Layout::Dense,
            [1, crate::schema::MAX_BLOCK_CAPACITY as u64]
                .iter()
                .copied()
                .product::<u64>() as usize,
        )
        .unwrap(),
    };

    let request = BatchRequest::linear(999_999_999_999_999_968, 32).unwrap();
    assert!(huge.plan(&request, None).unwrap().len() <= 2);
}

#[tokio::test]
async fn singleton_axes_do_not_expand_narrow_requests() {
    let batch_len = crate::expression::MAX_BATCH_ELEMENTS;
    let block_len = 128; // Deliberately narrower than one execution batch.
    let expected_runs = batch_len.div_ceil(block_len);
    let geometry = crate::StorageGeometry::new(
        crate::TensorSchema::new(
            number_general::NumberType::Float(number_general::FloatType::F64),
            [batch_len as u64 + 1, 1].to_vec().into(),
        )
        .unwrap(),
        crate::Layout::Dense,
        [block_len as u64, 1].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let storage = StorageShape {
        geometry: &geometry,
    };

    for request in [
        BatchRequest::linear(0, batch_len).unwrap(),
        BatchRequest::rectangles(vec![
            Cartesian::new(vec![
                Axis::range(0, batch_len as u64),
                Axis::selected(vec![0]),
            ])
            .unwrap(),
        ])
        .unwrap(),
    ] {
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                let runs = storage.plan(&request, None).unwrap();
                assert_eq!(runs.len(), expected_runs);
                crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().boundary_decodes, 0));
            })
            .await;
    }
}

#[test]
fn malformed_progressions_and_runs_fail_closed() {
    let geometry = crate::StorageGeometry::new(
        crate::TensorSchema::new(
            number_general::NumberType::Float(number_general::FloatType::F64),
            [8, 8].to_vec().into(),
        )
        .unwrap(),
        crate::Layout::Dense,
        [8, 8].iter().copied().product::<u64>() as usize,
    )
    .unwrap();

    let storage = StorageShape {
        geometry: &geometry,
    };

    for (offset, stride) in [(64, 1), (-1, 1), (1, -1), (i128::MAX, i128::MAX)] {
        let map = affine(offset, stride, shape![3]);
        assert!(
            storage
                .plan(&BatchRequest::linear(0, 3).unwrap(), Some(&map))
                .is_err()
        );
    }

    assert!(
        storage
            .plan(&BatchRequest::linear(63, 2).unwrap(), None)
            .is_err()
    );

    for run in [
        Run {
            output: 0,
            offset: 0,
            stride: -1,
            len: 2,
        },
        Run {
            output: usize::MAX,
            offset: 0,
            stride: 0,
            len: 2,
        },
        Run {
            output: 0,
            offset: 63,
            stride: 1,
            len: 2,
        },
    ] {
        let mut output = vec![0; 64];
        assert!(run.scatter(&[1; 64], &mut output).is_err());
        assert!(output.iter().all(|v| *v == 0));
    }
}
