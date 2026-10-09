use super::*;

#[test]
fn matrix_shapes_reject_overflow_and_empty_dimensions() {
    assert!(crate::validate::matmul_output_shape(&[u64::MAX, 2], &[2, 1]).is_err());
    assert!(crate::validate::matmul_output_shape(&[u64::MAX, 1], &[1, 2]).is_err());
    assert!(crate::validate::matmul_output_shape(&[1, 0], &[0, 1]).is_err());
}

#[test]
fn tile_planning_is_bounded_and_preserves_requests() {
    let shape = ha_ndarray::shape![2, 1000, 1000];
    let strides = crate::schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides);
    // Fixed 64-by-64 fixture spans four current spatial tiles.
    let coords: Vec<_> = (0..64 * 64)
        .map(|i| vec![1, (i / 64) as u64, (i % 64) as u64])
        .collect();
    let tiles = plan_tiles(
        &mapping,
        &shape,
        &strides,
        &BatchRequest::explicit(coords).unwrap(),
    )
    .unwrap();
    assert_eq!(tiles.len(), 4);
    assert_eq!(tiles.values().map(Vec::len).sum::<usize>(), 64 * 64);

    for requests in tiles.values() {
        let rows: std::collections::BTreeSet<_> = requests.iter().map(|(_, r, _)| r).collect();
        let columns: std::collections::BTreeSet<_> = requests.iter().map(|(_, _, c)| c).collect();
        assert!(
            rows.len() * contraction_step(rows.len(), columns.len())
                <= expression::MAX_BATCH_ELEMENTS
        );
        assert!(
            columns.len() * contraction_step(rows.len(), columns.len())
                <= expression::MAX_BATCH_ELEMENTS
        );
    }
    assert!(
        BatchRequest::explicit(vec![vec![0, 0, 0]; expression::MAX_BATCH_ELEMENTS + 1]).is_err()
    );
    let requests = plan_tiles(
        &mapping,
        &shape,
        &strides,
        &BatchRequest::explicit(vec![vec![0, 1, 2], vec![0, 1, 2]]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        requests.values().next().unwrap(),
        &vec![(0, 1, 2), (1, 1, 2)]
    );
}

#[test]
fn irregular_rectangles_bound_arithmetic_and_retain_duplicates() {
    for requests in [
        (0..TILE_SIDE)
            .map(|i| (i, i as u64, i as u64))
            .collect::<Vec<_>>(),
        (0..TILE_SIDE * TILE_SIDE)
            .map(|i| (i, (i / TILE_SIDE) as u64, (i % TILE_SIDE) as u64))
            .collect(),
        vec![(0, 0, 0), (1, 0, 0), (2, 1, 1), (3, 0, 1)],
    ] {
        let unique: std::collections::BTreeSet<_> =
            requests.iter().map(|(_, r, c)| (*r, *c)).collect();
        let rectangles = plan_rectangles(requests.clone());
        assert!(
            rectangles
                .iter()
                .map(|r| r.rows.len() * r.columns.len())
                .sum::<usize>()
                <= MAX_RECTANGLE_AMPLIFICATION * unique.len()
        );
        let mut seen = vec![false; requests.len()];

        for rect in rectangles {
            rect.scatter.each(rect.columns.len(), |position, offset| {
                assert!(!seen[position]);
                seen[position] = true;
                assert_eq!(
                    (
                        rect.rows[offset / rect.columns.len()],
                        rect.columns[offset % rect.columns.len()]
                    ),
                    (requests[position].1, requests[position].2)
                );
            });
        }
        assert!(seen.into_iter().all(|v| v));
    }
    assert_eq!(
        contraction_step(TILE_SIDE, TILE_SIDE),
        expression::MAX_BATCH_ELEMENTS / TILE_SIDE
    );
    assert_eq!(contraction_step(1, 1), expression::MAX_BATCH_ELEMENTS);
    assert_eq!(contraction_step(2, 1), expression::MAX_BATCH_ELEMENTS / 2);
    assert!(combine_partial(&mut Some(vec![1u8]), vec![2, 3], 1).is_err());
    assert!(combine_partial(&mut Some(vec![1u8, 2]), vec![3], 1).is_err());
}

#[test]
fn planner_work_counts() {
    for (name, m, k, n) in [
        ("square", 32, 129, 32),
        ("wide", 32, 17, 4097),
        ("tall", 129, 129, 1),
        ("dot", 1, 4097, 1),
    ] {
        let shape = ha_ndarray::shape![m, n];
        let strides = crate::schema::contiguous_strides(&shape).unwrap();
        let mapping = CoordinateMap::identity(shape.clone(), &strides);
        let mut totals = Vec::new();

        for revised in [false, true] {
            let requests: expression::RequestIterator = if revised {
                Box::new(request::tiled_requests(&shape).unwrap())
            } else {
                Box::new(request::linear_requests(&shape).unwrap())
            };

            let mut operands = 0;
            let mut calls = 0;

            for request in requests {
                for plan in plan_request(&mapping, &shape, &strides, &request).unwrap() {
                    let rect = plan.rectangle;
                    operands += (rect.rows.len() + rect.columns.len()) * k;
                    calls += k.div_ceil(if revised {
                        contraction_step(rect.rows.len(), rect.columns.len())
                    } else {
                        // Historical fixed-chunk baseline, not the current adaptive limit.
                        128
                    });
                }
            }
            totals.push((operands, calls));
            println!("PLAN,{name},{revised},{operands},{calls}");
        }
        assert!(totals[1].0 <= totals[0].0);
        assert!(totals[1].1 <= totals[0].1);
    }
}

fn planned_coordinates(plans: &[MatrixPlan], len: usize) -> Vec<Vec<u64>> {
    let mut coords = vec![Vec::new(); len];

    for plan in plans {
        let rect = &plan.rectangle;
        rect.scatter.each(rect.columns.len(), |position, offset| {
            assert!(coords[position].is_empty());
            let mut coord = plan.prefix.clone();
            coord.extend([
                rect.rows[offset / rect.columns.len()],
                rect.columns[offset % rect.columns.len()],
            ]);
            coords[position] = coord;
        });
    }
    coords
}

#[test]
fn compact_and_explicit_plans_agree() {
    let shape = ha_ndarray::shape![2, 35, 67];
    let strides = crate::schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides);
    let mut requests: Vec<_> = request::tiled_requests(&shape).unwrap().collect();
    requests.extend([
        BatchRequest::linear(66, 3000).unwrap(),
        BatchRequest::linear(2000, 2690).unwrap(),
        BatchRequest::rectangles(vec![
            Cartesian::new(vec![
                Axis::Selected(vec![1, 0, 1]),
                Axis::Selected(vec![34, 0, 34, 1]),
                Axis::Selected(vec![66, 0, 31, 32]),
            ])
            .unwrap(),
        ])
        .unwrap(),
    ]);

    for request in requests {
        let expected = request.coordinates(&shape).unwrap();
        let direct = plan_request(&mapping, &shape, &strides, &request).unwrap();
        assert!(
            direct
                .iter()
                .all(|p| !matches!(p.rectangle.scatter, Scatter::Explicit(_)))
        );
        let explicit = plan_request(
            &mapping,
            &shape,
            &strides,
            &BatchRequest::explicit(expected.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(planned_coordinates(&direct, request.len()), expected);
        assert_eq!(planned_coordinates(&explicit, request.len()), expected);
    }

    let flipped = mapping.flip(2).unwrap();
    let request = BatchRequest::linear(60, 100).unwrap();
    let expected: Vec<_> = request
        .coordinates(&shape)
        .unwrap()
        .iter()
        .map(|c| flipped.resolve(c, &shape, &strides).unwrap())
        .collect();
    let plans = plan_request(&flipped, &shape, &strides, &request).unwrap();
    assert!(
        plans
            .iter()
            .all(|p| matches!(p.rectangle.scatter, Scatter::Explicit(_)))
    );
    assert_eq!(planned_coordinates(&plans, request.len()), expected);
}

#[test]
fn compact_planning_representation_counts() {
    let shape = ha_ndarray::shape![32, 4097];
    let strides = crate::schema::contiguous_strides(&shape).unwrap();
    let mapping = CoordinateMap::identity(shape.clone(), &strides);
    let mut descriptors = 0;
    let mut logical = 0;
    let mut scatter_slots = 0;
    let mut calls = 0;

    for request in request::tiled_requests(&shape).unwrap() {
        let RequestKind::Rectangles(rectangles) = request.kind() else {
            panic!("expanded regular traversal")
        };
        descriptors += rectangles.len();
        logical += request.len();

        for plan in plan_request(&mapping, &shape, &strides, &request).unwrap() {
            let Scatter::Cartesian { rows, columns, .. } = &plan.rectangle.scatter else {
                panic!("expanded regular scatter")
            };
            scatter_slots += rows.len() + columns.len();
            calls += 17usize.div_ceil(contraction_step(
                plan.rectangle.rows.len(),
                plan.rectangle.columns.len(),
            ));
        }
    }
    assert_eq!(
        (logical, descriptors, scatter_slots, calls),
        (131104, 129, 8225, 129)
    );
    println!(
        "COMPACT,wide,logical={logical},rectangles={descriptors},scatter_axis_slots={scatter_slots},matmul_calls={calls},pre_evaluation_coordinate_payloads=0"
    );
}
