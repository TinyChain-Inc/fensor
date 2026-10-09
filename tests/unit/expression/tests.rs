use super::*;
use crate::Layout;
use crate::test_support::counters::Counter;

// Wrap real storage only to observe dispatch; numerical reads still delegate.
#[derive(Clone)]
struct Provider<'a> {
    source: &'a Tensor<crate::test_support::FsEntry, u8>,
    calls: &'a Counter,
    provide: fn(&[u64]) -> Result<Option<RequestIterator>>,
}

impl TensorGeometry for Provider<'_> {
    type DType = u8;

    fn dtype(&self) -> crate::NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        self.source.shape()
    }

    fn layout(&self) -> Layout {
        self.source.layout()
    }
}

impl crate::expression::traversal::Plan for Provider<'_> {
    fn preferred_step<'a>(&'a self, shape: &'a [u64]) -> Result<traversal::Preferred<'a>> {
        self.calls.increment();
        (self.provide)(shape).map(traversal::Preferred::Ready)
    }

    fn ordered_step(&self, slice: crate::slice::Slice) -> Result<traversal::Ordered<'_>> {
        self.calls.increment();
        Ok(traversal::Ordered::Ready(
            match (self.provide)(self.shape())? {
                Some(requests) => futures::stream::iter(requests.map(Ok)).boxed(),
                None => slice.stream(),
            },
        ))
    }
}

impl Expression for Provider<'_> {
    fn build<'a>(
        &'a self,
        context: Context<'a>,
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<Batch<u8>>> {
        self.source.build(context, request)
    }
}

#[tokio::test]
async fn request_providers_delegate_once_in_order_and_propagate_errors() {
    use crate::{TensorMath, TensorUnaryBoolean, TensorWhere};

    type Provide = fn(&[u64]) -> Result<Option<RequestIterator>>;
    let none: Provide = |_| Ok(None);
    let some: Provide = |shape| Ok(Some(Box::new(request::linear_requests(shape)?)));
    let error: Provide = |_| Err(Error::Unsupported("provider failure".into()));
    let (root, dir) = crate::test_support::new_dir("request_providers").await;
    let tensor = Tensor::create(
        dir,
        crate::TensorSchema::new(
            <u8 as number_general::DType>::dtype(),
            ha_ndarray::shape![2, 3],
        )
        .unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();

    for (providers, expected, fails, supplies) in [
        ([some, error, error], [1, 0, 0], false, true),
        ([some, some, some], [1, 0, 0], false, true),
        ([none, some, error], [1, 1, 0], false, true),
        ([none, none, some], [1, 1, 1], false, true),
        ([error, some, some], [1, 0, 0], true, false),
        ([none, error, some], [1, 1, 0], true, false),
        ([none, none, error], [1, 1, 1], true, false),
        ([none, none, none], [1, 1, 1], false, false),
    ] {
        let calls = [Counter::new(), Counter::new(), Counter::new()];
        let [a, b, c] = std::array::from_fn(|i| Provider {
            source: &tensor,
            calls: &calls[i],
            provide: providers[i],
        });
        // Consumer shape is forwarded unchanged, rather than replaced by a leaf shape.
        let selected = a.cond(&b, &c).await.unwrap().not().await.unwrap();
        let result = traversal::preferred(&selected, &[7, 3]);
        assert_eq!(result.is_err(), fails);
        assert_eq!(matches!(&result, Ok(Some(_))), supplies);
        if let Ok(Some(mut requests)) = result {
            assert_eq!(requests.next().unwrap().len(), 21);
            assert!(requests.next().is_none());
        }
        assert_eq!(calls.each_ref().map(|n| n.read()), expected);

        for call in &calls {
            call.reset();
        }

        let binary = a.add(&b).await.unwrap();
        let result = traversal::preferred(&binary, &[7, 3]);
        assert_eq!(result.is_err(), fails && expected[2] == 0);
        assert_eq!(matches!(&result, Ok(Some(_))), supplies && expected[2] == 0);
        assert_eq!(
            calls.each_ref().map(|n| n.read()),
            [expected[0], expected[1], 0]
        );
    }
    crate::test_support::cleanup(&root).await;
}

#[tokio::test]
async fn ordered_providers_visit_operands_left_to_right() {
    use crate::{TensorAbs, TensorMath, TensorWhere};

    type Provide = fn(&[u64]) -> Result<Option<RequestIterator>>;
    let none: Provide = |_| Ok(None);
    let some: Provide = |_| {
        Ok(Some(Box::new(
            [
                BatchRequest::explicit(vec![vec![0], vec![0], vec![2]])?,
                BatchRequest::explicit(vec![vec![4]])?,
            ]
            .into_iter(),
        )))
    };

    let error: Provide = |_| Err(Error::Unsupported("ordered provider failure".into()));
    let (root, tensor) = crate::test_support::fixture::source(
        "ordered_providers",
        vec![6].into(),
        Layout::Sparse { axis: None },
        2,
        4096,
        [0u8; 6],
    )
    .await;

    for (providers, expected, fails) in [
        ([none, error, none], [1, 1, 0], true),
        ([none, none, error], [1, 1, 1], true),
        ([error, none, none], [1, 0, 0], true),
        ([none, some, some], [1, 1, 1], false),
        ([some, none, some], [1, 1, 1], false),
        ([some, some, none], [1, 1, 1], false),
    ] {
        let calls = [Counter::new(), Counter::new(), Counter::new()];
        let [condition, left, right] = std::array::from_fn(|i| Provider {
            source: &tensor,
            calls: &calls[i],
            provide: providers[i],
        });
        let selected = condition
            .cond(&left, &right)
            .await
            .unwrap()
            .abs()
            .await
            .unwrap();
        let requests = traversal::ordered(&selected, crate::slice::Slice::full(&[6]).unwrap());
        assert_eq!(requests.is_err(), fails);
        assert_eq!(calls.each_ref().map(|n| n.read()), expected);
        if let Ok(mut requests) = requests {
            let mut coordinates = Vec::new();

            while let Some(request) = requests.try_next().await.unwrap() {
                assert!(request.len() <= MAX_BATCH_ELEMENTS);
                coordinates.extend(request.into_coordinates(&[6]).unwrap());
            }
            assert_eq!(coordinates, (0..6).map(|i| vec![i]).collect::<Vec<_>>());
        }
    }

    for (providers, expected) in [([error, none], [1, 0]), ([none, error], [1, 1])] {
        let calls = [Counter::new(), Counter::new()];
        let [left, right] = std::array::from_fn(|i| Provider {
            source: &tensor,
            calls: &calls[i],
            provide: providers[i],
        });
        assert!(
            traversal::ordered(
                &left.add(&right).await.unwrap(),
                crate::slice::Slice::full(&[6]).unwrap(),
            )
            .is_err()
        );
        assert_eq!(calls.each_ref().map(|n| n.read()), expected);
    }
    crate::test_support::cleanup(&root).await;
}

#[tokio::test]
async fn sparse_output_preserves_order_duplicates_and_nan() {
    use crate::mapping::CoordinateMap;
    use crate::request::{Axis, Cartesian, RequestKind};

    let mapping = CoordinateMap::identity(ha_ndarray::shape![2, 3], &[3, 1])
        .transpose(None)
        .unwrap()
        .flip(0)
        .unwrap();
    let mut high_rank = vec![1; crate::PORTABLE_INLINE_RANK + 2];
    *high_rank.last_mut().unwrap() = 7;
    let pattern = [
        3f32,
        -0.,
        4.,
        f32::from_bits(0x7fc01234),
        0.,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let cases = [
        (
            "explicit",
            BatchRequest::explicit(vec![vec![2], vec![0], vec![2], vec![1], vec![0]]).unwrap(),
            vec![3],
            &pattern[..],
        ),
        (
            "linear batch boundary",
            BatchRequest::linear(MAX_BATCH_ELEMENTS as u64 - 1, MAX_BATCH_ELEMENTS).unwrap(),
            vec![2 * MAX_BATCH_ELEMENTS as u64],
            &pattern[..],
        ),
        (
            "rectangular duplicates",
            BatchRequest::rectangles(vec![
                Cartesian::new(vec![
                    Axis::Selected(vec![2, 0, 2]),
                    Axis::Selected(vec![1, 3]),
                ])
                .unwrap(),
            ])
            .unwrap(),
            vec![3, 4],
            &pattern[..],
        ),
        (
            "transformed flat runs",
            BatchRequest::linear(0, 6)
                .unwrap()
                .mapped(&mapping, &[2, 3], &[3, 1])
                .unwrap(),
            vec![2, 3],
            &pattern[..],
        ),
        (
            "high rank",
            BatchRequest::linear(0, 7).unwrap(),
            high_rank,
            &pattern[..],
        ),
        (
            "empty",
            BatchRequest::linear(0, 0).unwrap(),
            vec![1],
            &pattern[..],
        ),
        (
            "all zeros",
            BatchRequest::linear(0, 7).unwrap(),
            vec![7],
            &[0., -0.][..],
        ),
    ];

    for (name, request, shape, pattern) in cases {
        let explicit = matches!(request.kind(), RequestKind::Explicit(_));
        let expected_coords = request.coordinates(&shape).unwrap();
        let pointers = match request.kind() {
            RequestKind::Explicit(coords) => coords
                .iter()
                .map(|coord| coord.as_ptr())
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };

        let values: Vec<_> = (0..request.len())
            .map(|i| pattern[i % pattern.len()])
            .collect();
        let expected: Vec<_> = expected_coords
            .iter()
            .zip(&values)
            .enumerate()
            .filter(|(_, (_, value))| **value != 0.)
            .map(|(i, (coord, value))| (i, coord.clone(), value.to_bits()))
            .collect();
        crate::read_metrics::CURRENT
            .scope(Default::default(), async {
                let output = sparse_elements(request, EvaluatedBatch { values }, &shape).unwrap();
                crate::read_metrics::CURRENT.with(|m| {
                    assert_eq!(m.borrow().expanded_coordinates, 0, "{name}");
                });
                let output = output.collect::<Result<Vec<_>>>().unwrap();
                assert_eq!(output.len(), expected.len(), "{name}");

                for ((coord, value), (i, expected_coord, bits)) in output.iter().zip(&expected) {
                    assert_eq!(coord, expected_coord, "{name}");
                    assert_eq!(value.to_bits(), *bits, "{name}");
                    if explicit {
                        assert_eq!(coord.as_ptr(), pointers[*i], "{name}");
                    }
                }
                crate::read_metrics::CURRENT.with(|m| {
                    assert_eq!(
                        m.borrow().expanded_coordinates,
                        if explicit { 0 } else { expected.len() },
                        "{name}"
                    );
                });
            })
            .await;
    }
}

#[test]
fn sparse_output_rejects_mismatched_values_and_invalid_requests() {
    let batch = EvaluatedBatch {
        values: vec![1u8, 2],
    };
    assert!(matches!(
        sparse_elements(BatchRequest::point(&[0]), batch, &[1]),
        Err(Error::InvalidLayout(_))
    ));

    for request in [
        BatchRequest::linear(2, 1).unwrap(),
        BatchRequest::linear(2, 0).unwrap(),
        BatchRequest::rectangles(vec![
            request::Cartesian::new(vec![request::Axis::range(1, 1)]).unwrap(),
        ])
        .unwrap(),
    ] {
        let batch = EvaluatedBatch {
            values: vec![0u8; request.len()],
        };
        assert!(matches!(
            sparse_elements(request, batch, &[1]),
            Err(Error::InvalidCoord(_))
        ));
    }
}

#[test]
fn tiled_requests_retain_only_a_bounded_prefix() {
    let shape = [2, 1_000_000_000, 1_000_000_000];
    let request = request::tiled_requests(&shape).unwrap().next().unwrap();
    let coords = request.coordinates(&shape).unwrap();
    assert_eq!(coords.len(), MAX_BATCH_ELEMENTS);
    assert_eq!(coords[32], vec![0, 1, 0]);

    for shape in [vec![2, 33, 35], vec![1], vec![2, 1, 9]] {
        let mut tiled: Vec<_> = request::tiled_requests(&shape)
            .unwrap()
            .flat_map(|request| request.coordinates(&shape).unwrap())
            .collect();
        tiled.sort();
        assert_eq!(
            tiled,
            crate::schema::row_major_coords(&shape)
                .unwrap()
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn malformed_batches_fail_closed() {
    assert!(batch_array(vec![0u8; MAX_BATCH_ELEMENTS + 1]).is_err());
    assert!(Batch::from_values(vec![0u8; MAX_BATCH_ELEMENTS + 1]).is_err());
    let oversized = Array::new(
        Buffer::from(vec![0u8; MAX_BATCH_ELEMENTS + 1]),
        ha_ndarray::shape![MAX_BATCH_ELEMENTS + 1],
    )
    .unwrap();
    assert!(Batch::from_array(ArrayAccess::from(oversized)).is_err());
    let batch = Batch::from_values(vec![1u8, 2]).unwrap();
    assert!(batch.validate(1).is_err());
    let evaluated = EvaluatedBatch { values: vec![1u8] };
    assert!(evaluated.validate(2).is_err());
    let evaluated = EvaluatedBatch {
        values: vec![0u8; MAX_BATCH_ELEMENTS + 1],
    };
    assert!(evaluated.validate(MAX_BATCH_ELEMENTS + 1).is_err());
}

#[test]
fn batching_only_consumes_its_bounded_prefix() {
    use std::cell::Cell;
    let consumed = Cell::new(0);
    let coords = (0..1_000_000_000u64).map(|i| {
        consumed.set(consumed.get() + 1);
        vec![i]
    });
    let mut batches = crate::traits::coordinate_batches(coords);
    assert_eq!(consumed.get(), 0);
    assert_eq!(batches.next().unwrap().len(), MAX_BATCH_ELEMENTS);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
    drop(batches);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
}
