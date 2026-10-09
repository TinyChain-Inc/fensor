use crate::expression::{self, MAX_BATCH_ELEMENTS};
use crate::request::BatchRequest;
use crate::test_support::{cleanup, fixture};
use crate::{Layout, TensorGeometry, TensorMatMul, TensorMatrixUnary, TensorUnary};

#[tokio::test]
async fn projection_preserves_values_in_one_bounded_source_request() {
    let (root, tensor) = fixture::source(
        "diag_batch",
        smallvec::smallvec![2, 2],
        Layout::Sparse { axis: None },
        2,
        4096,
        [1.2f32, 7., 0., 0.],
    )
    .await;
    let product = tensor.view().matmul(&tensor.view()).await.unwrap();
    assert!(
        crate::expression::traversal::preferred(&product, product.shape())
            .unwrap()
            .is_some()
    );
    let diagonal = product.diag().await.unwrap();
    assert!(
        crate::expression::traversal::preferred(&diagonal, diagonal.shape())
            .unwrap()
            .is_none()
    );
    let view = tensor.view().round().await.unwrap().diag().await.unwrap();
    let request = BatchRequest::explicit(
        (0..MAX_BATCH_ELEMENTS)
            .map(|i| vec![(i % 2) as u64])
            .collect(),
    )
    .unwrap();
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let batch = expression::evaluate_batch(&view, &request).await.unwrap();

            for (i, value) in batch.values.into_iter().enumerate() {
                let expected = u8::from(i % 2 == 0);
                assert_eq!(value, expected as f32);
            }
            crate::read_metrics::CURRENT.with(|metrics| {
                let metrics = metrics.borrow();
                assert_eq!(metrics.requested, MAX_BATCH_ELEMENTS);
                assert_eq!(metrics.logical_payload_reads, 2);
            });
        })
        .await;
    cleanup(&root).await;
}
