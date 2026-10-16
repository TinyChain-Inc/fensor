use futures::TryStreamExt;
use ha_ndarray::{axes, shape};
use number_general::DType;

use super::*;
use crate::AxisRange;
use crate::TensorSource;
use crate::test_support::{FsEntry, new_dir};
use crate::{
    Layout, Tensor, TensorRead, TensorReduce, TensorReduceAll, TensorSchema, TensorTransform,
    TensorUnary, TensorWrite,
};

#[test]
fn ordered_coverage_and_permanent_exhaustion() {
    let slice = Slice::new(
        &[3, 6000],
        vec![
            Axis::Selected(vec![2, 0, 2]),
            Axis::Span {
                start: 1,
                step: 2,
                len: 2999,
            },
        ],
    )
    .unwrap();
    let mut requests = slice.requests();
    let mut actual = Vec::new();

    for request in requests.by_ref() {
        assert!(request.len() <= MAX_BATCH_ELEMENTS);
        actual.extend(request.coordinates(&[3, 6000]).unwrap());
    }

    let expected: Vec<_> = [2, 0, 2]
        .into_iter()
        .flat_map(|row| (0..2999).map(move |i| vec![row, 1 + i * 2]))
        .collect();
    assert_eq!(actual, expected);
    assert!(requests.next().is_none());
    assert!(requests.next().is_none());
    assert!(
        Slice::new(&[4], vec![Axis::range(4, 0)])
            .unwrap()
            .requests()
            .next()
            .is_none()
    );
    assert!(Slice::full(&[]).is_err());
    let mut scalar = Slice::new(&[], vec![]).unwrap().requests();
    assert_eq!(scalar.next().unwrap().len(), 1);
    assert!(scalar.next().is_none());
}

#[test]
fn validation_and_huge_bounded_prefix() {
    assert!(Slice::new(&[2], vec![]).is_err());
    let slice = Slice::full(&[3]).unwrap();
    assert!(slice.intersect(&[]).is_err());
    assert!(slice.intersect(&[(2, 1)]).is_err());
    assert!(
        Slice::new(
            &[10],
            vec![Axis::Span {
                start: 0,
                step: 0,
                len: 1
            }]
        )
        .is_err()
    );
    assert!(
        Slice::new(
            &[u64::MAX],
            vec![Axis::Span {
                start: 1,
                step: u64::MAX,
                len: 2
            }]
        )
        .is_err()
    );
    assert!(Slice::new(&[3], vec![Axis::Selected(vec![3])]).is_err());
    assert!(Slice::full(&[u64::MAX, 2]).is_err());
    let slice = Slice::full(&[1_000_000_000, 1_000_000_000]).unwrap();
    let request = slice.requests().next().unwrap();
    assert_eq!(request.len(), MAX_BATCH_ELEMENTS);
    assert!(matches!(
        request.kind(),
        crate::request::RequestKind::Rectangles(_)
    ));
}

#[tokio::test]
async fn huge_sparse_slices_keep_zero_candidates_and_release_guards() {
    let (root, dir) = new_dir("indexed_reduction").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![2, 1_000_000_000]).unwrap(),
        Layout::Sparse { axis: None },
        7,
    )
    .await
    .unwrap();
    tensor.write_value(&[0, 999_999_998], 0.2).await.unwrap();
    tensor.write_value(&[1, 3], 2.).await.unwrap();
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            assert_eq!(tensor.sum_all().await.unwrap(), 2.2);
            let round = tensor.view().round().await.unwrap();
            let result = round.sum(axes![1], false).await.unwrap();
            assert_eq!(result.read_value(&[0]).await.unwrap(), 0.);
            let slice = tensor
                .view()
                .slice(vec![AxisRange::At(0), AxisRange::In(2, 1_000_000_000, 2)].into())
                .unwrap();
            assert_eq!(slice.sum_all().await.unwrap(), 0.2);
            crate::read_metrics::CURRENT.with(|m| {
                assert!(m.borrow().index_entries <= 4);
                assert!(m.borrow().requested <= 32);
            });
        })
        .await;
    let mut requests =
        crate::expression::traversal::selection(&tensor, Slice::full(&[2, 1_000_000_000]).unwrap())
            .unwrap();
    assert!(requests.try_next().await.unwrap().is_some());
    tensor.write_value(&[1, 4], 3.).await.unwrap();
    drop(requests);
    assert_eq!(tensor.sum_all().await.unwrap(), 5.2);
    crate::test_support::cleanup(&root).await;
}

#[tokio::test]
async fn packed_dense_groups_share_one_block() {
    let (root, dir) = new_dir("packed_reduction").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![64, 4]).unwrap(),
        Layout::Dense,
        crate::schema::MAX_BLOCK_CAPACITY,
    )
    .await
    .unwrap();

    for row in 0..64 {
        tensor.write_value(&[row, 0], row as f32).await.unwrap();
    }

    let reduced = tensor.view().sum(axes![1], false).await.unwrap();
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let request = BatchRequest::linear(0, 64).unwrap();
            let batch = crate::expression::evaluate_batch(&reduced, &request)
                .await
                .unwrap();
            assert_eq!(batch.values, (0..64).map(|i| i as f32).collect::<Vec<_>>());
            crate::read_metrics::CURRENT.with(|m| assert_eq!(m.borrow().borrows, 1));
        })
        .await;
    crate::test_support::cleanup(&root).await;
}

#[tokio::test]
async fn sparse_regions_filter_corruption_and_validate_grid_keys() {
    let (root, dir) = new_dir("slice_corruption").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(f32::dtype(), shape![2, 10_000]).unwrap(),
        Layout::Sparse { axis: None },
        7,
    )
    .await
    .unwrap();
    tensor.write_value(&[0, 0], 1.).await.unwrap();
    tensor.write_value(&[0, 9_999], 2.).await.unwrap();
    crate::tensor::corruption::corrupt_sparse_payload(&tensor, 0).await;
    let slice = tensor
        .view()
        .slice(vec![AxisRange::At(0), AxisRange::In(5000, 10000, 1)].into())
        .unwrap();
    assert_eq!(slice.sum_all().await.unwrap(), 2.);
    assert!(tensor.sum_all().await.is_err());
    crate::tensor::corruption::corrupt_sparse_key(&tensor, 0, 9_999).await;
    assert!(matches!(
        tensor
            .occupied_blocks(0..tensor.storage_geometry().block_count())
            .try_collect::<Vec<_>>()
            .await,
        Err(Error::InvalidLayout(_))
    ));
    assert!(tensor.validate().await.is_err());
    crate::test_support::cleanup(&root).await;
}

#[tokio::test]
async fn occupied_slices_respect_shared_pages_and_live_sources() {
    let (root, dir) = new_dir("slice_alias").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![2, 10000]).unwrap(),
        Layout::Sparse { axis: None },
        7,
    )
    .await
    .unwrap();
    tensor.write_value(&[0, 0], 2.).await.unwrap();
    tensor.write_value(&[0, 1], 3.).await.unwrap();
    tensor.write_value(&[1, 0], 2.).await.unwrap();
    tensor.write_value(&[1, 1], 3.).await.unwrap();
    assert_eq!(tensor.sum_all().await.unwrap(), 10.);
    let transposed = tensor.view().transpose(None).unwrap();
    assert_eq!(transposed.product_all().await.unwrap(), 0.);
    tensor.write_value(&[1, 0], 0.).await.unwrap();
    tensor.write_value(&[1, 1], 0.).await.unwrap();
    assert_eq!(transposed.sum_all().await.unwrap(), 5.);
    crate::test_support::cleanup(&root).await;
}
