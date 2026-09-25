//! Logical geometry is independent of backend allocation and pointer width.

use fensor::{
    AxisRange, Error, Layout, Shape, Strides, Tensor, TensorGeometry, TensorMatMul, TensorRead,
    TensorReduce, TensorReduceAll, TensorSchema, TensorTransform, TensorWrite, contiguous_strides,
};
use futures::TryStreamExt;
use number_general::DType;
use smallvec::smallvec;

mod common;

#[test]
fn inline_rank_is_not_a_rank_limit() {
    for rank in [7, 8, 9] {
        let shape = Shape::from_elem(1, rank);
        let strides = contiguous_strides(&shape).unwrap();
        assert_eq!(shape.spilled(), rank > 8);
        assert_eq!(strides.spilled(), rank > 8);
        assert_eq!(strides, Strides::from_elem(1, rank));
        assert_eq!(
            TensorSchema::new(u8::dtype(), shape.clone())
                .unwrap()
                .shape(),
            &shape
        );
    }
}

#[test]
fn cardinality_overflow_is_rejected() {
    assert!(matches!(
        TensorSchema::new(u8::dtype(), smallvec![u64::MAX, 2]),
        Err(Error::InvalidSchema(_))
    ));
    assert!(contiguous_strides(&[2, u64::MAX, 2]).is_err());
}

#[tokio::test]
async fn large_logical_coordinates_survive_transforms_and_reload() {
    let (root, dir) = common::new_dir("u64_geometry").await;
    let tensor = Tensor::<common::FsEntry, u8>::create(
        dir,
        TensorSchema::new(u8::dtype(), smallvec![u64::MAX]).unwrap(),
        Layout::Sparse { axis: None },
        7,
    )
    .await
    .unwrap();
    let at = i64::MAX as u64 + 17;
    tensor.write_value(&[at], 127).await.unwrap();
    tensor.write_value(&[u64::MAX - 1], 255).await.unwrap();
    assert_eq!(tensor.size().unwrap(), u64::MAX);
    assert_eq!(tensor.read_value(&[at]).await.unwrap(), 127);
    assert_eq!(
        tensor
            .view()
            .flip(0)
            .unwrap()
            .read_value(&[0])
            .await
            .unwrap(),
        255
    );

    let selected = tensor
        .view()
        .slice(smallvec![AxisRange::In(at, at + 3, 1)])
        .unwrap();
    let blocks: Vec<_> = selected
        .read_coordinate_blocks()
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(
        blocks,
        vec![(vec![vec![0], vec![1], vec![2]], vec![127, 0, 0])]
    );
    assert_eq!(selected.sum_all().await.unwrap(), 127);
    let gather = tensor
        .view()
        .slice(smallvec![AxisRange::Of(vec![at, u64::MAX - 1, at])])
        .unwrap();
    assert_eq!(
        gather
            .clone()
            .flip(0)
            .unwrap()
            .read_value(&[1])
            .await
            .unwrap(),
        255
    );
    assert!(
        tensor
            .view()
            .slice(smallvec![AxisRange::In(at, u64::MAX, 0)])
            .is_err()
    );
    assert!(tensor.read_value(&[u64::MAX]).await.is_err());

    let (right_root, right_dir) = common::new_dir("u64_matrix_operand").await;
    let right = Tensor::<common::FsEntry, u8>::create(
        right_dir,
        TensorSchema::new(u8::dtype(), smallvec![1, 1]).unwrap(),
        Layout::Dense,
        1,
    )
    .await
    .unwrap();
    right.write_value(&[0, 0], 1).await.unwrap();
    let matrix = tensor.view().reshape(smallvec![u64::MAX, 1]).unwrap();
    let product = matrix.matmul(&right.view()).await.unwrap();
    assert_eq!(product.read_value(&[at, 0]).await.unwrap(), 127);
    let reduced = matrix.sum(smallvec![1], false).await.unwrap();
    assert_eq!(reduced.read_value(&[at]).await.unwrap(), 127);
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(10), tensor.sum_all(),)
            .await
            .unwrap()
            .unwrap(),
        126
    );
    drop(reduced);
    drop(product);
    drop(matrix);
    drop(right);
    common::cleanup(&right_root).await;

    tensor.sync().await.unwrap();
    drop(gather);
    drop(selected);
    drop(tensor);
    let reopened = Tensor::<common::FsEntry, u8>::load(common::open_dir(&root).unwrap())
        .await
        .unwrap();
    assert_eq!(reopened.size().unwrap(), u64::MAX);
    assert_eq!(reopened.read_value(&[at]).await.unwrap(), 127);
    drop(reopened);
    common::cleanup(&root).await;
}

#[tokio::test]
async fn spilled_geometry_preserves_access_and_transforms() {
    let (root, dir) = common::new_dir("spilled_geometry").await;
    let tensor = Tensor::<common::FsEntry, u8>::create(
        dir,
        TensorSchema::new(u8::dtype(), Shape::from_elem(1, 9)).unwrap(),
        Layout::Dense,
        1,
    )
    .await
    .unwrap();
    tensor.write_value(&[0; 9], 127).await.unwrap();
    let view = tensor
        .view()
        .transpose(None)
        .unwrap()
        .unsqueeze(smallvec![0])
        .unwrap();
    assert_eq!(view.shape(), &[1; 10]);
    assert_eq!(view.read_value(&[0; 10]).await.unwrap(), 127);
    assert_eq!(view.sum_all().await.unwrap(), 127);
    drop(view);
    drop(tensor);
    common::cleanup(&root).await;
}
