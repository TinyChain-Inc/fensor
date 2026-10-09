use fensor::{
    BoxFuture, Layout, NumberType, StorageGeometry, Tensor, TensorArray, TensorExpression,
    TensorGeometry, TensorRead, TensorSchema, TensorSource, TensorStatistics, TensorTransform,
    TensorTrig, TensorUnary, TensorView, TensorWrite,
};
use futures::{TryStreamExt, stream::BoxStream};
use number_general::DType;

mod common;
use common::{FsEntry, fixture, new_dir, open_dir};

#[derive(Clone)]
struct Source(Tensor<FsEntry, f64>);

impl TensorGeometry for Source {
    type DType = f64;

    fn dtype(&self) -> NumberType {
        self.0.dtype()
    }

    fn layout(&self) -> Layout {
        self.0.layout()
    }

    fn shape(&self) -> &[u64] {
        self.0.shape()
    }
}

impl TensorArray for Source {
    fn schema(&self) -> &TensorSchema {
        self.0.schema()
    }

    fn strides(&self) -> &[u64] {
        self.0.strides()
    }
}

impl TensorSource for Source {
    fn storage_geometry(&self) -> StorageGeometry {
        self.0.storage_geometry()
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, fensor::Result<Vec<f64>>> {
        self.0.read_logical_block(id)
    }

    fn occupied_blocks(
        &self,
        range: std::ops::Range<u64>,
    ) -> BoxStream<'static, fensor::Result<u64>> {
        self.0.occupied_blocks(range)
    }
}

#[tokio::test]
async fn sparse_axis_selects_dense_suffix_and_scalar_payloads() {
    for (axis, block) in [
        (None, [1, 1, 1]),
        (Some(0), [1, 3, 5]),
        (Some(1), [1, 1, 5]),
        (Some(2), [1, 1, 1]),
    ] {
        let (root, dir) = new_dir("dense_sparse_geometry").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), vec![2, 3, 5].into()).unwrap(),
            Layout::Sparse { axis },
            16,
        )
        .await
        .unwrap();
        assert_eq!(tensor.storage_geometry().block_shape(), block);
        tensor.write_value(&[1, 2, 4], 7.).await.unwrap();
        tensor.sync().await.unwrap();
        drop(tensor);

        let tensor = Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(tensor.storage_geometry().block_shape(), block);
        assert_eq!(tensor.read_value(&[1, 2, 4]).await.unwrap(), 7.);
        assert_eq!(tensor.read_value(&[0, 0, 0]).await.unwrap(), 0.);
    }
}

#[tokio::test]
async fn dense_sparse_payloads_preserve_values_padding_and_reopening() {
    let (root, dir) = new_dir("dense_sparse_payloads").await;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), vec![2, 5].into()).unwrap(),
        Layout::Sparse { axis: Some(0) },
        4,
    )
    .await
    .unwrap();
    assert_eq!(tensor.storage_geometry().block_shape(), [1, 4]);
    tensor
        .replace_logical_block(0, vec![0., 2., 0., -0.])
        .await
        .unwrap();
    assert_eq!(
        tensor
            .occupied_blocks(0..tensor.storage_geometry().block_count())
            .try_collect::<Vec<_>>()
            .await
            .unwrap(),
        [0]
    );

    let count = tensor.storage_geometry().block_count();
    for range in [count..count, 1..1] {
        assert!(
            tensor
                .occupied_blocks(range)
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
    }
    for range in [count..0, 0..count + 1] {
        assert!(tensor.occupied_blocks(range).try_next().await.is_err());
    }
    assert_eq!(tensor.read_logical_block(1).await.unwrap(), vec![0.; 4]);

    let native = TensorExpression::new(tensor.clone()).unwrap();
    assert!(matches!(
        native.exp().await,
        Err(fensor::Error::WouldDensify { operation: "exp" })
    ));
    let dense = native.clone().into_dense();
    assert_eq!(dense.layout(), Layout::Dense);
    assert_eq!(native.layout(), Layout::Sparse { axis: Some(0) });
    let transposed = native.clone().transpose(None).unwrap();
    assert_eq!(transposed.layout(), Layout::Sparse { axis: None });
    assert_eq!(
        transposed.transpose(None).unwrap().layout(),
        Layout::Sparse { axis: Some(0) }
    );
    let expected = [1., 2.0_f64.exp(), 1., 1., 1., 1., 1., 1., 1., 1.];
    fixture::reads(&dense.exp().await.unwrap(), &expected, |a, b| a == b).await;
    let view = TensorView::new(Source(tensor.clone()));
    assert!(matches!(
        view.exp().await,
        Err(fensor::Error::WouldDensify { operation: "exp" })
    ));
    let expected = [0., 2.0_f64.sin(), 0., 0., 0., 0., 0., 0., 0., 0.];
    fixture::reads(&native.sin().await.unwrap(), &expected, |a, b| a == b).await;
    fixture::reads(&view.sin().await.unwrap(), &expected, |a, b| a == b).await;
    assert_eq!(native.mean_all().await.unwrap(), 0.2);
    assert_eq!(
        view.mean_all().await.unwrap(),
        dense.mean_all().await.unwrap()
    );
    drop(view);
    drop(native);
    drop(dense);

    tensor.write_value(&[0, 1], 0.).await.unwrap();
    assert!(
        tensor
            .occupied_blocks(0..tensor.storage_geometry().block_count())
            .try_next()
            .await
            .unwrap()
            .is_none()
    );
    fixture::blocks(&tensor.view().sin().await.unwrap(), &[0.; 10], |a, b| {
        a == b
    })
    .await;

    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    tensor
        .replace_logical_block(0, vec![0., nan, f64::INFINITY, f64::NEG_INFINITY])
        .await
        .unwrap();
    tensor
        .replace_logical_block(1, vec![3., 0., 0., 0.])
        .await
        .unwrap();
    assert!(
        tensor
            .replace_logical_block(1, vec![3., 0., 9., 0.])
            .await
            .is_err()
    );
    assert_eq!(tensor.read_value(&[0, 4]).await.unwrap(), 3.);
    let mut detached = tensor.read_logical_block(0).await.unwrap();
    detached[1] = 9.;
    assert_eq!(
        tensor.read_value(&[0, 1]).await.unwrap().to_bits(),
        nan.to_bits()
    );

    let clone = tensor.clone();
    let view = clone.view();
    let mut stream = view.read_blocks().unwrap();
    assert!(stream.try_next().await.unwrap().is_some());
    drop(stream);
    drop(view);
    drop(tensor);
    clone.write_value(&[0, 0], 5.).await.unwrap();
    clone.validate().await.unwrap();
    clone.sync().await.unwrap();
    drop(clone);

    let loaded = Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
        .await
        .unwrap();
    fixture::blocks(
        &loaded,
        &[
            5.,
            nan,
            f64::INFINITY,
            f64::NEG_INFINITY,
            3.,
            0.,
            0.,
            0.,
            0.,
            0.,
        ],
        |a, b| a.to_bits() == b.to_bits(),
    )
    .await;
}

#[tokio::test]
async fn dense_sparse_values_are_independent_of_chunk_capacity() {
    let mut high_rank = vec![1; fensor::Shape::new().inline_size() + 2];
    let rank = high_rank.len();
    high_rank[rank - 2] = 2;
    high_rank[rank - 1] = 5;

    for (shape, axis, capacity) in [
        (vec![1, 8], 0, 2),
        (vec![1, 8], 0, 4),
        (vec![1, 8], 0, 16),
        (vec![2, 2, 8], 1, 4),
        (high_rank, rank - 2, 4),
    ] {
        let (root, dir) = new_dir("dense_sparse_region").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), shape.clone().into()).unwrap(),
            Layout::Sparse { axis: Some(axis) },
            capacity,
        )
        .await
        .unwrap();
        let mut coordinate = vec![0; shape.len()];
        for prefix in 0..=axis {
            coordinate[prefix] = shape[prefix] - 1;
        }
        coordinate[axis + 1] = 1;
        tensor.write_value(&coordinate, 2.).await.unwrap();
        let expected: Vec<_> = common::iter_coords(&shape)
            .map(|coord| {
                if coord == coordinate {
                    2.0_f64.exp()
                } else {
                    1.
                }
            })
            .collect();
        let occupied = tensor
            .occupied_blocks(0..tensor.storage_geometry().block_count())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let geometry = tensor.storage_geometry();
        let first = geometry.block_position(&coordinate).unwrap().0;
        assert_eq!(occupied, [first]);
        if geometry.block_count() > 1 {
            let zero = (0..geometry.block_count()).find(|&id| id != first).unwrap();
            assert!(
                tensor
                    .read_logical_block(zero)
                    .await
                    .unwrap()
                    .iter()
                    .all(|&value| value == 0.)
            );
        }
        fixture::reads(
            &TensorExpression::new(TensorView::new(Source(tensor.clone())))
                .unwrap()
                .into_dense()
                .exp()
                .await
                .unwrap(),
            &expected,
            |a, b| a == b,
        )
        .await;
        tensor.sync().await.unwrap();
        drop(tensor);

        let tensor = Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(
            tensor
                .occupied_blocks(0..tensor.storage_geometry().block_count())
                .try_collect::<Vec<_>>()
                .await
                .unwrap(),
            occupied
        );
        fixture::reads(
            &TensorExpression::new(tensor.clone())
                .unwrap()
                .into_dense()
                .exp()
                .await
                .unwrap(),
            &expected,
            |a, b| a == b,
        )
        .await;
        tensor.write_value(&coordinate, 0.).await.unwrap();
        assert!(
            tensor
                .occupied_blocks(0..tensor.storage_geometry().block_count())
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
        fixture::blocks(
            &tensor.view().sin().await.unwrap(),
            &vec![0.; expected.len()],
            |a, b| a == b,
        )
        .await;
        tensor.sync().await.unwrap();
        drop(tensor);
        let tensor = Tensor::<FsEntry, f64>::load(open_dir(&root).unwrap())
            .await
            .unwrap();
        assert!(
            tensor
                .occupied_blocks(0..tensor.storage_geometry().block_count())
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
    }
}

#[tokio::test]
async fn lazy_dense_conversion_includes_implicit_zeros_across_batches() {
    let (_root, dir) = new_dir("lazy_dense_conversion").await;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), vec![2, 4097].into()).unwrap(),
        Layout::Sparse { axis: Some(0) },
        4096,
    )
    .await
    .unwrap();
    tensor.write_value(&[0, 0], 2.).await.unwrap();
    let sparse = TensorExpression::new(tensor).unwrap();
    let dense = sparse.clone().into_dense();
    assert_eq!(sparse.layout(), Layout::Sparse { axis: Some(0) });
    assert_eq!(dense.layout(), Layout::Dense);
    assert!(matches!(
        sparse.exp().await,
        Err(fensor::Error::WouldDensify { operation: "exp" })
    ));
    let output = TensorExpression::new(dense.exp().await.unwrap()).unwrap();
    let mut entries = output.into_sparse_elements().unwrap();
    for coordinate in common::iter_coords(&[2, 4097]) {
        let (actual, value) = entries.try_next().await.unwrap().unwrap();
        assert_eq!(actual, coordinate);
        assert_eq!(
            value,
            if coordinate == [0, 0] {
                2.0_f64.exp()
            } else {
                1.
            }
        );
    }
    assert!(entries.try_next().await.unwrap().is_none());
}
