use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fensor::{
    BoxFuture, Layout, NumberType, StorageGeometry, Tensor, TensorArray, TensorExpression,
    TensorGeometry, TensorMathScalar, TensorRead, TensorSchema, TensorSource, TensorTransform,
    TensorView, TensorWrite,
};
use futures::{StreamExt, TryStreamExt};
use number_general::FloatType;

mod common;
use common::{FsEntry, cleanup, new_dir};

#[derive(Clone)]
struct Source {
    tensor: Tensor<FsEntry, f64>,
    first: Vec<f64>,
    scans: Arc<AtomicUsize>,
}
impl TensorGeometry for Source {
    type DType = f64;
    fn dtype(&self) -> NumberType {
        self.tensor.dtype()
    }
    fn layout(&self) -> Layout {
        self.tensor.layout()
    }
    fn shape(&self) -> &[u64] {
        self.tensor.shape()
    }
}
impl TensorArray for Source {
    fn schema(&self) -> &TensorSchema {
        self.tensor.schema()
    }
    fn strides(&self) -> &[u64] {
        self.tensor.strides()
    }
}
impl TensorSource for Source {
    fn storage_geometry(&self) -> StorageGeometry {
        self.tensor.storage_geometry()
    }
    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, fensor::Result<Vec<f64>>> {
        Box::pin(async move {
            if id == 0 {
                Ok(self.first.clone())
            } else {
                self.tensor.read_logical_block(id).await
            }
        })
    }
    fn occupied_regions(
        &self,
        after: Option<[u64; 2]>,
        lo: u64,
        hi: u64,
    ) -> futures::stream::BoxStream<'static, fensor::Result<[u64; 2]>> {
        self.scans.fetch_add(1, Ordering::Relaxed);
        let mut keys = self.tensor.occupied_regions(after, lo, hi);
        let mut ended = false;
        futures::stream::poll_fn(move |cx| {
            assert!(!ended, "custom occupied scan polled after EOF");
            let next = keys.as_mut().poll_next(cx);
            ended = matches!(next, std::task::Poll::Ready(None));
            next
        })
        .boxed()
    }
}

#[tokio::test]
async fn custom_owned_source_feeds_existing_lazy_operations() {
    let (root, dir) = new_dir("owned_source").await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F64), vec![2, 2].into()).unwrap();
    let tensor = Tensor::create(dir, schema, Layout::Dense, 2).await.unwrap();
    tensor.write_value(&[1, 0], 3.0).await.unwrap();
    tensor.write_value(&[1, 1], 4.0).await.unwrap();
    let expression = {
        let source = Source {
            tensor: tensor.clone(),
            first: vec![7.0, 8.0],
            scans: Arc::default(),
        };
        let view = TensorView::new(source).transpose(None).unwrap();
        TensorExpression::new(view.add_scalar(1.0).await.unwrap()).unwrap()
    };
    assert_eq!(tensor.read_value(&[0, 0]).await.unwrap(), 0.0);
    drop(tensor);
    let values = expression
        .into_blocks()
        .unwrap()
        .try_concat()
        .await
        .unwrap();
    assert_eq!(values, [8.0, 4.0, 9.0, 5.0]);
    cleanup(&root).await;
}

#[tokio::test]
async fn strict_load_never_initializes_missing_storage() {
    let (root, dir) = new_dir("strict_empty").await;
    assert!(Tensor::<FsEntry, f64>::load(dir.clone()).await.is_err());
    assert!(dir.read().await.is_empty());
    let schema = TensorSchema::new(NumberType::Float(FloatType::F64), vec![4].into()).unwrap();
    let tensor = Tensor::<FsEntry, f64>::create(dir.clone(), schema, Layout::Dense, 2)
        .await
        .unwrap();
    tensor.sync().await.unwrap();
    let blocks = dir.read().await.get_dir("blocks").cloned().unwrap();
    blocks.write().await.delete("0").await;
    assert!(Tensor::<FsEntry, f64>::load(dir.clone()).await.is_err());
    assert!(!blocks.read().await.contains("0"));
    drop(tensor);
    cleanup(&root).await;
}

#[tokio::test]
async fn geometric_block_bounds_enclose_transformed_coordinates_without_expansion() {
    use fensor::AxisRange;
    let (root, dir) = new_dir("block_bounds").await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F64), vec![4, 8].into()).unwrap();
    let tensor = Tensor::<FsEntry, f64>::create(dir, schema, Layout::Dense, 4)
        .await
        .unwrap();
    let geometry = tensor.storage_geometry();
    let point = tensor
        .view()
        .slice(vec![AxisRange::At(2), AxisRange::At(7)].into())
        .unwrap();
    let id = geometry.block_position(&[2, 7]).unwrap().0;
    assert_eq!(point.logical_block_range(&geometry).unwrap(), id..id + 1);
    for view in [
        tensor
            .view()
            .slice(vec![AxisRange::At(0), AxisRange::In(0, 4, 1)].into())
            .unwrap(),
        tensor.view().transpose(None).unwrap(),
        tensor.view().flip(1).unwrap(),
        tensor
            .view()
            .slice(vec![AxisRange::Of(vec![3, 1]), AxisRange::In(1, 8, 2)].into())
            .unwrap(),
    ] {
        let bounds = view.logical_block_range(&geometry).unwrap();
        for coord in fensor::row_major_coords(view.shape()).unwrap() {
            let block = geometry
                .block_position(&view.resolve_base_coord(&coord).unwrap())
                .unwrap()
                .0;
            assert!(bounds.contains(&block));
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn owned_and_borrowed_consumers_preserve_geometry_support_and_nonfinite_values() {
    fn same(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!(left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan()));
        }
    }
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, tensor) = common::fixture::source(
            "owned_consumers",
            vec![2, 3].into(),
            layout,
            2,
            1_000_000,
            [0.0f64, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 3.0],
        )
        .await;
        // Subtraction retains the original support of the stored value 1, even
        // though its intermediate value is zero. The outer addition observes it.
        let inner = tensor.view().sub_scalar(1.0).await.unwrap();
        let expression = TensorExpression::new(inner.add_scalar(2.0).await.unwrap()).unwrap();
        for value in [
            expression.clone(),
            expression.transpose(None).unwrap().flip(0).unwrap(),
        ] {
            let borrowed = value.read_blocks().unwrap().try_concat().await.unwrap();
            let owned = value
                .clone()
                .into_blocks()
                .unwrap()
                .try_concat()
                .await
                .unwrap();
            same(&borrowed, &owned);
            let entries: Vec<_> = value
                .clone()
                .into_sparse_elements()
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
            if matches!(layout, Layout::Sparse { .. }) {
                let range = value
                    .shape()
                    .iter()
                    .map(|&dim| fensor::AxisRange::In(0, dim, 1))
                    .collect();
                let order = (0..value.ndim()).collect();
                let borrowed: Vec<_> = value
                    .read_sparse_elements_in_order(range, order)
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                assert_eq!(
                    entries.iter().map(|(coord, _)| coord).collect::<Vec<_>>(),
                    borrowed.iter().map(|(coord, _)| coord).collect::<Vec<_>>()
                );
                same(
                    &entries.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
                    &borrowed.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
                );
                assert!(entries.iter().any(|(_, value)| *value == 2.0));
                assert_eq!(entries.len(), 5);
            }
        }
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn sparse_consumption_retains_one_source_scan_across_pages_and_batches() {
    // The final partial output batch encounters EOF before delivery. Consuming
    // the following output poll must not poll this custom source after EOF.
    let rows = 4097;
    let (_root, tensor) = common::fixture::source(
        "owned_occupied_scan",
        vec![rows, 2].into(),
        Layout::Sparse { axis: Some(0) },
        2,
        1_000_000,
        std::iter::repeat_n(2.0f64, rows as usize * 2),
    )
    .await;
    let scans = Arc::default();
    let source = Source {
        tensor,
        first: vec![2.; 2],
        scans: Arc::clone(&scans),
    };
    let expression = TensorExpression::new(TensorView::new(source)).unwrap();
    let mut entries = expression.into_sparse_elements().unwrap();
    let mut count = 0;
    while let Some((coord, value)) = entries.try_next().await.unwrap() {
        assert_eq!(coord, [count / 2, count % 2]);
        assert_eq!(value, 2.);
        count += 1;
    }
    assert_eq!(count, rows * 2);
    assert_eq!(scans.load(Ordering::Relaxed), 1);
}
