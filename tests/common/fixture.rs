//! Concrete filesystem fixtures and consumer checks. Comparators belong to the
//! numerical case: exact equality, signed zeros, NaNs, and tolerances differ.

use std::path::PathBuf;

use fensor::{
    AxisRange, Layout, Shape, Tensor, TensorElement, TensorFileEntry, TensorRead, TensorSchema,
    TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::Number;

use super::{FsEntry, cleanup, iter_coords, new_dir, open_dir, unique_tmp_dir};

pub fn layout(sparse: bool) -> Layout {
    if sparse {
        Layout::Sparse { axis: None }
    } else {
        Layout::Dense
    }
}

pub async fn source<T: TensorElement>(
    name: &str,
    shape: Shape,
    layout: Layout,
    capacity: usize,
    cache: usize,
    values: impl IntoIterator<Item = T>,
) -> (PathBuf, Tensor<FsEntry, T>)
where
    FsEntry: TensorFileEntry<T>,
{
    let root = unique_tmp_dir(name);
    tokio::fs::create_dir(&root)
        .await
        .expect("create fixture dir");
    let dir = freqfs::Cache::new(cache, None, 0, std::time::Duration::from_secs(1))
        .load(root.clone())
        .unwrap();
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(T::dtype(), shape.clone()).unwrap(),
        layout,
        capacity,
    )
    .await
    .unwrap();
    let mut values = values.into_iter();
    for coord in iter_coords(&shape) {
        let value = values.next().expect("fixture value for every coordinate");
        if matches!(layout, Layout::Dense) || value != T::ZERO {
            tensor.write_value(&coord, value).await.unwrap();
        }
    }
    assert!(values.next().is_none(), "excess fixture values");
    (root, tensor)
}

pub async fn blocks<V: TensorRead>(
    view: &V,
    expected: &[V::DType],
    equal: impl Fn(V::DType, V::DType) -> bool,
) where
    V::DType: TensorElement,
{
    let mut blocks = view.read_blocks().unwrap();
    let mut offset = 0;
    while let Some(values) = blocks.try_next().await.unwrap() {
        // Pin the documented default stream limit independently of private constants.
        assert!(values.len() <= 4096);
        for value in values {
            assert!(offset < expected.len(), "unexpected output at {offset}");
            assert!(
                equal(value, expected[offset]),
                "at {offset}: actual {value:?}, expected {:?}",
                expected[offset]
            );
            offset += 1;
        }
    }
    assert_eq!(offset, expected.len(), "output count");
}

pub async fn consumers<V: TensorRead>(
    view: &V,
    expected: &[V::DType],
    equal: impl Fn(V::DType, V::DType) -> bool,
    copied_equal: impl Fn(V::DType, V::DType) -> bool,
) where
    FsEntry: TensorFileEntry<V::DType>,
    V::DType: TensorElement,
{
    blocks(view, expected, &equal).await;
    for (coord, value) in iter_coords(view.shape()).zip(expected) {
        let actual = view.read_value(&coord).await.unwrap();
        assert!(
            equal(actual, *value),
            "at {coord:?}: {actual:?} != {value:?}"
        );
    }
    let (root, dir) = new_dir("consumer_copy").await;
    let copy: Tensor<FsEntry, V::DType> = Tensor::copy_from(dir.clone(), view, 17).await.unwrap();
    copy.sync().await.unwrap();
    drop(copy);
    drop(dir);
    let copy = Tensor::<FsEntry, V::DType>::load(open_dir(&root).unwrap())
        .await
        .unwrap();
    blocks(&copy, expected, copied_equal).await;
    drop(copy);
    cleanup(&root).await;

    if matches!(view.layout(), Layout::Sparse { .. }) {
        let mut entries = view
            .read_sparse_elements_in_order(
                view.shape()
                    .iter()
                    .map(|d| AxisRange::In(0, *d, 1))
                    .collect(),
                (0..view.ndim()).collect(),
            )
            .await
            .unwrap();
        for (coord, value) in iter_coords(view.shape())
            .zip(expected)
            .filter(|(_, v)| **v != V::DType::ZERO)
        {
            let (actual_coord, actual) = entries.try_next().await.unwrap().expect("nonzero entry");
            assert_eq!(actual_coord, coord);
            assert!(
                equal(actual, *value),
                "at {coord:?}: {actual:?} != {value:?}"
            );
        }
        assert!(entries.try_next().await.unwrap().is_none());
    }
}
