mod common;

use fensor::{
    Layout, Tensor, TensorRead, TensorReduce, TensorReduceAll, TensorSchema, TensorTransform,
    TensorWrite,
};
use ha_ndarray::{axes, shape};
use number_general::DType;

macro_rules! indexed_dtype {
    ($test:ident,$dtype:ty) => {
        #[tokio::test]
        async fn $test() {
            for axis in [None, Some(0), Some(1)] {
                let (root, dir) = common::new_dir("slice_dtype").await;
                let tensor = Tensor::<common::FsEntry, $dtype>::create(
                    dir,
                    TensorSchema::new(<$dtype>::dtype(), shape![2, 8193]).unwrap(),
                    Layout::Sparse { axis },
                    31,
                )
                .await
                .unwrap();
                tensor.write_value(&[0, 1], 2 as $dtype).await.unwrap();
                tensor.write_value(&[0, 8192], 3 as $dtype).await.unwrap();
                assert_eq!(tensor.sum_all().await.unwrap(), 5 as $dtype);
                assert_eq!(tensor.product_all().await.unwrap(), 6 as $dtype);
                assert_eq!(tensor.min_all().await.unwrap(), 2 as $dtype);
                assert_eq!(tensor.max_all().await.unwrap(), 3 as $dtype);

                for view in [tensor.view(), tensor.view().flip(1).unwrap()] {
                    let reduced = view.product(axes![1], false).await.unwrap();
                    assert_eq!(reduced.read_value(&[0]).await.unwrap(), 6 as $dtype);
                    assert_eq!(reduced.read_value(&[1]).await.unwrap(), 0 as $dtype);
                }
                tensor.sync().await.unwrap();
                drop(tensor);
                let loaded =
                    Tensor::<common::FsEntry, $dtype>::load(common::open_dir(&root).unwrap())
                        .await
                        .unwrap();
                assert_eq!(loaded.sum_all().await.unwrap(), 5 as $dtype);
                common::cleanup(&root).await;
            }
        }
    };
}

indexed_dtype!(indexed_f32, f32);
indexed_dtype!(indexed_f64, f64);
indexed_dtype!(indexed_u8, u8);

// Index topology is dtype-independent; the fixtures above own dtype parity.
#[tokio::test]
async fn indexed_topology() {
    // Enough keys to split the index: successor seeks must include leaf
    // entries between internal separators, for both sparse-axis orders.
    for axis in [None, Some(1)] {
        for capacity in [1, 7, 31, 4096] {
            for (rows, columns) in [(32usize, 129usize), (1, 8193)] {
                let (root, dir) = common::new_dir("slice_index_nodes").await;
                let tensor = Tensor::<common::FsEntry, f32>::create(
                    dir,
                    TensorSchema::new(f32::dtype(), shape![rows, columns]).unwrap(),
                    Layout::Sparse { axis },
                    capacity,
                )
                .await
                .unwrap();

                for flat in (0..rows * columns).step_by(97) {
                    tensor
                        .write_value(&[(flat / columns) as u64, (flat % columns) as u64], 1.)
                        .await
                        .unwrap();
                }

                let expected = (rows * columns).div_ceil(97) as f32;
                assert_eq!(tensor.sum_all().await.unwrap(), expected);
                let transposed = tensor.view().transpose(None).unwrap();
                assert_eq!(transposed.sum_all().await.unwrap(), expected);
                assert_eq!(
                    transposed
                        .sum(axes![0], false)
                        .await
                        .unwrap()
                        .sum_all()
                        .await
                        .unwrap(),
                    expected
                );
                tensor.sync().await.unwrap();
                drop(tensor);
                let loaded = Tensor::<common::FsEntry, f32>::load(common::open_dir(&root).unwrap())
                    .await
                    .unwrap();
                assert_eq!(loaded.sum_all().await.unwrap(), expected);
                common::cleanup(&root).await;
            }
        }
    }
}
