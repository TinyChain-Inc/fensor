//! One storage constructor consumes base tensors and geometric views as readers.

use common::{FsEntry, new_dir};
use fensor::{
    DType, Error, Layout, Tensor, TensorArray, TensorGeometry, TensorRead, TensorSchema,
    TensorTransform, TensorWrite,
};
use ha_ndarray::{AxisRange, axes, range, shape};

mod common;

#[tokio::test]
async fn copy_base_and_geometric_readers_into_independent_storage() {
    for layout in [Layout::Dense, Layout::Sparse { axis: Some(1) }] {
        let (root, dir) = new_dir("copy_source").await;
        let source = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(DType::U8, shape![2, 3]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();
        source.write_value(&[1, 2], 255).await.unwrap();
        let (base_root, base_dir) = new_dir("copy_base").await;
        let copied = Tensor::copy_from(base_dir, &source, 3).await.unwrap();
        assert_eq!(copied.schema(), source.schema());
        assert_eq!(
            copied.layout(),
            match layout {
                Layout::Dense => Layout::Dense,
                Layout::Sparse { .. } => Layout::Sparse { axis: None },
            }
        );
        let view = source.view().transpose(Some(axes![1, 0])).unwrap();
        let (view_root, view_dir) = new_dir("copy_view").await;
        let transposed = Tensor::copy_from(view_dir, &view, 2).await.unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        for coord in common::iter_coords(&[3, 2]) {
            assert_eq!(
                transposed.read_value(&coord).await.unwrap(),
                view.read_value(&coord).await.unwrap()
            );
        }
        source.write_value(&[1, 2], 7).await.unwrap();
        assert_eq!(copied.read_value(&[1, 2]).await.unwrap(), 255);
        assert_eq!(transposed.read_value(&[2, 1]).await.unwrap(), 255);
        copied.write_value(&[1, 2], 3).await.unwrap();
        assert_eq!(source.read_value(&[1, 2]).await.unwrap(), 7);
        for root in [&root, &base_root, &view_root] {
            common::cleanup(root).await;
        }
    }
}

#[tokio::test]
async fn copy_propagates_source_and_destination_errors() {
    let (root, dir) = new_dir("copy_errors").await;
    let source = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(DType::F32, shape![2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let (out_root, out_dir) = new_dir("copy_errors_out").await;
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &source, 0).await,
        Err(Error::InvalidSchema(_))
    ));
    let scalar = source.view().slice(range![AxisRange::At(0)]).unwrap();
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &scalar, 2).await,
        Err(Error::InvalidSchema(_))
    ));
    assert!(!out_root.join("blocks").exists());
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete("0").await;
    assert!(Tensor::copy_from(out_dir, &source, 2).await.is_err());
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}
