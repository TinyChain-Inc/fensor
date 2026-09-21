//! Number classes follow the typed filesystem adapter.

use fensor::{
    Error, Layout, Tensor, TensorArray, TensorCast, TensorElement, TensorFileEntry, TensorGeometry,
    TensorNumeric, TensorRead, TensorSchema, TensorWrite,
};
use ha_ndarray::shape;
use number_general::{ComplexType, DType, FloatType, IntType, NumberType, UIntType};

use common::{FsEntry, new_dir};
mod common;

async fn check_storage<T: TensorElement>()
where
    FsEntry: TensorFileEntry<T>,
{
    let (root, dir) = new_dir("number_type_storage").await;
    let tensor = Tensor::<FsEntry, T>::create(
        dir.clone(),
        TensorSchema::new(T::dtype(), shape![2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    tensor.write_value(&[1], T::ONE).await.unwrap();
    assert_eq!(tensor.dtype(), T::dtype());
    assert_eq!(tensor.view().dtype(), T::dtype());
    assert_eq!(tensor.schema().dtype(), T::dtype());
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let metadata = blocks.read().await.get_file("metadata").unwrap().clone();
    assert_eq!(
        metadata
            .read::<fensor::TensorMetadata<T>>()
            .await
            .unwrap()
            .shape()
            .as_slice(),
        &[2]
    );
    dir.sync().await.unwrap();
    drop(tensor);
    drop(metadata);
    drop(blocks);
    drop(dir);
    let reloaded = Tensor::<FsEntry, T>::load(common::open_dir(&root).unwrap())
        .await
        .unwrap();
    assert_eq!(reloaded.schema().dtype(), T::dtype());
    assert_eq!(reloaded.read_value(&[1]).await.unwrap(), T::ONE);
    common::cleanup(&root).await;
}

#[tokio::test]
async fn number_classes_follow_typed_storage() {
    check_storage::<u8>().await;
    check_storage::<f32>().await;
    check_storage::<f64>().await;
}

#[test]
fn unsupported_number_classes_are_rejected_at_schema_construction() {
    for dtype in [
        NumberType::Number,
        NumberType::Bool,
        NumberType::Complex(ComplexType::C32),
        NumberType::Int(IntType::I32),
        NumberType::UInt(UIntType::U16),
        NumberType::Float(FloatType::Float),
        NumberType::UInt(UIntType::UInt),
    ] {
        assert!(matches!(
            TensorSchema::new(dtype, shape![2]),
            Err(Error::InvalidSchema(_))
        ));
    }
}

#[tokio::test]
async fn computed_views_report_the_output_number_class() {
    let (root, dir) = new_dir("number_type_expression").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let cast = tensor.view().cast().await.unwrap();
    assert_eq!(cast.dtype(), f64::dtype());
    assert_eq!(cast.is_nan().await.unwrap().dtype(), u8::dtype());
    common::cleanup(&root).await;
}
