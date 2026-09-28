//! Number classes follow the typed filesystem adapter.

use fensor::{Error, Layout, Tensor, TensorCast, TensorGeometry, TensorNumeric, TensorSchema};
use ha_ndarray::shape;
use number_general::{ComplexType, DType, FloatType, IntType, NumberType, UIntType};

use common::{FsEntry, new_dir};

mod common;

#[test]
fn unsupported_number_classes_are_rejected_at_schema_construction() {
    for dtype in [
        NumberType::Number,
        NumberType::Bool,
        NumberType::Complex(ComplexType::Complex),
        NumberType::Int(IntType::Int),
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
    let cast = TensorCast::<f64>::cast(&tensor.view()).await.unwrap();
    assert_eq!(cast.dtype(), f64::dtype());
    assert_eq!(cast.is_nan().await.unwrap().dtype(), u8::dtype());
    common::cleanup(&root).await;
}
