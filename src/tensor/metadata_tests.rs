use super::*;

#[test]
fn typed_tensor_rejects_schema_dtype_mismatch() {
    let err = validate_tensor_dtype::<f32>(NumberType::Float(FloatType::F64))
        .expect_err("expected mismatch");
    assert!(matches!(err, Error::InvalidSchema(_)));
}

#[tokio::test]
async fn metadata_publication_does_not_replace_existing_contents() {
    use crate::test_support::{FsEntry, new_dir};

    let (_root, dir) = new_dir("metadata_once").await;
    let mut tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), vec![3].into()).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let file = tensor
        .storage
        .blocks()
        .read()
        .await
        .get_file(METADATA)
        .cloned()
        .unwrap();
    let original = file.read::<TensorMetadata<f32>>().await.unwrap().clone();
    tensor.schema = TensorSchema::new(NumberType::Float(FloatType::F32), vec![4].into()).unwrap();
    assert!(matches!(
        tensor.persist_metadata().await,
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(*file.read::<TensorMetadata<f32>>().await.unwrap(), original);
}
