mod common;

use common::{FsEntry, new_dir};
use fensor::{DType, Tensor, TensorArray, TensorRead, TensorSchema, TensorWrite};
use ha_ndarray::shape;
use std::io;

use crate::common::create_dense_tensor;

#[tokio::test]
async fn filesystem_tensor_f64_write_read_roundtrip() -> io::Result<()> {
    let (root, dir) = new_dir("f64_roundtrip").await;

    let schema = TensorSchema::new(DType::F64, shape![2, 2]).expect("valid schema");

    let tensor = create_dense_tensor::<f64>(dir.clone(), schema.clone()).await;

    tensor
        .write_value(&[0, 0], std::f64::consts::PI)
        .await
        .expect("write pi");
    tensor
        .write_value(&[1, 1], std::f64::consts::E)
        .await
        .expect("write e");

    dir.sync().await?;

    let pi = tensor.read_value(&[0, 0]).await.expect("read pi");
    let e = tensor.read_value(&[1, 1]).await.expect("read e");

    assert_eq!(pi, std::f64::consts::PI);
    assert_eq!(e, std::f64::consts::E);

    let loaded = Tensor::<FsEntry, f64>::load(dir.clone())
        .await
        .expect("load tensor");
    assert_eq!(schema, *loaded.schema());
    let loaded_e = loaded.read_value(&[1, 1]).await.expect("read loaded e");
    assert_eq!(loaded_e, std::f64::consts::E);

    dir.sync().await?;
    let _ = tokio::fs::remove_dir_all(&root).await;

    Ok(())
}
