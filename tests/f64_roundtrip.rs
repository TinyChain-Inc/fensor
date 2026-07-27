mod common;

use common::{FsEntry, unique_tmp_dir};
use fensor::{
    DType, Layout, Tensor, TensorArray, TensorRead, TensorSchema, TensorWrite, contiguous_strides,
};
use freqfs::Cache;
use ha_ndarray::{Shape, shape};
use std::io;

#[tokio::test]
async fn filesystem_tensor_f64_write_read_roundtrip() -> io::Result<()> {
    let root = unique_tmp_dir("f64_roundtrip");
    tokio::fs::create_dir(&root).await?;

    let cache = Cache::<FsEntry>::new(1_000_000, None);
    let dir = cache.load(root.clone())?;

    let shape: Shape = shape![2, 2];
    let schema = TensorSchema::new(
        DType::F64,
        shape.clone(),
        Layout::Dense,
        shape![1, 2],
        contiguous_strides(&shape),
    )
    .expect("valid schema");

    let tensor = Tensor::<FsEntry, f64>::create(dir.clone(), schema.clone())
        .await
        .expect("create tensor");

    tensor
        .write_value(&[0, 0], std::f64::consts::PI)
        .await
        .expect("write pi");
    tensor
        .write_value(&[1, 1], std::f64::consts::E)
        .await
        .expect("write e");

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
