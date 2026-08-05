//! Exercises `Tensor`'s actual destream wire format via a real `tbon` encoder/decoder.

mod common;

use fensor::{
    DType, Layout, Tensor, TensorArray, TensorSchema, TensorTransform, contiguous_strides,
};
use ha_ndarray::{axes, shape};

use common::{FsEntry, cleanup, open_dir, unique_tmp_dir};

fn dense_schema_f32(shape: ha_ndarray::Shape, block_shape: ha_ndarray::Shape) -> TensorSchema {
    let strides = contiguous_strides(&shape);
    TensorSchema::new(DType::F32, shape, Layout::Dense, block_shape, strides).expect("schema")
}

#[tokio::test]
async fn base_tensor_round_trips_through_tbon() {
    let root = unique_tmp_dir("stream_base_roundtrip");
    tokio::fs::create_dir(&root).await.expect("mkdir");
    let schema = dense_schema_f32(shape![2, 3], shape![1, 3]);

    let dir = open_dir(&root).expect("open");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema.clone())
        .await
        .expect("create");

    let encoded = tbon::en::encode(tensor).expect("encode base tensor");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: Tensor<FsEntry, f32> = tbon::de::try_decode(dir2, encoded)
        .await
        .expect("decode base tensor");

    assert_eq!(decoded.schema(), &schema);

    cleanup(&root).await;
}

#[tokio::test]
async fn non_identity_view_is_rejected_on_encode() {
    let root = unique_tmp_dir("stream_non_identity_rejected");
    tokio::fs::create_dir(&root).await.expect("mkdir");
    let schema = dense_schema_f32(shape![2, 3], shape![1, 3]);

    let dir = open_dir(&root).expect("open");
    let tensor = Tensor::<FsEntry, f32>::create(dir, schema)
        .await
        .expect("create");

    let transposed = tensor.transpose(Some(axes![1, 0])).expect("transpose");

    let result = tbon::en::encode(transposed);
    assert!(
        result.is_err(),
        "encoding a non-identity view must be rejected, not silently drop the transform"
    );

    cleanup(&root).await;
}
