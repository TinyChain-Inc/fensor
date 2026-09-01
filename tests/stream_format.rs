//! Exercises `Tensor`'s actual destream wire format via a real `tbon` encoder/decoder.

mod common;

use fensor::{DType, Layout, Tensor, TensorArray, TensorSchema, TensorViewDecoder};
use ha_ndarray::shape;

use common::{FsEntry, cleanup, open_dir, unique_tmp_dir};

fn dense_schema_f32(shape: ha_ndarray::Shape) -> TensorSchema {
    TensorSchema::new(DType::F32, shape).expect("schema")
}

#[tokio::test]
async fn base_tensor_round_trips_through_tbon() {
    let root = unique_tmp_dir("stream_base_roundtrip");
    tokio::fs::create_dir(&root).await.expect("mkdir");
    let schema = dense_schema_f32(shape![2, 3]);

    let dir = open_dir(&root).expect("open");
    let tensor =
        Tensor::<FsEntry, f32>::create(dir.clone(), dir, schema.clone(), Layout::Dense, 1000)
            .await
            .expect("create");

    let view = tensor.view();
    let encoded = tbon::en::encode(view.view_encoder()).expect("encode base tensor");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: TensorViewDecoder<FsEntry, f32> =
        tbon::de::try_decode((dir2.clone(), dir2), encoded)
            .await
            .expect("decode base tensor");
    let decoded = decoded.into_inner();

    assert_eq!(decoded.schema(), &schema);

    cleanup(&root).await;
}
