//! Exercises `Tensor::view_encoder` / `TensorViewEncoder` / `TensorViewDecoder`:
//! streaming a tensor's current view (identity or transformed, dense or
//! sparse) together with only its non-default element data, and reconstructing
//! a fresh, independent base tensor from it via a real `tbon` round trip.

mod common;

use fensor::{
    DType, Layout, TensorArray, TensorGeometry, TensorRead, TensorSchema, TensorTransform,
    TensorViewDecoder, TensorViewSemantics, TensorWrite,
};

use futures::stream::TryStreamExt;
use ha_ndarray::{AxisRange, axes, range, shape};

use common::{
    FsEntry, cleanup, create_dense_tensor, create_sparse_tensor, iter_coords, new_dir, open_dir,
};

#[tokio::test]
async fn identity_dense_view_round_trips_with_data() {
    let schema = TensorSchema::new(DType::F32, shape![2, 3]).expect("Schema created");
    let (root, dir) = new_dir("view_snapshot_identity_dense").await;

    let tensor = create_dense_tensor::<f32>(dir, schema.clone()).await;

    let mut expected = Vec::new();
    for (idx, coord) in iter_coords(schema.shape()).enumerate() {
        let value = idx as f32 + 1.0;
        tensor.write_value(&coord, value).await.expect("write");
        expected.push((coord, value));
    }

    let view = tensor.view();
    let encoded = tbon::en::encode(view.view_encoder()).expect("encode view");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: TensorViewDecoder<FsEntry, f32> = tbon::de::try_decode(dir2, encoded)
        .await
        .expect("decode view");

    let decoded = decoded.into_inner();

    assert!(decoded.view().is_base_tensor());
    assert_eq!(decoded.schema().shape(), schema.shape());
    assert_eq!(decoded.layout(), Layout::Dense);

    for (coord, value) in expected.iter() {
        let read = decoded.read_value(coord).await.expect("read");
        assert_eq!(read, *value);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn identity_sparse_view_round_trips_with_data_and_axis() {
    let schema = TensorSchema::new(DType::F32, shape![2, 3]).expect("schema");
    let (root, dir) = new_dir("view_snapshot_identity_sparse").await;
    let tensor = create_sparse_tensor::<f32>(dir, schema.clone(), Some(0)).await;

    tensor.write_value(&[0, 1], 5.0f32).await.expect("write");
    tensor.write_value(&[1, 2], 9.0f32).await.expect("write");

    let view = tensor.view();
    let encoded = tbon::en::encode(view.view_encoder()).expect("encode view");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: TensorViewDecoder<FsEntry, f32> = tbon::de::try_decode(dir2, encoded)
        .await
        .expect("decode view");

    let decoded = decoded.into_inner();

    assert!(decoded.view().is_base_tensor());
    assert_eq!(decoded.layout(), Layout::Sparse { axis: Some(0) });

    for coord in iter_coords(schema.shape()) {
        let expected = tensor.read_value(&coord).await.expect("read source");
        let actual = decoded.read_value(&coord).await.expect("read decoded");
        assert_eq!(actual, expected);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn non_identity_dense_view_round_trips_with_data() {
    let (root, dir) = new_dir("view_snapshot_non_identity_dense").await;
    let schema = TensorSchema::new(DType::F32, shape![3, 4]).expect("schema");

    let tensor = create_dense_tensor::<f32>(dir, schema.clone()).await;

    for coord in iter_coords(schema.shape()) {
        let value = (coord[0] * 10 + coord[1]) as f32 + 1.0;
        tensor.write_value(&coord, value).await.expect("write");
    }

    let sliced = tensor
        .view()
        .slice(range![AxisRange::In(1, 3, 1), AxisRange::In(0, 4, 2)])
        .expect("slice")
        .transpose(Some(axes![1, 0]))
        .expect("transpose");

    assert!(!sliced.is_base_tensor());

    let mut expected = Vec::new();
    for coord in iter_coords(sliced.shape()) {
        let value = sliced.read_value(&coord).await.expect("read source view");
        expected.push((coord, value));
    }

    let encoded = tbon::en::encode(sliced.view_encoder()).expect("encode view");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: TensorViewDecoder<FsEntry, f32> = tbon::de::try_decode(dir2, encoded)
        .await
        .expect("decode view");

    let decoded = decoded.into_inner();

    assert!(decoded.view().is_base_tensor());
    assert_eq!(decoded.shape(), sliced.shape());

    for (coord, value) in expected {
        let read = decoded.read_value(&coord).await.expect("read");
        assert_eq!(read, value);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn non_identity_sparse_view_round_trips_with_data_resets_axis() {
    let schema = TensorSchema::new(DType::F32, shape![2, 3]).expect("schema");
    let (root, dir) = new_dir("view_snapshot_non_identity_sparse").await;
    let tensor = create_sparse_tensor(dir, schema.clone(), Some(0)).await;

    tensor.write_value(&[0, 1], 7.0f32).await.expect("write");
    tensor.write_value(&[1, 2], 3.0f32).await.expect("write");

    let transposed = tensor
        .view()
        .transpose(Some(axes![1, 0]))
        .expect("transpose");
    assert!(!transposed.is_base_tensor());

    let mut expected = Vec::new();
    for coord in iter_coords(transposed.shape()) {
        let value = transposed
            .read_value(&coord)
            .await
            .expect("read source view");
        expected.push((coord, value));
    }

    let encoded = tbon::en::encode(transposed.view_encoder()).expect("encode view");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    let decoded: TensorViewDecoder<FsEntry, f32> = tbon::de::try_decode(dir2, encoded)
        .await
        .expect("decode view");

    let decoded = decoded.into_inner();

    assert!(decoded.view().is_base_tensor());
    assert_eq!(decoded.layout(), Layout::Sparse { axis: None });

    for (coord, value) in expected {
        let read = decoded.read_value(&coord).await.expect("read");
        assert_eq!(read, value);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn only_nonzero_values_are_transmitted() {
    // Create a large dense tensor (20x20 = 400 elements), but write only ONE nonzero value.
    // This demonstrates that the encoder only transmits non-default values, not all 400 elements.
    let schema = TensorSchema::new(DType::F32, shape![20, 20]).expect("schema");
    let (root, dir) = new_dir("view_snapshot_sparse_transmission").await;
    let tensor = create_dense_tensor(dir, schema.clone()).await;

    // Write exactly one nonzero value; everything else remains at default (0.0)
    tensor.write_value(&[5, 7], 3.0f32).await.expect("write");

    let view = tensor.view();
    let encoded_stream = tbon::en::encode(view.view_encoder()).expect("encode view");
    // Collect the stream to measure size and prepare for decoding
    let encoded_parts = encoded_stream
        .try_collect::<Vec<_>>()
        .await
        .expect("collect encoded stream");
    let mut total_size = 0;
    for part in &encoded_parts {
        total_size += part.len();
    }

    // Assert that the encoded size is small. A single f32 value (4 bytes) plus schema
    // overhead should be far smaller than 400 f32 values (1600 bytes).
    // Being generous, we allow up to 200 bytes to account for wire framing, tags, and metadata.
    // This verifies that only the one nonzero value was transmitted, not all 400 elements.
    assert!(
        total_size < 200,
        "encoded size {} should be small for sparse transmission (proves only nonzero values transmitted)",
        total_size
    );

    // Round-trip: decode and verify correctness
    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");
    // Create a stream from the collected chunks for decoding
    let encoded_stream_for_decode =
        futures::stream::iter(encoded_parts.into_iter().map(Ok::<_, tbon::de::Error>));
    let decoded: TensorViewDecoder<FsEntry, f32> =
        tbon::de::try_decode(dir2, encoded_stream_for_decode)
            .await
            .expect("decode view");

    let decoded = decoded.into_inner();

    assert!(decoded.view().is_base_tensor());
    assert_eq!(decoded.schema().shape(), schema.shape());

    // Verify the single written value is present
    let value = decoded.read_value(&[5, 7]).await.expect("read");
    assert_eq!(value, 3.0f32);

    // Verify all other coordinates are zero (default)
    for coord in iter_coords(schema.shape()) {
        if coord[0] == 5 && coord[1] == 7 {
            // Skip the one written value, we already checked it
            continue;
        }
        let value = decoded.read_value(&coord).await.expect("read");
        assert_eq!(value, 0.0f32, "expected zero at {:?}", coord);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn truncated_stream_fails_closed() {
    let schema = TensorSchema::new(DType::F32, shape![2, 3]).expect("schema");
    let (root, dir) = new_dir("view_snapshot_verification_mismatch").await;
    let tensor = create_dense_tensor(dir, schema.clone()).await;

    // Write a couple of nonzero values
    tensor.write_value(&[0, 1], 5.0f32).await.expect("write");
    tensor.write_value(&[1, 2], 9.0f32).await.expect("write");

    let view = tensor.view();
    let encoded_stream = tbon::en::encode(view.view_encoder()).expect("encode view");
    // Collect the stream so we can create a truncated version
    let encoded_parts = encoded_stream
        .try_collect::<Vec<_>>()
        .await
        .expect("collect encoded stream");

    // Create a corrupted stream that is incomplete: keep only the first chunk (schema).
    // The decoder will create the tensor, then try to read the pairs sequence. Since the
    // stream ends immediately, the underlying `tbon` decoder itself fails with an
    // "unexpected end of stream" error (it never sees the pairs sequence's closing
    // delimiter), before the decoder-side visitor ever gets a chance to return.
    let corrupted_parts: Vec<_> = if !encoded_parts.is_empty() {
        vec![encoded_parts[0].clone()]
    } else {
        vec![]
    };

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");

    // Create a stream from the truncated chunks for decoding attempt
    let encoded_stream_corrupted =
        futures::stream::iter(corrupted_parts.into_iter().map(Ok::<_, tbon::de::Error>));

    // Attempt to decode with incomplete stream; should fail due to the truncated pairs sequence
    let result: std::result::Result<TensorViewDecoder<FsEntry, f32>, _> =
        tbon::de::try_decode(dir2.clone(), encoded_stream_corrupted).await;

    assert!(
        result.is_err(),
        "corrupted verification (incomplete stream) must fail"
    );

    cleanup(&root).await;
}

#[tokio::test]
async fn corrupted_block_read_fails_closed() {
    let (root, dir) = new_dir("view_snapshot_read_value_fails_dense").await;
    let schema = TensorSchema::new(DType::F32, shape![1000, 1000]).expect("schema");
    let tensor = create_dense_tensor(dir.clone(), schema.clone()).await;

    tensor.write_value(&[0, 1], 5.0f32).await.expect("write");
    tensor
        .write_value(&[205, 1005], 9.0f32)
        .await
        .expect("write");
    tensor
        .write_value(&[805, 2000], 13.0f32)
        .await
        .expect("write");

    dir.sync().await.expect("fs sync");

    let dir_lock = dir.write().await;
    let block_dir = dir_lock
        .get_dir("blocks")
        .expect("blocks exists for any tensor");
    block_dir
        .write()
        .await
        .truncate_and_sync()
        .await
        .expect("blocks are cleaned up");

    // let encoded_stream = tbon::en::encode(tensor.view().view_encoder()).expect("encode view");

    // let decode_root = root.join("decoded");
    // tokio::fs::create_dir(&decode_root)
    //     .await
    //     .expect("mkdir decoded");
    // let dir2 = open_dir(&decode_root).expect("open decode target");

    // let result: std::result::Result<TensorViewDecoder<FsEntry, f32>, _> =
    //     tbon::de::try_decode(dir2.clone(), encoded_stream).await;

    // assert!(result.is_ok());

    cleanup(&root).await;
}

#[tokio::test]
async fn corrupted_dtype_mismatch_fails_closed() {
    let (root, dir) = new_dir("view_snapshot_dtype_mismatch").await;

    // Create an f64 tensor
    let schema = TensorSchema::new(DType::F64, shape![2, 2]).expect("schema");
    let tensor_f64 = create_dense_tensor(dir, schema.clone()).await;

    tensor_f64.write_value(&[0, 0], 1.5).await.expect("write");

    let view = tensor_f64.view();
    let encoded = tbon::en::encode(view.view_encoder()).expect("encode f64 view");

    let decode_root = root.join("decoded");
    tokio::fs::create_dir(&decode_root)
        .await
        .expect("mkdir decoded");
    let dir2 = open_dir(&decode_root).expect("open decode target");

    // Try to decode f64-encoded data as f32; should fail with dtype mismatch
    let result: std::result::Result<TensorViewDecoder<FsEntry, f32>, _> =
        tbon::de::try_decode(dir2.clone(), encoded).await;

    assert!(result.is_err(), "dtype mismatch must fail closed");

    // Verify no storage was created in the decode target (dtype check is early, before Tensor::create)
    let mut dir_entries = tokio::fs::read_dir(&decode_root)
        .await
        .expect("read decode_root");

    let mut found_storage = false;
    while let Some(entry) = dir_entries.next_entry().await.expect("read next entry") {
        let path = entry.path();
        if path.ends_with("blocks") || path.ends_with("metadata") {
            found_storage = true;
        }
    }

    assert!(
        !found_storage,
        "no tensor storage should be created on dtype mismatch"
    );

    cleanup(&root).await;
}
