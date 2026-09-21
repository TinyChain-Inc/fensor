//! Dtype-changing expressions retain root support and evaluate at consumption.

use fensor::unary::Cast;
use fensor::{
    Layout, Tensor, TensorArray, TensorCast, TensorRead, TensorSchema, TensorTransform, TensorTrig,
    TensorUnary, TensorView, TensorWrite, UnaryView,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, ArrayAccess, AxisRange, Buffer, NDArrayCast, NDArrayRead, NDArrayTrig, NDArrayUnary,
    axes, range, shape,
};
use number_general::{FloatType, NumberType};

use common::{FsEntry, new_dir};

mod common;

fn assert_value(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual, expected);
    }
}

#[tokio::test]
async fn f32_to_f64_cast_agrees_across_consumers_and_reload() {
    let (root, dir) = new_dir("cast_dense").await;
    let input = vec![
        0.0f32,
        -0.0,
        0.1,
        -0.1,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![input.len()]).unwrap(),
        Layout::Dense,
        3,
    )
    .await
    .unwrap();
    let cast: UnaryView<TensorView<'_, FsEntry, f32>, Cast<f64>> =
        tensor.view().cast().await.unwrap();
    // Constructing the cast must not capture source values.
    for (i, &value) in input.iter().enumerate() {
        tensor.write_value(&[i as u64], value).await.unwrap();
    }
    let reference: ArrayAccess<'_, f64> = ArrayAccess::from(
        Array::new(Buffer::from(input.clone()), shape![input.len()])
            .unwrap()
            .cast()
            .unwrap(),
    );
    let expected = reference.buffer().unwrap().to_slice().unwrap().into_vec();
    let blocks: Vec<Vec<f64>> = cast.read_blocks().unwrap().try_collect().await.unwrap();
    let values: Vec<_> = blocks.into_iter().flatten().collect();
    let (out_root, out_dir) = new_dir("cast_dense_out").await;
    let output: Tensor<FsEntry, f64> = Tensor::copy_from(out_dir.clone(), &cast, 4).await.unwrap();
    assert_eq!(output.schema().dtype(), NumberType::Float(FloatType::F64));
    out_dir.sync().await.unwrap();
    drop(output);
    drop(out_dir);
    let reloaded = Tensor::<FsEntry, f64>::load(common::open_dir(&out_root).unwrap())
        .await
        .unwrap();
    for (i, expected) in expected.into_iter().enumerate() {
        let coord = [i as u64];
        assert_value(expected, f64::from(input[i]));
        assert_value(cast.read_value(&coord).await.unwrap(), expected);
        assert_value(values[i], expected);
        assert_value(reloaded.read_value(&coord).await.unwrap(), expected);
    }
    assert!(cast.read_value(&[1]).await.unwrap().is_sign_negative());
    assert!(values[1].is_sign_negative());
    assert!(reloaded.read_value(&[1]).await.unwrap().is_sign_negative());
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}

#[tokio::test]
async fn mixed_dtype_chain_preserves_precision_transforms_and_batching() {
    let (root, dir) = new_dir("cast_chain").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![1, 4100]).unwrap(),
        Layout::Dense,
        128,
    )
    .await
    .unwrap();
    for i in 0..4100 {
        tensor
            .write_value(&[0, i], i as f32 / 4100.0)
            .await
            .unwrap();
    }
    let expression = tensor
        .view()
        .transpose(None)
        .unwrap()
        .sin()
        .await
        .unwrap()
        .flip(0)
        .unwrap()
        .cast()
        .await
        .unwrap()
        .exp()
        .await
        .unwrap()
        .squeeze(axes![1])
        .unwrap();
    let mut dropped = expression.read_blocks().unwrap();
    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let (first, second) = futures::try_join!(
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4096, 4]
    );
    let input: Vec<f32> = (0..4100).rev().map(|i| i as f32 / 4100.0).collect();
    let reference: ArrayAccess<'_, f64> = ArrayAccess::from(
        Array::new(Buffer::from(input), shape![4100])
            .unwrap()
            .sin()
            .unwrap()
            .cast()
            .unwrap(),
    );
    let expected = reference
        .exp()
        .unwrap()
        .buffer()
        .unwrap()
        .to_slice()
        .unwrap()
        .into_vec();
    assert_eq!(first.into_iter().flatten().collect::<Vec<_>>(), expected);
    for i in [0, 4095, 4099] {
        assert_eq!(
            expression.read_value(&[i]).await.unwrap(),
            expected[i as usize]
        );
    }
    common::cleanup(&root).await;
}

#[tokio::test]
async fn sparse_cast_preserves_original_support_and_filters_only_final_zeros() {
    let (root, dir) = new_dir("cast_sparse").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        3,
    )
    .await
    .unwrap();
    tensor.write_value(&[1, 2], 0.2).await.unwrap();
    let expression = tensor
        .view()
        .round()
        .await
        .unwrap()
        .cast()
        .await
        .unwrap()
        .transpose(None)
        .unwrap()
        .cos()
        .await
        .unwrap();
    assert_eq!(expression.read_value(&[2, 1]).await.unwrap(), 1.0f64);
    assert_eq!(expression.read_value(&[0, 0]).await.unwrap(), 0.0f64);
    let selection = range![AxisRange::In(0, 3, 1), AxisRange::In(0, 2, 1)];
    let rows: Vec<_> = expression
        .read_sparse_elements_in_order(selection.clone(), axes![0, 1])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows, vec![(vec![2, 1], 1.0f64)]);
    let (out_root, out_dir) = new_dir("cast_sparse_out").await;
    let output = Tensor::copy_from(out_dir, &expression, 2).await.unwrap();
    let blocks: Vec<Vec<f64>> = expression
        .read_blocks()
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let values: Vec<_> = blocks.into_iter().flatten().collect();
    for (i, coord) in common::iter_coords(&[3, 2]).enumerate() {
        let expected = if coord == vec![2, 1] { 1.0 } else { 0.0 };
        assert_eq!(values[i], expected);
        assert_eq!(output.read_value(&coord).await.unwrap(), expected);
    }
    let final_zero = expression.ln().await.unwrap();
    let rows: Vec<_> = final_zero
        .read_sparse_elements_in_order(selection, axes![0, 1])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert!(rows.is_empty());
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}
