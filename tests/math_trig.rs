//! Absolute-value and trigonometric views use the same filesystem consumers.

use fensor::unary::UnaryOp;
use fensor::{
    Layout, Tensor, TensorAbs, TensorElement, TensorFileEntry, TensorGeometry, TensorRead,
    TensorSchema, TensorTransform, TensorTrig, TensorUnary, TensorView, TensorWrite, UnaryView,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, ArrayAccess, AxisRange, Buffer, NDArrayAbs, NDArrayRead, NDArrayTrig, axes, range, shape,
};
use number_general::FloatType;

use common::{FsEntry, new_dir};

mod common;

fn assert_float<T: Copy + Into<f64>>(actual: T, expected: T) {
    let (actual, expected) = (actual.into(), expected.into());
    if expected.is_nan() {
        assert!(actual.is_nan(), "expected NaN, got {actual}");
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else {
        let tolerance = if std::mem::size_of::<T>() == 4 {
            1e-6
        } else {
            1e-12
        };
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }
}

async fn check_consumers<T, O>(view: UnaryView<TensorView<'_, FsEntry, T>, O>, expected: &[T])
where
    T: TensorElement + Into<f64>,
    FsEntry: TensorFileEntry<T>,
    O: UnaryOp<T, Output = T>,
{
    let blocks: Vec<Vec<T>> = view.read_blocks().unwrap().try_collect().await.unwrap();
    let values: Vec<_> = blocks.into_iter().flatten().collect();
    assert_eq!(values.len(), expected.len());
    let (out_root, out_dir) = new_dir("trig_materialized").await;
    let output = Tensor::copy_from(out_dir, &view, 3).await.unwrap();
    for (i, &expected) in expected.iter().enumerate() {
        let coord = [i as u64];
        let point = view.read_value(&coord).await.unwrap();
        assert_float(point, expected);
        assert_float(values[i], expected);
        assert_float(output.read_value(&coord).await.unwrap(), expected);
        // Sparse storage omits zeros; dense consumers preserve signed zero.
        if matches!(view.layout(), Layout::Dense) && expected == T::default() {
            assert_eq!(
                Into::<f64>::into(point).is_sign_negative(),
                Into::<f64>::into(expected).is_sign_negative()
            );
            assert_eq!(
                Into::<f64>::into(values[i]).is_sign_negative(),
                Into::<f64>::into(expected).is_sign_negative()
            );
            assert_eq!(
                Into::<f64>::into(output.read_value(&coord).await.unwrap()).is_sign_negative(),
                Into::<f64>::into(expected).is_sign_negative()
            );
        }
    }
    if matches!(view.layout(), Layout::Sparse { .. }) {
        let rows: Vec<_> = view
            .read_sparse_elements_in_order(range![AxisRange::In(0, expected.len(), 1)], axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let expected_rows: Vec<_> = expected
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != T::default())
            .collect();
        assert_eq!(rows.len(), expected_rows.len());
        for ((coord, actual), (i, expected)) in rows.into_iter().zip(expected_rows) {
            assert_eq!(coord, vec![i as u64]);
            assert_float(actual, *expected);
        }
    }
    common::cleanup(&out_root).await;
}

macro_rules! operation_matrix {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
                let (root, dir) = new_dir(stringify!($name)).await;
                let input: Vec<$t> = vec![
                    -2.0,
                    -1.0,
                    -0.5,
                    -0.0,
                    0.0,
                    0.2,
                    0.5,
                    1.0,
                    2.0,
                    1000.0,
                    -1000.0,
                    <$t>::INFINITY,
                    <$t>::NEG_INFINITY,
                    <$t>::NAN,
                ];
                let tensor = Tensor::<FsEntry, $t>::create(
                    dir,
                    TensorSchema::new(<$t as number_general::DType>::dtype(), shape![input.len()])
                        .unwrap(),
                    layout,
                    3,
                )
                .await
                .unwrap();
                for (i, &value) in input.iter().enumerate() {
                    tensor.write_value(&[i as u64], value).await.unwrap();
                }
                macro_rules! check {
                    ($method:ident) => {{
                        let mut expected = Vec::new();
                        for &value in &input {
                            if matches!(layout, Layout::Sparse { .. }) && value == 0.0 {
                                expected.push(0.0);
                            } else {
                                let array = ArrayAccess::from(
                                    Array::new(Buffer::from(vec![value]), shape![1]).unwrap(),
                                );
                                let result = array
                                    .$method()
                                    .unwrap()
                                    .buffer()
                                    .unwrap()
                                    .to_slice()
                                    .unwrap()
                                    .into_vec();
                                expected.push(result[0]);
                            }
                        }
                        check_consumers(tensor.view().$method().await.unwrap(), &expected).await;
                    }};
                }
                check!(abs);
                check!(sin);
                check!(asin);
                check!(sinh);
                check!(cos);
                check!(acos);
                check!(cosh);
                check!(tan);
                check!(atan);
                check!(tanh);
                common::cleanup(&root).await;
            }
        }
    };
}

operation_matrix!(f32_abs_and_trig_consumers, f32);
operation_matrix!(f64_abs_and_trig_consumers, f64);

#[tokio::test]
async fn mixed_sparse_chain_preserves_support_transforms_and_reuse() {
    let (root, dir) = new_dir("trig_sparse_chain").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(fensor::NumberType::Float(FloatType::F32), shape![1, 4102]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        4096,
    )
    .await
    .unwrap();
    for i in 0..4101 {
        tensor.write_value(&[0, i], -0.2).await.unwrap();
    }
    let expression = tensor
        .view()
        .transpose(None)
        .unwrap()
        .abs()
        .await
        .unwrap()
        .round()
        .await
        .unwrap()
        .flip(0)
        .unwrap()
        .cos()
        .await
        .unwrap()
        .squeeze(axes![1])
        .unwrap();
    let mut dropped = expression.read_blocks().unwrap();
    let first = dropped.try_next().await.unwrap().unwrap();
    assert_eq!(first.len(), 4096);
    assert_eq!(first[0], 0.0);
    assert!(first[1..].iter().all(|&v| v == 1.0));
    drop(dropped);
    let (first, second) = futures::try_join!(
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4096, 6]
    );
    let rows: Vec<_> = expression
        .read_sparse_elements_in_order(range![AxisRange::In(0, expression.size(), 1)], axes![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows.len(), 4101);
    assert!(
        rows.iter()
            .all(|(coord, value)| coord[0] > 0 && *value == 1.0)
    );
    let final_zero = expression.ln().await.unwrap().sin().await.unwrap();
    let rows: Vec<_> = final_zero
        .read_sparse_elements_in_order(range![AxisRange::In(0, 4102, 1)], axes![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert!(rows.is_empty());
    // Check terminal consumers on a small selection; the full expression above
    // exercises batch boundaries without allocating thousands of sparse files.
    let expression = expression.slice(range![AxisRange::In(0, 8, 1)]).unwrap();
    let (out_root, out_dir) = new_dir("trig_chain_out").await;
    let output = Tensor::copy_from(out_dir, &expression, 16).await.unwrap();
    for i in 0..8 {
        let expected = if i == 0 { 0.0 } else { 1.0 };
        assert_eq!(expression.read_value(&[i]).await.unwrap(), expected);
        assert_eq!(output.read_value(&[i]).await.unwrap(), expected);
    }
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}
