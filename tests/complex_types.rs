#![cfg(feature = "complex")]

use fensor::{
    Layout, Tensor, TensorAbs, TensorBoolean, TensorBooleanScalar, TensorCast, TensorCompare,
    TensorCompareScalar, TensorComplex, TensorElement, TensorExpression, TensorFileEntry,
    TensorMatMul, TensorMath, TensorMathScalar, TensorMatrixUnary, TensorNumeric, TensorRead,
    TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorTransform, TensorTrig, TensorUnary,
    TensorUnaryBoolean, TensorWhere, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, Buffer, Complex, MatrixDual, NDArrayAbs, NDArrayBoolean, NDArrayBooleanScalar,
    NDArrayCompare, NDArrayCompareScalar, NDArrayComplex, NDArrayMath, NDArrayMathScalar,
    NDArrayNumeric, NDArrayRead, NDArrayReduceAll, NDArrayTrig, NDArrayUnary, NDArrayUnaryBoolean,
    Number, axes, shape,
};

use common::{FsEntry, cleanup, fixture, numbers::same};

mod common;

async fn operations<T: TensorElement + Complex>(input: Vec<T>)
where
    T::Real: TensorElement,
    FsEntry: TensorFileEntry<T> + TensorFileEntry<T::Real>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, tensor) = fixture::source(
            "complex_operations",
            shape![2, 2],
            layout,
            3,
            1024,
            input.clone(),
        )
        .await;
        let v = tensor.view();
        let dense = TensorExpression::new(v.clone()).unwrap().into_dense();
        let a = Array::new(Buffer::from(input.clone()), shape![2, 2]).unwrap();
        macro_rules! unary {
            ($m:ident, $source:ident) => {
                fixture::blocks(
                    &$source.$m().await.unwrap(),
                    &a.clone()
                        .$m()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .to_vec(),
                    same,
                )
                .await;
            };
        }
        unary!(abs, v);
        unary!(conj, v);
        unary!(re, v);
        unary!(im, v);
        unary!(angle, v);
        unary!(exp, dense);
        unary!(ln, dense);
        unary!(sin, v);
        unary!(asin, v);
        unary!(sinh, v);
        unary!(cos, dense);
        unary!(acos, dense);
        unary!(cosh, dense);
        unary!(tan, v);
        unary!(atan, v);
        unary!(tanh, v);
        unary!(not, dense);
        unary!(is_nan, v);
        unary!(is_inf, v);
        macro_rules! binary {
            ($m:ident, $s:ident, $binary_source:ident, $scalar_source:ident) => {
                fixture::blocks(
                    &$binary_source.$m(&$binary_source).await.unwrap(),
                    &a.clone()
                        .$m(a.clone())
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .to_vec(),
                    same,
                )
                .await;
                fixture::blocks(
                    &$scalar_source.$s(T::ONE).await.unwrap(),
                    &a.clone()
                        .$s(T::ONE)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .to_vec(),
                    same,
                )
                .await;
            };
        }
        binary!(add, add_scalar, v, dense);
        binary!(sub, sub_scalar, v, dense);
        binary!(mul, mul_scalar, v, v);
        binary!(div, div_scalar, dense, v);
        binary!(pow, pow_scalar, dense, v);
        binary!(log, log_scalar, dense, dense);
        binary!(eq, eq_scalar, dense, v);
        binary!(ne, ne_scalar, v, dense);
        binary!(and, and_scalar, v, v);
        binary!(or, or_scalar, v, dense);
        binary!(xor, xor_scalar, v, dense);
        let values = &input;
        let values_array = Array::new(Buffer::from(input.clone()), shape![input.len()]).unwrap();
        assert!(same(
            tensor.sum_all().await.unwrap(),
            values_array.clone().sum_all().unwrap()
        ));
        assert!(same(
            tensor.product_all().await.unwrap(),
            values_array.product_all().unwrap()
        ));
        assert_eq!(
            tensor.all().await.unwrap(),
            values.iter().all(|v| *v != T::ZERO)
        );
        assert_eq!(
            tensor.any().await.unwrap(),
            values.iter().any(|v| *v != T::ZERO)
        );
        for keep in [false, true] {
            fixture::consumers(
                &v.sum(axes![0], keep).await.unwrap(),
                &[
                    Number::add(input[0], input[2]),
                    Number::add(input[1], input[3]),
                ],
                same,
                same,
            )
            .await;
            let expected: Vec<_> = [vec![0, 2], vec![1, 3]]
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|i| input[*i])
                        .reduce(Number::mul)
                        .unwrap_or(T::ZERO)
                })
                .collect();
            fixture::blocks(&v.product(axes![0], keep).await.unwrap(), &expected, same).await;
        }
        fixture::consumers(
            &v.mt().await.unwrap(),
            &[input[0], input[2], input[1], input[3]],
            same,
            same,
        )
        .await;
        fixture::consumers(&v.diag().await.unwrap(), &[input[0], input[3]], same, same).await;
        let expected = a
            .clone()
            .matmul(a)
            .unwrap()
            .buffer()
            .unwrap()
            .to_slice()
            .unwrap()
            .to_vec();
        fixture::consumers(&v.matmul(&v).await.unwrap(), &expected, same, same).await;
        fixture::blocks(
            &dense.eq(&dense).await.unwrap().cond(&v, &v).await.unwrap(),
            &input,
            same,
        )
        .await;
        let conjugate = v.conj().await.unwrap();
        let expected: Vec<_> = input.iter().map(|v| Complex::conj(*v)).collect();
        // Sparse materialization omits all-zero elements, canonicalizing their zero signs.
        fixture::consumers(&conjugate, &expected, same, |a, b| {
            same(a, b) || (matches!(layout, Layout::Sparse { .. }) && a == T::ZERO && b == T::ZERO)
        })
        .await;
        let entries = conjugate
            .read_coordinate_blocks()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for (coords, values) in entries {
            for (c, value) in coords.iter().zip(values) {
                assert!(same(value, expected[(c[0] * 2 + c[1]) as usize]));
            }
        }
        let cancelled = TensorExpression::new(v.sub(&v).await.unwrap())
            .unwrap()
            .into_dense()
            .exp()
            .await
            .unwrap();
        fixture::blocks(&cancelled, &[T::ONE; 4], same).await;

        if matches!(layout, Layout::Sparse { .. }) {
            for coord in common::iter_coords(&[2, 2]) {
                tensor.write_value(&coord, T::ZERO).await.unwrap();
            }
            assert_eq!(tensor.sum_all().await.unwrap(), T::ZERO);
            assert_eq!(tensor.product_all().await.unwrap(), T::ZERO);
            assert!(!tensor.all().await.unwrap());
            assert!(!tensor.any().await.unwrap());
            fixture::blocks(
                &TensorExpression::new(v.product(axes![1], false).await.unwrap())
                    .unwrap()
                    .into_dense()
                    .exp()
                    .await
                    .unwrap(),
                &[T::ONE; 2],
                same,
            )
            .await;
        }
        drop(tensor);
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn complex32_operations() {
    use fensor::complex::Complex32 as C;
    operations(vec![
        C::new(0., 0.),
        C::new(1., 2.),
        C::new(2., -1.),
        C::new(3., 1.),
    ])
    .await;
}

#[tokio::test]
async fn complex64_operations() {
    use fensor::complex::Complex64 as C;
    operations(vec![
        C::new(0., 0.),
        C::new(1., 2.),
        C::new(2., -1.),
        C::new(3., 1.),
    ])
    .await;
}

macro_rules! exceptional {
    ($test:ident,$c:ty,$r:ty) => {
        #[tokio::test]
        async fn $test() {
            type C = $c;
            let input = vec![
                C::new(-1., 0.),
                C::new(-1., -0.),
                C::new(0., -0.),
                C::new(-0., 0.),
                C::new(<$r>::NAN, 2.),
                C::new(2., <$r>::NAN),
                C::new(<$r>::INFINITY, 0.),
                C::new(0., <$r>::NEG_INFINITY),
                C::new(3., 4.),
            ];
            let (root, tensor) = fixture::source(
                "complex_special",
                shape![input.len() as u64],
                Layout::Dense,
                2,
                1024,
                input.clone(),
            )
            .await;
            fixture::consumers(&tensor.view(), &input, same, same).await;
            fixture::blocks(
                &tensor.view().is_nan().await.unwrap(),
                &[0, 0, 0, 0, 1, 1, 0, 0, 0],
                |a, b| a == b,
            )
            .await;
            fixture::blocks(
                &tensor.view().is_inf().await.unwrap(),
                &[0, 0, 0, 0, 0, 0, 1, 1, 0],
                |a, b| a == b,
            )
            .await;
            fixture::blocks(
                &tensor.view().not().await.unwrap(),
                &[0, 0, 1, 1, 0, 0, 0, 0, 0],
                |a, b| a == b,
            )
            .await;
            assert_eq!(
                tensor
                    .view()
                    .abs()
                    .await
                    .unwrap()
                    .read_value(&[8])
                    .await
                    .unwrap(),
                5.
            );
            let ln = tensor.view().ln().await.unwrap();
            let above = ln.read_value(&[0]).await.unwrap();
            let below = ln.read_value(&[1]).await.unwrap();
            assert_eq!(above.re, 0.);
            assert_eq!(below.re, 0.);
            assert!(above.im > 0. && below.im < 0.);
            for (i, &value) in input.iter().enumerate() {
                assert!(same(ln.read_value(&[i as u64]).await.unwrap(), value.ln()));
            }
            cleanup(&root).await;
        }
    };
}

exceptional!(complex32_exceptional, fensor::complex::Complex32, f32);
exceptional!(complex64_exceptional, fensor::complex::Complex64, f64);

#[tokio::test]
async fn projection_batching_and_live_sources() {
    use fensor::complex::Complex64 as C;
    let (root, tensor) = fixture::source(
        "complex_projection",
        shape![1, 2],
        Layout::Sparse { axis: None },
        2,
        1024,
        [C::new(0., 2.), C::new(0., 0.)],
    )
    .await;
    let projected = tensor.view().re().await.unwrap();
    let expanded = TensorExpression::new(projected.broadcast(shape![4097, 2]).unwrap())
        .unwrap()
        .into_dense()
        .exp()
        .await
        .unwrap()
        .flip(0)
        .unwrap();
    let expected: Vec<_> = (0..4097).flat_map(|_| [1., 1.]).collect();
    let mut dropped = expanded.read_blocks().unwrap();
    dropped.try_next().await.unwrap().unwrap();
    drop(dropped);
    let (a, b) = futures::try_join!(
        expanded.read_blocks().unwrap().try_collect::<Vec<_>>(),
        expanded.read_blocks().unwrap().try_collect::<Vec<_>>()
    )
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.into_iter().flatten().collect::<Vec<_>>(), expected);
    assert_eq!(
        TensorExpression::new(TensorCast::<i64>::cast(&tensor.view()).await.unwrap())
            .unwrap()
            .into_dense()
            .eq_scalar(0)
            .await
            .unwrap()
            .read_value(&[0, 0])
            .await
            .unwrap(),
        1
    );
    let (copy_root, dir) = common::new_dir("projection_boundary").await;
    let copy: Tensor<FsEntry, f64> = Tensor::copy_from(dir, &tensor.view().re().await.unwrap(), 2)
        .await
        .unwrap();
    assert_eq!(
        TensorExpression::new(copy.view())
            .unwrap()
            .into_dense()
            .exp()
            .await
            .unwrap()
            .read_value(&[0, 0])
            .await
            .unwrap(),
        1.
    );
    tensor.write_value(&[0, 0], C::new(1., 2.)).await.unwrap();
    assert_eq!(expanded.read_value(&[0, 0]).await.unwrap(), 1f64.exp());
    cleanup(&root).await;
    cleanup(&copy_root).await;
}
