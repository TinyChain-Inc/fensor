//! Shared execution paths for all real storage widths; integer edge expectations are independent.

use fensor::{
    Layout, TensorAbs, TensorBoolean, TensorBooleanScalar, TensorCompare, TensorCompareScalar,
    TensorElement, TensorFileEntry, TensorMatMul, TensorMath, TensorMathScalar, TensorMatrixUnary,
    TensorRead, TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorTransform,
    TensorUnaryBoolean, TensorWhere, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, Buffer, NDArrayMath, NDArrayMathScalar, NDArrayRead, Number, Real, axes, shape,
};

use common::{FsEntry, cleanup, fixture};

mod common;

async fn real_operations<T: TensorElement + Real<Abs = T>>(input: Vec<T>)
where
    FsEntry: TensorFileEntry<T>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
        let (root, tensor) =
            fixture::source("real_width", shape![2, 2], layout, 3, 1024, input.clone()).await;
        let view = tensor.view();
        let backend = Array::new(Buffer::from(input.clone()), shape![2, 2]).unwrap();
        // Concrete expression types stay in each invocation; no type erasure.
        macro_rules! arithmetic {
            ($method:ident, $scalar:ident) => {
                let mut expected = backend
                    .clone()
                    .$method(backend.clone())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .to_vec();
                if matches!(layout, Layout::Sparse { .. }) {
                    for (source, value) in input.iter().zip(&mut expected) {
                        if *source == T::ZERO {
                            *value = T::ZERO;
                        }
                    }
                }
                fixture::blocks(&view.$method(&view).await.unwrap(), &expected, |a, b| {
                    a == b
                })
                .await;
                let mut expected = backend
                    .clone()
                    .$scalar(T::ONE)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .to_vec();
                if matches!(layout, Layout::Sparse { .. }) {
                    for (source, value) in input.iter().zip(&mut expected) {
                        if *source == T::ZERO {
                            *value = T::ZERO;
                        }
                    }
                }
                fixture::blocks(&view.$scalar(T::ONE).await.unwrap(), &expected, |a, b| {
                    a == b
                })
                .await;
            };
        }
        arithmetic!(add, add_scalar);
        arithmetic!(sub, sub_scalar);
        arithmetic!(mul, mul_scalar);
        arithmetic!(div, div_scalar);
        arithmetic!(pow, pow_scalar);
        arithmetic!(rem, rem_scalar);
        let supported: Vec<u8> = input
            .iter()
            .map(|v| u8::from(matches!(layout, Layout::Dense) || *v != T::ZERO))
            .collect();
        macro_rules! comparison {
            ($method:ident, $scalar:ident, $predicate:expr) => {
                let expected: Vec<u8> = input
                    .iter()
                    .zip(&supported)
                    .map(|(&v, &s)| s * u8::from(($predicate)(v, v)))
                    .collect();
                fixture::blocks(&view.$method(&view).await.unwrap(), &expected, |a, b| {
                    a == b
                })
                .await;
                let expected: Vec<u8> = input
                    .iter()
                    .zip(&supported)
                    .map(|(&v, &s)| s * u8::from(($predicate)(v, T::ONE)))
                    .collect();
                fixture::blocks(&view.$scalar(T::ONE).await.unwrap(), &expected, |a, b| {
                    a == b
                })
                .await;
            };
        }
        comparison!(eq, eq_scalar, |a: T, b: T| a == b);
        comparison!(ne, ne_scalar, |a: T, b: T| a != b);
        comparison!(gt, gt_scalar, |a: T, b: T| a > b);
        comparison!(ge, ge_scalar, |a: T, b: T| a >= b);
        comparison!(lt, lt_scalar, |a: T, b: T| a < b);
        comparison!(le, le_scalar, |a: T, b: T| a <= b);
        comparison!(and, and_scalar, |a: T, b: T| a != T::ZERO && b != T::ZERO);
        comparison!(or, or_scalar, |a: T, b: T| a != T::ZERO || b != T::ZERO);
        comparison!(xor, xor_scalar, |a: T, b: T| (a != T::ZERO)
            != (b != T::ZERO));
        let expected: Vec<u8> = input
            .iter()
            .zip(&supported)
            .map(|(v, s)| s * u8::from(*v == T::ZERO))
            .collect();
        fixture::blocks(&view.not().await.unwrap(), &expected, |a, b| a == b).await;
        fixture::blocks(
            &view.abs().await.unwrap(),
            &input.iter().map(|v| Number::abs(*v)).collect::<Vec<_>>(),
            |a, b| a == b,
        )
        .await;
        let values: Vec<T> = input
            .iter()
            .zip(&supported)
            .filter(|(_, s)| **s != 0)
            .map(|(v, _)| *v)
            .collect();
        assert_eq!(
            tensor.sum_all().await.unwrap(),
            values.iter().copied().reduce(Number::add).unwrap()
        );
        assert_eq!(
            tensor.product_all().await.unwrap(),
            values.iter().copied().reduce(Number::mul).unwrap()
        );
        assert_eq!(
            tensor.min_all().await.unwrap(),
            values.iter().copied().reduce(Real::min).unwrap()
        );
        assert_eq!(
            tensor.max_all().await.unwrap(),
            values.iter().copied().reduce(Real::max).unwrap()
        );
        assert_eq!(
            tensor.all().await.unwrap(),
            values.iter().all(|v| *v != T::ZERO)
        );
        assert_eq!(
            tensor.any().await.unwrap(),
            values.iter().any(|v| *v != T::ZERO)
        );
        for keep in [false, true] {
            for axes in [axes![], axes![0], axes![1], axes![1, 0]] {
                // Independent per-group expected values and empty-support policy.
                let groups: Vec<Vec<usize>> = match axes.as_slice() {
                    [] => vec![vec![0], vec![1], vec![2], vec![3]],
                    [0] => vec![vec![0, 2], vec![1, 3]],
                    [1] => vec![vec![0, 1], vec![2, 3]],
                    _ => vec![vec![0, 1, 2, 3]],
                };
                macro_rules! axis {
                    ($method:ident, $combine:path) => {
                        let expected: Vec<_> = groups
                            .iter()
                            .map(|g| {
                                g.iter()
                                    .filter(|i| supported[**i] != 0)
                                    .map(|i| input[*i])
                                    .reduce($combine)
                                    .unwrap_or(T::ZERO)
                            })
                            .collect();
                        fixture::blocks(
                            &view.$method(axes.clone(), keep).await.unwrap(),
                            &expected,
                            |a, b| a == b,
                        )
                        .await;
                    };
                }
                axis!(sum, Number::add);
                axis!(product, Number::mul);
                axis!(min, Real::min);
                axis!(max, Real::max);
            }
        }
        fixture::consumers(
            &view.mt().await.unwrap(),
            &[input[0], input[2], input[1], input[3]],
            |a, b| a == b,
            |a, b| a == b,
        )
        .await;
        fixture::consumers(
            &view.diag().await.unwrap(),
            &[input[0], input[3]],
            |a, b| a == b,
            |a, b| a == b,
        )
        .await;
        let product = view.matmul(&view).await.unwrap();
        let expected: Vec<T> = (0..2)
            .flat_map(|i| {
                let input = &input;
                (0..2).map(move |j| {
                    Number::add(
                        Number::mul(input[i * 2], input[j]),
                        Number::mul(input[i * 2 + 1], input[2 + j]),
                    )
                })
            })
            .collect();
        fixture::consumers(&product, &expected, |a, b| a == b, |a, b| a == b).await;
        fixture::blocks(
            &view
                .eq(&view)
                .await
                .unwrap()
                .cond(&view, &view)
                .await
                .unwrap(),
            &input,
            |a, b| a == b,
        )
        .await;
        // Geometric writes, zero lifecycle, and repeated/concurrent consumption.
        view.flip(0)
            .unwrap()
            .write_value(&[0, 1], T::ONE)
            .await
            .unwrap();
        assert_eq!(tensor.read_value(&[1, 1]).await.unwrap(), T::ONE);
        tensor.write_value(&[1, 1], T::ZERO).await.unwrap();
        assert_eq!(tensor.read_value(&[1, 1]).await.unwrap(), T::ZERO);
        let expanded = tensor
            .view()
            .slice(fensor::Range::from_iter([
                fensor::AxisRange::In(0, 1, 1),
                fensor::AxisRange::In(0, 1, 1),
            ]))
            .unwrap()
            .broadcast(shape![1, 4097])
            .unwrap();
        let mut cancelled = expanded.read_blocks().unwrap();
        assert_eq!(cancelled.try_next().await.unwrap().unwrap().len(), 4096);
        drop(cancelled);
        let (a, b) = futures::try_join!(
            expanded.read_blocks().unwrap().try_collect::<Vec<_>>(),
            expanded.read_blocks().unwrap().try_collect::<Vec<_>>()
        )
        .unwrap();
        assert_eq!(a, b);
        drop(tensor);
        cleanup(&root).await;
    }
}

macro_rules! integer_case {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            real_operations::<$t>(vec![0, 1, 3, 2]).await;
            let (root, tensor) = fixture::source(
                "integer_edges",
                shape![4],
                Layout::Dense,
                2,
                1024,
                [<$t>::MAX, <$t>::MIN, 2, 0],
            )
            .await;
            let view = tensor.view();
            assert_eq!(
                view.add_scalar(1)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap(),
                <$t>::MAX.wrapping_add(1)
            );
            assert_eq!(
                view.sub_scalar(1)
                    .await
                    .unwrap()
                    .read_value(&[1])
                    .await
                    .unwrap(),
                <$t>::MIN.wrapping_sub(1)
            );
            assert_eq!(
                view.mul_scalar(2)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap(),
                <$t>::MAX.wrapping_mul(2)
            );
            assert_eq!(
                view.pow_scalar(8)
                    .await
                    .unwrap()
                    .read_value(&[2])
                    .await
                    .unwrap(),
                (2 as $t).wrapping_pow(8)
            );
            fixture::blocks(&view.div_scalar(0).await.unwrap(), &[0; 4], |a, b| a == b).await;
            fixture::blocks(&view.rem_scalar(0).await.unwrap(), &[0; 4], |a, b| a == b).await;
            assert_eq!(
                view.pow_scalar(0)
                    .await
                    .unwrap()
                    .read_value(&[3])
                    .await
                    .unwrap(),
                1
            );
            cleanup(&root).await;
        }
    };
}

integer_case!(u8_operations, u8);
integer_case!(u16_operations, u16);
integer_case!(u32_operations, u32);
integer_case!(u64_operations, u64);
integer_case!(i8_operations, i8);
integer_case!(i16_operations, i16);
integer_case!(i32_operations, i32);
integer_case!(i64_operations, i64);

macro_rules! signed_case {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            let (root, tensor) = fixture::source(
                "signed_min",
                shape![4],
                Layout::Dense,
                2,
                1024,
                [<$t>::MIN, -1, 2, 0],
            )
            .await;
            let v = tensor.view();
            assert_eq!(
                v.abs().await.unwrap().read_value(&[0]).await.unwrap(),
                <$t>::MIN
            );
            assert_eq!(
                v.div_scalar(-1)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap(),
                <$t>::MIN
            );
            assert_eq!(
                v.rem_scalar(-1)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap(),
                0
            );
            fixture::blocks(&v.pow_scalar(-3).await.unwrap(), &[0, -1, 0, 0], |a, b| {
                a == b
            })
            .await;
            fixture::blocks(
                &v.pow_scalar(<$t>::MIN).await.unwrap(),
                &[0, 1, 0, 0],
                |a, b| a == b,
            )
            .await;
            cleanup(&root).await;
        }
    };
}

signed_case!(i8_negative_powers, i8);
signed_case!(i16_negative_powers, i16);
signed_case!(i32_negative_powers, i32);
signed_case!(i64_negative_powers, i64);

#[tokio::test]
async fn full_width_powers() {
    let (root, tensor) = fixture::source(
        "full_width_power",
        shape![1],
        Layout::Dense,
        1,
        1024,
        [2u64],
    )
    .await;
    assert_eq!(
        tensor
            .view()
            .pow_scalar(1u64 << 32)
            .await
            .unwrap()
            .read_value(&[0])
            .await
            .unwrap(),
        0
    );
    cleanup(&root).await;
    let (root, tensor) = fixture::source(
        "signed_width_power",
        shape![1],
        Layout::Dense,
        1,
        1024,
        [-1i64],
    )
    .await;
    assert_eq!(
        tensor
            .view()
            .pow_scalar((1i64 << 32) + 1)
            .await
            .unwrap()
            .read_value(&[0])
            .await
            .unwrap(),
        -1
    );
    cleanup(&root).await;
}
