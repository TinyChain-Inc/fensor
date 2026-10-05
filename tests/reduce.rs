//! Filesystem-backed reduction semantics and bounded consumption.
use fensor::{
    AxisRange, Layout, Shape, Tensor, TensorCompareScalar, TensorElement, TensorFileEntry,
    TensorGeometry, TensorMath, TensorRead, TensorReduce, TensorReduceAll, TensorReduceBoolean,
    TensorSchema, TensorTransform, TensorUnary, TensorWhere, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{Array, Buffer, NDArrayRead, NDArrayReduce, NDArrayReduceAll, axes, range, shape};
use number_general::DType;

mod common;
use common::{FsEntry, new_dir};

async fn source<T: TensorElement>(
    values: Vec<T>,
    shape: Shape,
    sparse: bool,
) -> (common::Directory, Tensor<FsEntry, T>)
where
    FsEntry: TensorFileEntry<T>,
{
    common::fixture::source(
        "reduce",
        shape,
        common::fixture::layout(sparse),
        31,
        1_000_000,
        values,
    )
    .await
}

async fn check<V: TensorRead>(view: &V, expected: &[V::DType])
where
    FsEntry: TensorFileEntry<V::DType>,
    V::DType: TensorElement,
{
    common::fixture::consumers(view, expected, |a, b| a == b, |a, b| a == b).await;
}

macro_rules! parity {
    ($name:ident, $t:ty, $large_consumers:expr) => {
        #[tokio::test]
        async fn $name() {
            for sparse in [false, true] {
                for n in [1, 7, 8, 9, 63, 64, 65, 129, 4095, 4096, 4097] {
                    if sparse && !matches!(n, 7 | 4097) {
                        continue;
                    }
                    let values: Vec<$t> = (0..n)
                        .map(|i| {
                            if sparse {
                                u64::from(i % 257 == 0) as $t
                            } else {
                                (i % 3) as $t
                            }
                        })
                        .collect();
                    let (_tensor_root, tensor) = source(values.clone(), shape![n], sparse).await;
                    let supported: Vec<_> = values
                        .iter()
                        .copied()
                        .filter(|v| !sparse || *v != 0 as $t)
                        .collect();
                    macro_rules! op {
                        ($method:ident, $all:ident, $consumers:expr) => {
                            let array = Array::new(
                                Buffer::from(supported.clone()),
                                shape![supported.len()],
                            )
                            .unwrap();
                            let expected = array.$all().unwrap();
                            assert_eq!(tensor.$all().await.unwrap(), expected);
                            let view = tensor.view().$method(axes![0], false).await.unwrap();
                            if $consumers && (n == 7 || n == 4097 && $large_consumers) {
                                check(&view, &[expected]).await;
                            } else {
                                common::fixture::blocks(&view, &[expected], |a, b| a == b).await;
                            }
                        };
                    }
                    op!(sum, sum_all, true);
                    op!(product, product_all, false);
                    op!(min, min_all, false);
                    op!(max, max_all, false);
                    assert_eq!(tensor.all().await.unwrap(), sparse);
                    assert_eq!(tensor.any().await.unwrap(), sparse || n > 1);
                }
                let values: Vec<$t> = (0..24).map(|i| (i % 4 + u64::from(sparse)) as $t).collect();
                let (_tensor_root, tensor) = source(values.clone(), shape![2, 3, 4], sparse).await;
                for axes in [
                    axes![],
                    axes![0],
                    axes![1],
                    axes![2],
                    axes![2, 0, 2],
                    axes![2, 1, 0],
                ] {
                    for keepdims in [false, true] {
                        macro_rules! op {
                            ($method:ident, $consumers:expr) => {
                                let expected =
                                    Array::new(Buffer::from(values.clone()), shape![2, 3, 4])
                                        .unwrap()
                                        .$method(axes.clone(), keepdims)
                                        .unwrap();
                                let view =
                                    tensor.view().$method(axes.clone(), keepdims).await.unwrap();
                                assert_eq!(
                                    view.shape(),
                                    ha_ndarray::NDArray::shape(&expected)
                                        .iter()
                                        .map(|&d| d as u64)
                                        .collect::<Vec<_>>()
                                );
                                let buffer = expected.buffer().unwrap();
                                let expected = buffer.to_slice().unwrap();
                                if $consumers && axes.as_slice() == [1] {
                                    check(&view, &expected).await;
                                } else {
                                    common::fixture::blocks(&view, &expected, |a, b| a == b).await;
                                }
                            };
                        }
                        op!(sum, true);
                        op!(product, false);
                        op!(min, false);
                        op!(max, false);
                    }
                }
                assert!(tensor.view().sum(axes![3], false).await.is_err());
            }
        }
    };
}
parity!(f32_parity, f32, true);
parity!(f64_parity, f64, false);
parity!(u8_parity, u8, false);

#[tokio::test]
async fn sparse_support_empty_groups_and_copy_boundary() {
    let (_tensor_root, tensor) = source(vec![0f32, 0., 0.2, 0., 2., 3.], shape![3, 2], true).await;
    assert_eq!(tensor.sum_all().await.unwrap(), 5.2);
    assert_eq!(tensor.min_all().await.unwrap(), 0.2);
    assert!(tensor.all().await.unwrap());
    let rounded = tensor.view().round().await.unwrap();
    assert!(!rounded.all().await.unwrap());
    assert_eq!(rounded.product_all().await.unwrap(), 0.);
    let sum = rounded.sum(axes![1], false).await.unwrap();
    check(&sum, &[0., 0., 5.]).await;
    check(&sum.eq_scalar(0.).await.unwrap(), &[0, 1, 0]).await;
    check(
        &rounded.product(axes![1], false).await.unwrap(),
        &[0., 0., 6.],
    )
    .await;
    check(&rounded.min(axes![1], false).await.unwrap(), &[0., 0., 2.]).await;
    check(&rounded.max(axes![1], false).await.unwrap(), &[0., 0., 3.]).await;
    let (_dir_root, dir) = new_dir("reduced_support").await;
    let copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &sum, 2).await.unwrap();
    assert_eq!(copy.min_all().await.unwrap(), 5.);
    assert_eq!(sum.min_all().await.unwrap(), 0.);
    let (_empty_root, empty) = source(vec![0u8; 6], shape![2, 3], true).await;
    assert_eq!(empty.sum_all().await.unwrap(), 0);
    assert_eq!(empty.product_all().await.unwrap(), 1);
    assert!(empty.all().await.unwrap());
    assert!(!empty.any().await.unwrap());
    assert!(matches!(
        empty.min_all().await,
        Err(fensor::Error::Unsupported(_))
    ));
    assert!(matches!(
        empty.max_all().await,
        Err(fensor::Error::Unsupported(_))
    ));
    check(
        &empty.view().product(axes![1], false).await.unwrap(),
        &[0, 0],
    )
    .await;
    check(&empty.view().min(axes![1], false).await.unwrap(), &[0, 0]).await;
}

#[tokio::test]
async fn composition_transforms_and_live_sources() {
    let (_a_root, a) = source((1..=12).map(|v| v as f64).collect(), shape![2, 3, 2], false).await;
    let view = a.view().sum(axes![1], false).await.unwrap();
    check(&view, &[9., 12., 27., 30.]).await;
    check(&view.clone().transpose(None).unwrap(), &[9., 27., 12., 30.]).await;
    check(
        &view.clone().reshape(shape![4]).unwrap().flip(0).unwrap(),
        &[30., 27., 12., 9.],
    )
    .await;
    check(
        &view
            .clone()
            .slice(range![AxisRange::At(1), AxisRange::Of(vec![1, 0, 1])])
            .unwrap(),
        &[30., 27., 30.],
    )
    .await;
    let single = view
        .clone()
        .slice(range![AxisRange::In(0, 1, 1), AxisRange::In(0, 2, 1)])
        .unwrap();
    check(
        &single.broadcast(shape![2, 2]).unwrap(),
        &[9., 12., 9., 12.],
    )
    .await;
    check(
        &view
            .clone()
            .unsqueeze(axes![0])
            .unwrap()
            .squeeze(axes![0])
            .unwrap(),
        &[9., 12., 27., 30.],
    )
    .await;
    let transformed = a
        .view()
        .transpose(Some(axes![2, 1, 0]))
        .unwrap()
        .sum(axes![1], false)
        .await
        .unwrap();
    check(&transformed, &[9., 27., 12., 30.]).await;
    assert_eq!(
        view.sum(axes![0], true)
            .await
            .unwrap()
            .sum_all()
            .await
            .unwrap(),
        78.
    );
    let selected = view
        .gt_scalar(20.)
        .await
        .unwrap()
        .cond(&view, &view.add(&view).await.unwrap())
        .await
        .unwrap();
    check(&selected.sum(axes![0], false).await.unwrap(), &[45., 54.]).await;
    a.write_value(&[0, 0, 0], 2.).await.unwrap();
    assert_eq!(view.read_value(&[0, 0]).await.unwrap(), 10.);
    let scalar = view
        .slice(range![AxisRange::At(0), AxisRange::At(0)])
        .unwrap();
    assert_eq!(scalar.read_value(&[]).await.unwrap(), 10.);
    assert!(scalar.sum_all().await.is_err());
    assert!(scalar.sum(axes![], false).await.is_err());
}

macro_rules! sparse_parity {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            let (_tensor_root, tensor) = source(
                vec![0 as $t, 0 as $t, 0 as $t, 2 as $t, 0 as $t, 3 as $t],
                shape![2, 3],
                true,
            )
            .await;
            check(
                &tensor.view().sum(axes![1], false).await.unwrap(),
                &[0 as $t, 5 as $t],
            )
            .await;
            common::fixture::blocks(
                &tensor.view().product(axes![1], false).await.unwrap(),
                &[0 as $t, 6 as $t],
                |a, b| a == b,
            )
            .await;
            common::fixture::blocks(
                &tensor.view().min(axes![1], false).await.unwrap(),
                &[0 as $t, 2 as $t],
                |a, b| a == b,
            )
            .await;
            common::fixture::blocks(
                &tensor.view().max(axes![1], false).await.unwrap(),
                &[0 as $t, 3 as $t],
                |a, b| a == b,
            )
            .await;
            assert_eq!(tensor.sum_all().await.unwrap(), 5 as $t);
            assert_eq!(tensor.product_all().await.unwrap(), 6 as $t);
            assert_eq!(tensor.min_all().await.unwrap(), 2 as $t);
            assert_eq!(tensor.max_all().await.unwrap(), 3 as $t);
            assert!(tensor.all().await.unwrap());
            assert!(tensor.any().await.unwrap());
        }
    };
}
sparse_parity!(sparse_f32, f32);
sparse_parity!(sparse_f64, f64);
sparse_parity!(sparse_u8, u8);

macro_rules! exceptional {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            let (_zeros_root, zeros) = source(vec![-0. as $t, 0. as $t], shape![2], false).await;
            assert_eq!(
                zeros.min_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            assert_eq!(
                zeros.max_all().await.unwrap().to_bits(),
                (0. as $t).to_bits()
            );
            let (_negative_root, negative) = source(vec![-0. as $t], shape![1], false).await;
            assert_eq!(
                negative.sum_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            assert_eq!(
                negative.product_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            let (_nan_root, nan) = source(vec![1 as $t, <$t>::NAN], shape![2], false).await;
            assert!(nan.min_all().await.unwrap().is_nan());
            assert!(nan.max_all().await.unwrap().is_nan());
            assert!(nan.sum_all().await.unwrap().is_nan());
            assert!(nan.product_all().await.unwrap().is_nan());
            assert!(nan.all().await.unwrap());
            for value in [<$t>::INFINITY, <$t>::NEG_INFINITY] {
                let (_tensor_root, tensor) = source(vec![value; 7], shape![7], false).await;
                assert_eq!(tensor.min_all().await.unwrap(), value);
                assert_eq!(tensor.max_all().await.unwrap(), value);
            }
            let subnormal = <$t>::from_bits(1);
            let (_tiny_root, tiny) = source(vec![subnormal; 2], shape![2], false).await;
            assert_eq!(tiny.sum_all().await.unwrap(), <$t>::from_bits(2));
        }
    };
}
exceptional!(f32_exceptional, f32);
exceptional!(f64_exceptional, f64);

#[tokio::test]
async fn integer_wrapping_is_independent_of_batching() {
    let (_tensor_root, tensor) = source(vec![255u8; 4097], shape![4097], false).await;
    assert_eq!(tensor.sum_all().await.unwrap(), 255);
    assert_eq!(tensor.product_all().await.unwrap(), 255);
    check(&tensor.view().sum(axes![0], false).await.unwrap(), &[255]).await;
}

fn exact_bound(values: &[f64], actual: f64, unit_roundoff: f64, product: bool) {
    use num_rational::BigRational;
    use num_traits::Signed;

    let zero = BigRational::from_integer(0.into());
    let one = BigRational::from_integer(1.into());
    let mut exact = if product { one.clone() } else { zero.clone() };
    let mut scale = zero;
    let mut numerator = one.numer().clone();
    let mut denominator = one.denom().clone();
    for value in values {
        let value = BigRational::from_float(*value).unwrap();
        scale += value.abs();
        if product {
            numerator *= value.numer();
            denominator *= value.denom();
        } else {
            exact += value;
        }
    }
    if product {
        // Normalize once instead of repeatedly taking GCDs of growing exact products.
        exact = BigRational::new(numerator, denominator);
        scale = exact.abs();
    }
    let ku = BigRational::from_float(unit_roundoff).unwrap()
        * BigRational::from_integer((8 * values.len()).into());
    assert!(ku < one);
    let gamma = &ku / (one - &ku);
    let error = (BigRational::from_float(actual).unwrap() - exact).abs();
    assert!(
        error <= gamma * scale,
        "aggregate exceeded its forward-error bound"
    );
}

macro_rules! accuracy {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for n in [7, 65, 129, 4097] {
                for product in [false, true] {
                    let values: Vec<$t> = (0..n)
                        .map(|i| {
                            if product {
                                (1. + ((i % 7) as f64 - 3.) / 65536.) as $t
                            } else {
                                [16., -16., 0.125, 0.0009765625, -0.0625][i % 5] as $t
                            }
                        })
                        .collect();
                    let (_tensor_root, tensor) =
                        source(values.clone(), shape![1 as u64, n as u64], false).await;
                    let actual = if product {
                        tensor.product_all().await.unwrap()
                    } else {
                        tensor.sum_all().await.unwrap()
                    };
                    let exact_values: Vec<f64> = values.iter().map(|v| *v as f64).collect();
                    exact_bound(
                        &exact_values,
                        actual as f64,
                        <$t>::EPSILON as f64 / 2.,
                        product,
                    );
                    let axis = if product {
                        tensor
                            .view()
                            .product(axes![1], false)
                            .await
                            .unwrap()
                            .read_value(&[0])
                            .await
                            .unwrap()
                    } else {
                        tensor
                            .view()
                            .sum(axes![1], false)
                            .await
                            .unwrap()
                            .read_value(&[0])
                            .await
                            .unwrap()
                    };
                    exact_bound(
                        &exact_values,
                        axis as f64,
                        <$t>::EPSILON as f64 / 2.,
                        product,
                    );
                }
            }
        }
    };
}
accuracy!(f32_exact_accuracy, f32);
accuracy!(f64_exact_accuracy, f64);

#[tokio::test]
async fn repeated_concurrent_and_dropped_reduction_streams() {
    let (_tensor_root, tensor) = source(vec![1u8; 8202], shape![4101, 2], false).await;
    let view = tensor.view().sum(axes![1], false).await.unwrap();
    let mut dropped = view.read_blocks().unwrap();
    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let consume = || async {
        view.read_blocks()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    };
    let (left, right) = futures::join!(consume(), consume());
    assert_eq!(left, right);
    assert_eq!(left.iter().map(Vec::len).collect::<Vec<_>>(), vec![4096, 5]);
    assert!(left.into_iter().flatten().all(|v| v == 2));
}

#[tokio::test]
async fn reduction_copy_spills_and_reloads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("reduce_cache_source").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let tensor = Tensor::<FsEntry, u8>::create(
            cache.load(root.to_path_buf()).unwrap(),
            TensorSchema::new(u8::dtype(), shape![1024, 4]).unwrap(),
            Layout::Dense,
            32,
        )
        .await
        .unwrap();
        tensor.write_value(&[1023, 3], 255).await.unwrap();
        let view = tensor.view().sum(axes![1], false).await.unwrap();
        let (root, _) = new_dir("reduce_cache_output").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root.to_path_buf()).unwrap();
        let copy: Tensor<FsEntry, u8> = Tensor::copy_from(dir.clone(), &view, 32).await.unwrap();
        copy.sync().await.unwrap();
        drop(copy);
        drop(dir);
        let copy = Tensor::<FsEntry, u8>::load(common::open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(copy.read_value(&[1023]).await.unwrap(), 255);
        assert_eq!(copy.sum_all().await.unwrap(), 255);
    })
    .await
    .expect("reductions must make progress under cache pressure");
}

#[tokio::test]
async fn statistics_count_original_support_across_batches_and_axes() {
    use fensor::{TensorMathScalar, TensorStatistics};
    for sparse in [false, true] {
        let (_tensor_root, tensor) =
            source(vec![2.0f64, 0.0, 4.0, 0.0, 0.0, 0.0], shape![2, 3], sparse).await;
        let shifted = tensor.view().sub_scalar(2.0).await.unwrap();
        let expected = if sparse { 1.0 } else { -1.0 };
        assert_eq!(shifted.mean_all().await.unwrap(), expected);
        let means = shifted.mean(axes![1], true).await.unwrap();
        assert_eq!(means.shape(), &[2, 1]);
        assert_eq!(
            means.read_value(&[0, 0]).await.unwrap(),
            if sparse { 1.0 } else { 0.0 }
        );
        if sparse {
            assert!(means.read_value(&[1, 0]).await.unwrap().is_nan());
            assert_eq!(shifted.std_all().await.unwrap(), 1.0);
            assert_eq!(shifted.norm_all().await.unwrap(), 2.0);
        }
    }
    let (_tensor_root, tensor) = source(vec![3.0f64; 8193], shape![8193], false).await;
    assert_eq!(tensor.mean_all().await.unwrap(), 3.0);
    assert_eq!(tensor.std_all().await.unwrap(), 0.0);
    assert_eq!(tensor.norm_all().await.unwrap(), (9.0f64 * 8193.0).sqrt());
    let (_empty_root, empty) = source(vec![0.0f64; 7], shape![7], true).await;
    assert!(empty.mean_all().await.unwrap().is_nan());
    assert!(empty.std_all().await.unwrap().is_nan());
    assert_eq!(empty.norm_all().await.unwrap(), 0.0);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (_tensor_root, tensor) = source(vec![value], shape![1], true).await;
        assert!(tensor.std_all().await.unwrap().is_nan());
        let mean = tensor.mean_all().await.unwrap();
        assert!(if value.is_nan() {
            mean.is_nan()
        } else {
            mean == value
        });
        assert!(if value.is_nan() {
            tensor.norm_all().await.unwrap().is_nan()
        } else {
            tensor.norm_all().await.unwrap().is_infinite()
        });
    }
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex_statistics_use_magnitude_and_preserve_mean_dtype() {
    use fensor::{TensorMathScalar, TensorStatistics, complex::Complex64};
    for sparse in [false, true] {
        let (_tensor_root, tensor) = source(
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            shape![2],
            sparse,
        )
        .await;
        assert_eq!(tensor.mean_all().await.unwrap(), Complex64::new(2.0, 3.0));
        assert_eq!(tensor.std_all().await.unwrap(), 2.0f64.sqrt());
        assert_eq!(tensor.norm_all().await.unwrap(), 30.0f64.sqrt());
        let zeros = tensor
            .view()
            .sub_scalar(Complex64::new(1.0, 2.0))
            .await
            .unwrap();
        assert_eq!(zeros.mean_all().await.unwrap(), Complex64::new(1.0, 1.0));
        let mean = zeros.mean(axes![0], true).await.unwrap();
        assert_eq!(
            mean.read_value(&[0]).await.unwrap(),
            Complex64::new(1.0, 1.0)
        );
        assert_eq!(
            zeros
                .std(axes![0], false)
                .await
                .unwrap()
                .read_value(&[0])
                .await
                .unwrap(),
            2.0f64.sqrt()
        );
    }
}

macro_rules! statistics_dtype {
    ($name:ident, $dtype:ty) => {
        #[tokio::test]
        async fn $name() {
            use fensor::TensorStatistics;
            for sparse in [false, true] {
                let (_tensor_root, tensor) = source(
                    vec![1 as $dtype, 2 as $dtype, 3 as $dtype, 4 as $dtype],
                    shape![2, 2],
                    sparse,
                )
                .await;
                assert_eq!(tensor.mean_all().await.unwrap(), 2.5f64);
                assert_eq!(tensor.std_all().await.unwrap(), 1.25f64.sqrt());
                assert_eq!(tensor.norm_all().await.unwrap(), 30.0f64.sqrt());
                let means = tensor.view().mean(axes![1], false).await.unwrap();
                assert_eq!(means.read_value(&[1]).await.unwrap(), 3.5f64);
            }
        }
    };
}
statistics_dtype!(statistics_u8, u8);
statistics_dtype!(statistics_u16, u16);
statistics_dtype!(statistics_u32, u32);
statistics_dtype!(statistics_u64, u64);
statistics_dtype!(statistics_i8, i8);
statistics_dtype!(statistics_i16, i16);
statistics_dtype!(statistics_i32, i32);
statistics_dtype!(statistics_i64, i64);
statistics_dtype!(statistics_f32, f32);
statistics_dtype!(statistics_f64, f64);

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex32_statistics_promote_to_complex64_and_float64() {
    use fensor::{
        TensorStatistics,
        complex::{Complex32, Complex64},
    };
    let (_tensor_root, tensor) = source(
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        shape![2],
        true,
    )
    .await;
    let mean: Complex64 = tensor.mean_all().await.unwrap();
    assert_eq!(mean, Complex64::new(2.0, 3.0));
    assert_eq!(tensor.std_all().await.unwrap(), 2.0f64.sqrt());
    assert_eq!(tensor.norm_all().await.unwrap(), 30.0f64.sqrt());
}
