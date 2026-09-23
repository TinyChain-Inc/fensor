//! Filesystem-backed reduction semantics and bounded consumption.
use fensor::{
    Layout, Tensor, TensorCompareScalar, TensorElement, TensorFileEntry, TensorGeometry,
    TensorMath, TensorRead, TensorReduce, TensorReduceAll, TensorReduceBoolean, TensorSchema,
    TensorTransform, TensorUnary, TensorWhere, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, AxisRange, Buffer, NDArrayRead, NDArrayReduce, NDArrayReduceAll, Number, Shape, axes,
    range, shape,
};
use number_general::DType;

mod common;
use common::{FsEntry, new_dir};

async fn source<T: TensorElement>(values: Vec<T>, shape: Shape, sparse: bool) -> Tensor<FsEntry, T>
where
    FsEntry: TensorFileEntry<T>,
{
    let (_, dir) = new_dir("reduce").await;
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(T::dtype(), shape.clone()).unwrap(),
        if sparse {
            Layout::Sparse { axis: None }
        } else {
            Layout::Dense
        },
        31,
    )
    .await
    .unwrap();
    for (coord, value) in common::iter_coords(&shape).zip(values) {
        // An implicit zero needs no write; explicit sparse zero writes delete index rows.
        if !sparse || value != T::ZERO {
            tensor.write_value(&coord, value).await.unwrap();
        }
    }
    tensor
}

async fn check<V: TensorRead>(view: &V, expected: &[V::DType])
where
    V::DType: TensorElement,
    FsEntry: TensorFileEntry<V::DType>,
{
    let blocks: Vec<_> = view.read_blocks().unwrap().try_collect().await.unwrap();
    assert!(blocks.iter().all(|b| b.len() <= 4096));
    let values: Vec<_> = blocks.into_iter().flatten().collect();
    assert_eq!(values, expected);
    for (coord, value) in common::iter_coords(view.shape()).zip(expected) {
        assert_eq!(view.read_value(&coord).await.unwrap(), *value);
    }
    let (root, dir) = new_dir("reduce_copy").await;
    let copy: Tensor<FsEntry, V::DType> = Tensor::copy_from(dir.clone(), view, 17).await.unwrap();
    copy.sync().await.unwrap();
    drop(copy);
    drop(dir);
    let copy = Tensor::<FsEntry, V::DType>::load(common::open_dir(&root).unwrap())
        .await
        .unwrap();
    let copied: Vec<_> = copy
        .read_blocks()
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(copied, expected);
    if matches!(view.layout(), Layout::Sparse { .. }) {
        let range = view
            .shape()
            .iter()
            .map(|d| AxisRange::In(0, *d, 1))
            .collect();
        let actual: Vec<_> = view
            .read_sparse_elements_in_order(range, (0..view.ndim()).collect())
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let expected: Vec<_> = common::iter_coords(view.shape())
            .zip(expected.iter().copied())
            .filter(|(_, v)| *v != V::DType::ZERO)
            .collect();
        assert_eq!(actual, expected);
    }
}

macro_rules! parity {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for n in [1, 7, 8, 9, 63, 64, 65, 129, 4095, 4096, 4097] {
                let values: Vec<$t> = (0..n).map(|i| (i % 3) as $t).collect();
                let tensor = source(values.clone(), shape![n], false).await;
                macro_rules! op {
                    ($method:ident, $all:ident) => {
                        let array = Array::new(Buffer::from(values.clone()), shape![n]).unwrap();
                        let expected = array.$all().unwrap();
                        assert_eq!(tensor.$all().await.unwrap(), expected);
                        check(
                            &tensor.view().$method(axes![0], false).await.unwrap(),
                            &[expected],
                        )
                        .await;
                    };
                }
                op!(sum, sum_all);
                op!(product, product_all);
                op!(min, min_all);
                op!(max, max_all);
                assert!(!tensor.all().await.unwrap());
                assert_eq!(tensor.any().await.unwrap(), n > 1);
            }
            let values: Vec<$t> = (0..24).map(|i| (i % 4) as $t).collect();
            let tensor = source(values.clone(), shape![2, 3, 4], false).await;
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
                        ($method:ident) => {
                            let expected =
                                Array::new(Buffer::from(values.clone()), shape![2, 3, 4])
                                    .unwrap()
                                    .$method(axes.clone(), keepdims)
                                    .unwrap();
                            let view = tensor.view().$method(axes.clone(), keepdims).await.unwrap();
                            assert_eq!(view.shape(), ha_ndarray::NDArray::shape(&expected));
                            check(&view, &expected.buffer().unwrap().to_slice().unwrap()).await;
                        };
                    }
                    op!(sum);
                    op!(product);
                    op!(min);
                    op!(max);
                }
            }
            assert!(tensor.view().sum(axes![3], false).await.is_err());
        }
    };
}
parity!(f32_parity, f32);
parity!(f64_parity, f64);
parity!(u8_parity, u8);

#[tokio::test]
async fn sparse_support_empty_groups_and_copy_boundary() {
    let tensor = source(vec![0f32, 0., 0.2, 0., 2., 3.], shape![3, 2], true).await;
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
    let (_, dir) = new_dir("reduced_support").await;
    let copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &sum, 2).await.unwrap();
    assert_eq!(copy.min_all().await.unwrap(), 5.);
    assert_eq!(sum.min_all().await.unwrap(), 0.);
    let empty = source(vec![0u8; 6], shape![2, 3], true).await;
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
    let a = source((1..=12).map(|v| v as f64).collect(), shape![2, 3, 2], false).await;
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
            .slice(range![
                AxisRange::At(1),
                AxisRange::Of(vec![1, 0, 1].into())
            ])
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
            let tensor = source(
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
            check(
                &tensor.view().product(axes![1], false).await.unwrap(),
                &[0 as $t, 6 as $t],
            )
            .await;
            check(
                &tensor.view().min(axes![1], false).await.unwrap(),
                &[0 as $t, 2 as $t],
            )
            .await;
            check(
                &tensor.view().max(axes![1], false).await.unwrap(),
                &[0 as $t, 3 as $t],
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
            let zeros = source(vec![-0. as $t, 0. as $t], shape![2], false).await;
            assert_eq!(
                zeros.min_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            assert_eq!(
                zeros.max_all().await.unwrap().to_bits(),
                (0. as $t).to_bits()
            );
            let negative = source(vec![-0. as $t], shape![1], false).await;
            assert_eq!(
                negative.sum_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            assert_eq!(
                negative.product_all().await.unwrap().to_bits(),
                (-0. as $t).to_bits()
            );
            let nan = source(vec![1 as $t, <$t>::NAN], shape![2], false).await;
            assert!(nan.min_all().await.unwrap().is_nan());
            assert!(nan.max_all().await.unwrap().is_nan());
            assert!(nan.sum_all().await.unwrap().is_nan());
            assert!(nan.product_all().await.unwrap().is_nan());
            assert!(nan.all().await.unwrap());
            for value in [<$t>::INFINITY, <$t>::NEG_INFINITY] {
                let tensor = source(vec![value; 7], shape![7], false).await;
                assert_eq!(tensor.min_all().await.unwrap(), value);
                assert_eq!(tensor.max_all().await.unwrap(), value);
            }
            let subnormal = <$t>::from_bits(1);
            let tiny = source(vec![subnormal; 2], shape![2], false).await;
            assert_eq!(tiny.sum_all().await.unwrap(), <$t>::from_bits(2));
        }
    };
}
exceptional!(f32_exceptional, f32);
exceptional!(f64_exceptional, f64);

#[tokio::test]
async fn integer_wrapping_is_independent_of_batching() {
    let tensor = source(vec![255u8; 4097], shape![4097], false).await;
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
                    let tensor = source(values.clone(), shape![1, n], false).await;
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
async fn selected_output_range_and_corruption_boundaries() {
    use fensor::TensorSparseIndex;

    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (_, dir) = new_dir("reduce_far_range").await;
        let tensor = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(f32::dtype(), shape![500_000_000, 2]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&[499_999_999, 0], 2.).await.unwrap();
        tensor.write_value(&[0, 0], 1.).await.unwrap();
        // Locate the corrupt block using the same sparse key convention as storage.
        let key = vec![0, 0];
        let id = tensor.lookup_block_id(&key).await.unwrap().unwrap();
        dir.read()
            .await
            .get_dir("blocks")
            .unwrap()
            .write()
            .await
            .delete(&id.to_string())
            .await;
        let view = tensor.view().sum(axes![1], false).await.unwrap();
        let entries: Vec<_> = view
            .read_sparse_elements_in_order(
                range![AxisRange::Of(
                    vec![499_999_999, 499_999_998, 499_999_999].into()
                )],
                axes![0],
            )
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(entries, vec![(vec![499_999_999], 2.)]);
        assert!(view.read_value(&[0]).await.is_err());
        assert!(
            view.read_sparse_elements_in_order(range![AxisRange::At(500_000_000)], axes![0])
                .await
                .is_err()
        );
        assert!(
            view.read_sparse_elements_in_order(range![AxisRange::At(1)], axes![1])
                .await
                .is_err()
        );
    })
    .await
    .expect("selected reduction must not enumerate unrelated outputs");
}

#[tokio::test]
async fn boolean_short_circuit_and_numeric_error_propagation() {
    use fensor::TensorSparseIndex;

    let (_, dir) = new_dir("reduce_corrupt_boolean").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(f32::dtype(), shape![8193]).unwrap(),
        Layout::Sparse { axis: None },
        1,
    )
    .await
    .unwrap();
    tensor.write_value(&[0], 0.2).await.unwrap();
    tensor.write_value(&[8192], 1.).await.unwrap();
    let id = tensor
        .lookup_block_id(&[8192, 8192])
        .await
        .unwrap()
        .unwrap();
    dir.read()
        .await
        .get_dir("blocks")
        .unwrap()
        .write()
        .await
        .delete(&id.to_string())
        .await;
    assert!(tensor.any().await.unwrap());
    assert!(!tensor.view().round().await.unwrap().all().await.unwrap());
    assert!(tensor.all().await.is_err());
    assert!(tensor.view().round().await.unwrap().any().await.is_err());
    assert!(tensor.sum_all().await.is_err());
}

#[tokio::test]
async fn repeated_concurrent_and_dropped_reduction_streams() {
    let tensor = source(vec![1u8; 8202], shape![4101, 2], false).await;
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
            cache.load(root).unwrap(),
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
        let dir = cache.load(root.clone()).unwrap();
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
