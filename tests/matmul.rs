//! Filesystem-backed tiled matrix multiplication and zero-extended sparse values.

use fensor::{
    AxisRange, Layout, Shape, Tensor, TensorCast, TensorCompareScalar, TensorElement,
    TensorExpression, TensorFileEntry, TensorGeometry, TensorMatMul, TensorMath, TensorRead,
    TensorReduce, TensorReduceAll, TensorSchema, TensorTransform, TensorUnary, TensorWhere,
    TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{Array, Buffer, MatrixDual, NDArrayRead, Number, axes, range, shape};
use number_general::DType;

mod common;
use common::{FsEntry, new_dir};

async fn source<T: TensorElement>(
    values: &[T],
    dims: Shape,
    sparse: bool,
    capacity: usize,
) -> (common::Directory, Tensor<FsEntry, T>)
where
    FsEntry: TensorFileEntry<T>,
{
    common::fixture::source(
        "matmul",
        dims,
        common::fixture::layout(sparse),
        capacity,
        1_000_000,
        values.iter().copied(),
    )
    .await
}

async fn check_blocks<V: TensorRead>(view: &V, expected: &[V::DType])
where
    V::DType: TensorElement,
{
    common::fixture::blocks(view, expected, common::numbers::same).await;
}

async fn check<V: TensorRead>(view: &V, expected: &[V::DType])
where
    V::DType: TensorElement,
    FsEntry: TensorFileEntry<V::DType>,
{
    common::fixture::consumers(view, expected, common::numbers::same, |a, b| {
        common::numbers::same(a, b) || (a == V::DType::ZERO && b == V::DType::ZERO)
    })
    .await;
}

macro_rules! parity {
    ($name:ident, $t:ty, $large_consumers:expr) => {
        #[tokio::test]
        async fn $name() {
            for (m, k, n) in [
                (1, 1, 1),
                (2, 3, 2),
                (8, 8, 8),
                (9, 17, 7),
                (33, 129, 35),
                (2, 127, 3),
                (2, 128, 3),
                (1, 4097, 2),
            ] {
                let av: Vec<$t> = (0..m * k).map(|i| (i % 3) as $t).collect();
                let bv: Vec<$t> = (0..k * n).map(|i| ((i + 1) % 4) as $t).collect();
                let reference = Array::new(Buffer::from(av.clone()), shape![m, k])
                    .unwrap()
                    .matmul(Array::new(Buffer::from(bv.clone()), shape![k, n]).unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec();
                for (ls, rs) in [(false, false), (false, true), (true, false), (true, true)] {
                    let (_a_root, a) = source(&av, shape![m as u64, k as u64], ls, 31).await;
                    let (_b_root, b) = source(&bv, shape![k as u64, n as u64], rs, 47).await;
                    let view = a.view().matmul(&b.view()).await.unwrap();
                    let large = (m, k, n) == (33, 129, 35);
                    if (m, k, n) == (2, 3, 2) || (large && $large_consumers && ls == rs) {
                        check(&view, &reference).await;
                    } else {
                        check_blocks(&view, &reference).await;
                    }
                    if large {
                        for [row, col] in [[0, 0], [0, 31], [0, 32], [31, 0], [32, 0], [32, 34]] {
                            assert!(common::numbers::same(
                                view.read_value(&[row, col]).await.unwrap(),
                                reference[row as usize * n + col as usize],
                            ));
                        }
                    }
                }
            }
        }
    };
}
parity!(f32_parity, f32, true);
parity!(f64_parity, f64, false);
parity!(u8_parity, u8, false);

#[tokio::test]
async fn sparse_zero_intermediates_and_copy_match_dense_semantics() {
    let (_a_root, a) = source(&[0.2f32, 0., 0., 0.], shape![2, 2], true, 2).await;
    let (_b_root, b) = source(&[0f32, 0., 0., 2.], shape![2, 2], true, 3).await;
    let view = a
        .view()
        .round()
        .await
        .unwrap()
        .matmul(&b.view())
        .await
        .unwrap();
    check(&view, &[0.; 4]).await;
    assert!(matches!(
        view.eq_scalar(0.).await,
        Err(fensor::Error::WouldDensify { .. })
    ));
    let (_dir_root, dir) = new_dir("matmul_zero_copy").await;
    let copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &view, view.layout(), 2)
        .await
        .unwrap();
    assert!(matches!(
        copy.view().eq_scalar(0.).await,
        Err(fensor::Error::WouldDensify { .. })
    ));
    assert_eq!(view.sum_all().await.unwrap(), 0.);
    let view = TensorExpression::new(view).unwrap().into_dense();
    let copy = TensorExpression::new(copy).unwrap().into_dense();
    check(&view.eq_scalar(0.).await.unwrap(), &[1; 4]).await;
    check(&copy.eq_scalar(0.).await.unwrap(), &[1; 4]).await;
    let (_empty_root, empty) = source(&[0f32; 4], shape![2, 2], true, 2).await;
    let empty = empty.view().matmul(&empty.view()).await.unwrap();
    assert!(matches!(
        empty.exp().await,
        Err(fensor::Error::WouldDensify { .. })
    ));
    check(
        &TensorExpression::new(empty)
            .unwrap()
            .into_dense()
            .exp()
            .await
            .unwrap(),
        &[1.; 4],
    )
    .await;
    let (_cancel_root, cancel) = source(&[1f32, -1.], shape![1, 2], true, 2).await;
    let (_ones_root, ones) = source(&[1f32, 1.], shape![2, 1], true, 1).await;
    let product = cancel.view().matmul(&ones.view()).await.unwrap();
    check(
        &TensorExpression::new(product)
            .unwrap()
            .into_dense()
            .exp()
            .await
            .unwrap(),
        &[1.],
    )
    .await;
}

#[tokio::test]
async fn exceptional_values_and_wrapping_arithmetic() {
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let (_a_root, a) = source(&[0f32], shape![1, 1], true, 1).await;
        let (_b_root, b) = source(&[value], shape![1, 1], true, 1).await;
        check(&a.view().matmul(&b.view()).await.unwrap(), &[f32::NAN]).await;
    }

    let (_a_root, a) = source(&[255u8; 129], shape![1, 129], false, 17).await;
    let (_b_root, b) = source(&[2u8; 129], shape![129, 1], false, 19).await;
    check(&a.view().matmul(&b.view()).await.unwrap(), &[254]).await;
    let (_a_root, a) = source(&[-0f64], shape![1, 1], false, 1).await;
    let (_b_root, b) = source(&[1f64], shape![1, 1], false, 1).await;
    let expected = Array::new(Buffer::from(vec![-0f64]), shape![1, 1])
        .unwrap()
        .matmul(Array::new(Buffer::from(vec![1f64]), shape![1, 1]).unwrap())
        .unwrap()
        .buffer()
        .unwrap()
        .to_slice()
        .unwrap()
        .into_vec();
    check(&a.view().matmul(&b.view()).await.unwrap(), &expected).await;
}

#[tokio::test]
async fn batches_nested_expressions_and_transforms() {
    let (_a_root, a) = source(
        &[1f32, 2., 3., 4., 5., 6., 7., 8.],
        shape![2, 2, 2],
        false,
        3,
    )
    .await;
    let (_identity_root, identity) = source(&[1f32, 0., 0., 1.], shape![1, 2, 2], false, 2).await;
    let b = identity.view().broadcast(shape![2, 2, 2]).unwrap();
    let product = a.view().matmul(&b).await.unwrap();
    check(&product, &[1., 2., 3., 4., 5., 6., 7., 8.]).await;
    check(
        &product.clone().transpose(Some(axes![0, 2, 1])).unwrap(),
        &[1., 3., 2., 4., 5., 7., 6., 8.],
    )
    .await;
    check(
        &product.clone().reshape(shape![8]).unwrap().flip(0).unwrap(),
        &[8., 7., 6., 5., 4., 3., 2., 1.],
    )
    .await;
    check(
        &product
            .clone()
            .slice(range![
                AxisRange::At(1),
                AxisRange::Of(vec![1, 0, 1]),
                AxisRange::In(0, 2, 1)
            ])
            .unwrap(),
        &[7., 8., 5., 6., 7., 8.],
    )
    .await;
    let selected = product
        .gt_scalar(0.)
        .await
        .unwrap()
        .cond(&product, &product)
        .await
        .unwrap();
    let nested = selected.matmul(&b).await.unwrap();
    check(
        &nested.sum(axes![0], false).await.unwrap(),
        &[6., 8., 10., 12.],
    )
    .await;
    let reduced = a.view().sum(axes![0], true).await.unwrap();
    check(
        &reduced.matmul(&identity.view()).await.unwrap(),
        &[6., 8., 10., 12.],
    )
    .await;
    let cast = TensorCast::<f64>::cast(&product).await.unwrap();
    check(
        &cast.matmul(&b.cast().await.unwrap()).await.unwrap(),
        &[1f64, 2., 3., 4., 5., 6., 7., 8.],
    )
    .await;
    assert!(a.view().matmul(&identity.view()).await.is_err());
    assert!(
        a.view()
            .reshape(shape![8])
            .unwrap()
            .matmul(&b)
            .await
            .is_err()
    );
    let (_wrong_root, wrong) = source(&[1f32; 6], shape![2, 3], false, 2).await;
    assert!(
        wrong
            .view()
            .matmul(&identity.view().reshape(shape![2, 2]).unwrap())
            .await
            .is_err()
    );
    check(
        &a.view()
            .add(&a.view())
            .await
            .unwrap()
            .matmul(&b)
            .await
            .unwrap(),
        &[2., 4., 6., 8., 10., 12., 14., 16.],
    )
    .await;
    check(
        &a.view()
            .transpose(Some(axes![0, 2, 1]))
            .unwrap()
            .matmul(&b)
            .await
            .unwrap(),
        &[1., 3., 2., 4., 5., 7., 6., 8.],
    )
    .await;
    check(
        &product
            .clone()
            .unsqueeze(axes![0])
            .unwrap()
            .squeeze(axes![0])
            .unwrap(),
        &[1., 2., 3., 4., 5., 6., 7., 8.],
    )
    .await;
    check(
        &product.clone().broadcast(shape![2, 2, 2, 2]).unwrap(),
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8.,
        ],
    )
    .await;
    a.write_value(&[0, 0, 0], 9.).await.unwrap();
    assert_eq!(product.read_value(&[0, 0, 0]).await.unwrap(), 9.);
}

fn check_dot_bound(left: &[f64], right: &[f64], actual: f64, u: f64) {
    use num_rational::BigRational;
    use num_traits::Signed;
    let mut exact = BigRational::from_integer(0.into());
    let mut scale = exact.clone();

    for (a, b) in left.iter().zip(right) {
        let term = BigRational::from_float(*a).unwrap() * BigRational::from_float(*b).unwrap();
        scale += term.abs();
        exact += term;
    }

    let ku =
        BigRational::from_float(u).unwrap() * BigRational::from_integer((8 * left.len()).into());
    let one = BigRational::from_integer(1.into());
    assert!(ku < one);
    let gamma = &ku / (one - &ku);
    let error = (BigRational::from_float(actual).unwrap() - exact).abs();
    assert!(error <= gamma * scale);
}

macro_rules! accuracy {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for k in [3, 127, 128, 129, 4097] {
                let av: Vec<$t> = (0..k)
                    .map(|i| [16., -16., 0.125, 0.0009765625, -0.0625][i % 5] as $t)
                    .collect();
                let bv: Vec<$t> = (0..k)
                    .map(|i| (1. + (i % 7) as f64 / 1024.) as $t)
                    .collect();
                let (_a_root, a) = source(&av, shape![1 as u64, k as u64], false, 31).await;
                let (_b_root, b) = source(&bv, shape![k as u64, 1 as u64], false, 17).await;
                let value = a
                    .view()
                    .matmul(&b.view())
                    .await
                    .unwrap()
                    .read_value(&[0, 0])
                    .await
                    .unwrap();
                check_dot_bound(
                    &av.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &bv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    value as f64,
                    <$t>::EPSILON as f64 / 2.,
                );
                if k == 4097 {
                    // Different request rectangles choose different contraction widths;
                    // compare each consumer to the exact bound, not bitwise equality.
                    let left = a.view().broadcast(shape![2 as u64, k as u64]).unwrap();
                    let right = b.view().broadcast(shape![k as u64, 33 as u64]).unwrap();
                    let product = left.matmul(&right).await.unwrap();
                    let ordinary: Vec<_> = product
                        .read_blocks()
                        .unwrap()
                        .try_collect::<Vec<_>>()
                        .await
                        .unwrap()
                        .into_iter()
                        .flatten()
                        .collect();
                    let (_dir_root, dir) = new_dir("adaptive_accuracy_copy").await;
                    let copy: Tensor<FsEntry, $t> =
                        Tensor::copy_from(dir, &product, product.layout(), 17)
                            .await
                            .unwrap();
                    for (coord, value) in [
                        (vec![0, 0], ordinary[0]),
                        (vec![1, 32], ordinary[65]),
                        (vec![0, 0], copy.read_value(&[0, 0]).await.unwrap()),
                        (vec![1, 32], copy.read_value(&[1, 32]).await.unwrap()),
                    ] {
                        check_dot_bound(
                            &av.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                            &bv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                            value as f64,
                            <$t>::EPSILON as f64 / 2.,
                        );
                        check_dot_bound(
                            &av.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                            &bv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                            product.read_value(&coord).await.unwrap() as f64,
                            <$t>::EPSILON as f64 / 2.,
                        );
                    }
                }
            }
        }
    };
}
accuracy!(f32_exact_dot, f32);
accuracy!(f64_exact_dot, f64);

#[tokio::test]
async fn repeated_concurrent_cancelled_and_cache_pressure_reads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("matmul_cache").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let a = Tensor::<FsEntry, u8>::create(
            cache.load(root.to_path_buf()).unwrap(),
            TensorSchema::new(u8::dtype(), shape![65, 2]).unwrap(),
            Layout::Dense,
            16,
        )
        .await
        .unwrap();

        for i in 0..65 {
            a.write_value(&[i, 0], 1).await.unwrap();
            a.write_value(&[i, 1], 2).await.unwrap();
        }

        let (_b_root, b) = source(&[1u8; 130], shape![2, 65], false, 17).await;
        let view = a.view().matmul(&b.view()).await.unwrap();
        let mut stream = view.read_blocks().unwrap();
        assert_eq!(stream.try_next().await.unwrap().unwrap().len(), 4096);
        drop(stream);
        let consume = || async {
            view.read_blocks()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
        };
        let (left, right) = futures::join!(consume(), consume());
        assert_eq!(left, right);
        assert!(left.into_iter().flatten().all(|v| v == 3));
        let (root, _) = new_dir("matmul_cache_output").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root.to_path_buf()).unwrap();
        let copy: Tensor<FsEntry, u8> = Tensor::copy_from(dir.clone(), &view, view.layout(), 32)
            .await
            .unwrap();
        copy.sync().await.unwrap();
        drop(copy);
        drop(dir);
        let copy = Tensor::<FsEntry, u8>::load(common::open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(copy.read_value(&[64, 64]).await.unwrap(), 3);
    })
    .await
    .expect("tiled product must progress under cache pressure");
}

#[tokio::test]
async fn coordinate_traversal_propagates_and_preserves_coverage() {
    let (_a_root, a) = source(&[1f32; 32], shape![32, 1], false, 7).await;
    let (_b_root, b) = source(&[2f32; 129], shape![1, 129], false, 31).await;
    let product = a.view().matmul(&b.view()).await.unwrap();

    async fn tiled<V: TensorRead<DType = f32>>(view: &V) {
        let mut blocks = view.read_coordinate_blocks().unwrap();
        let mut all = Vec::new();
        let mut first_tile = false;

        while let Some((coords, values)) = blocks.try_next().await.unwrap() {
            if coords.first() == Some(&vec![0, 0]) {
                assert_eq!(coords.len(), 4096);
                assert_eq!(coords[32], vec![1, 0]);
                assert_eq!(coords[1024], vec![0, 32]);
                first_tile = true;
            }
            assert!(coords.len() <= 4096);
            assert_eq!(coords.len(), values.len());
            assert!(values.iter().all(|v| *v == 2.));
            all.extend(coords);
        }
        assert!(first_tile);
        all.sort();
        assert_eq!(all, common::iter_coords(view.shape()).collect::<Vec<_>>());
    }
    tiled(&product).await;
    tiled(&product.round().await.unwrap()).await;
    let (_zero_root, zero) = source(&vec![0f32; 32 * 129], shape![32, 129], true, 31).await;
    tiled(&zero.view().add(&product).await.unwrap()).await;
    tiled(
        &product
            .gt_scalar(0.)
            .await
            .unwrap()
            .cond(&product, &zero.view())
            .await
            .unwrap(),
    )
    .await;

    tiled(&product.add(&zero.view()).await.unwrap()).await;
    let (_twos_root, twos) = source(&vec![2f32; 32 * 129], shape![32, 129], false, 31).await;
    let yes = twos.view().gt_scalar(0.).await.unwrap();
    let no = twos.view().lt_scalar(0.).await.unwrap();
    // Only the condition, then branch, or else branch supplies tiled requests.
    tiled(
        &product
            .gt_scalar(0.)
            .await
            .unwrap()
            .cond(&twos.view(), &zero.view())
            .await
            .unwrap(),
    )
    .await;
    tiled(&yes.cond(&product, &zero.view()).await.unwrap()).await;
    tiled(&no.cond(&zero.view(), &product).await.unwrap()).await;

    async fn covered<V: TensorRead<DType = f32>>(view: &V) {
        let mut seen = Vec::new();
        let mut stream = view.read_coordinate_blocks().unwrap();

        while let Some((coords, values)) = stream.try_next().await.unwrap() {
            assert_eq!(coords.len(), values.len());
            assert!(coords.len() <= 4096);

            for (coord, value) in coords.into_iter().zip(values) {
                seen.push(coord.clone());
                assert_eq!(value, view.read_value(&coord).await.unwrap());
            }
        }
        seen.sort();
        assert_eq!(seen, common::iter_coords(view.shape()).collect::<Vec<_>>());
    }
    covered(&twos.view()).await;
    // Even an empty-axis reduction is a new traversal boundary.
    covered(&product.sum(axes![], false).await.unwrap()).await;
    covered(&product.clone().reshape(shape![32 * 129]).unwrap()).await;

    let transformed = product.transpose(None).unwrap().flip(1).unwrap();
    let mut seen = Vec::new();
    let mut blocks = transformed.read_coordinate_blocks().unwrap();

    while let Some((coords, values)) = blocks.try_next().await.unwrap() {
        for (coord, value) in coords.iter().zip(values) {
            assert_eq!(transformed.read_value(coord).await.unwrap(), value);
            seen.push(coord.clone());
        }
    }
    seen.sort();
    assert_eq!(
        seen,
        common::iter_coords(transformed.shape()).collect::<Vec<_>>()
    );
}
