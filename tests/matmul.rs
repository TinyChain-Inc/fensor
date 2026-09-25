//! Filesystem-backed tiled matrix multiplication and source support.

use fensor::{
    AxisRange, Layout, Shape, Tensor, TensorCast, TensorCompareScalar, TensorElement,
    TensorFileEntry, TensorGeometry, TensorMatMul, TensorMath, TensorRead, TensorReduce,
    TensorReduceAll, TensorSchema, TensorTransform, TensorUnary, TensorWhere, TensorWrite,
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
) -> Tensor<FsEntry, T>
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
    .1
}

trait Matches: TensorElement {
    fn matches(self, other: Self) -> bool;
}

impl Matches for u8 {
    fn matches(self, other: Self) -> bool {
        self == other
    }
}

impl Matches for f32 {
    fn matches(self, other: Self) -> bool {
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}

impl Matches for f64 {
    fn matches(self, other: Self) -> bool {
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}

async fn check_blocks<V: TensorRead>(view: &V, expected: &[V::DType])
where
    V::DType: Matches,
{
    common::fixture::blocks(view, expected, Matches::matches).await;
}

async fn check<V: TensorRead>(view: &V, expected: &[V::DType])
where
    V::DType: Matches,
    FsEntry: TensorFileEntry<V::DType>,
{
    common::fixture::consumers(view, expected, Matches::matches, |a, b| {
        a.matches(b) || (a == V::DType::ZERO && b == V::DType::ZERO)
    })
    .await;
}

macro_rules! parity {
    ($name:ident, $t:ty) => {
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
                    let a = source(&av, shape![m as u64, k as u64], ls, 31).await;
                    let b = source(&bv, shape![k as u64, n as u64], rs, 47).await;
                    let view = a.view().matmul(&b.view()).await.unwrap();
                    // Full persistence/consumer parity belongs to these two shapes;
                    // every shape still checks the numerical block-stream path.
                    if [(2, 3, 2), (33, 129, 35)].contains(&(m, k, n)) {
                        check(&view, &reference).await;
                    } else {
                        check_blocks(&view, &reference).await;
                    }
                }
            }
        }
    };
}
parity!(f32_parity, f32);
parity!(f64_parity, f64);
parity!(u8_parity, u8);

#[tokio::test]
async fn sparse_union_support_and_copy_boundaries() {
    let a = source(&[0.2f32, 0., 0., 0.], shape![2, 2], true, 2).await;
    let b = source(&[0f32, 0., 0., 2.], shape![2, 2], true, 3).await;
    let view = a
        .view()
        .round()
        .await
        .unwrap()
        .matmul(&b.view())
        .await
        .unwrap();
    check(&view, &[0., 0., 0., 0.]).await;
    check(&view.eq_scalar(0.).await.unwrap(), &[1, 1, 0, 1]).await;
    let (_, dir) = new_dir("matmul_support_boundary").await;
    let copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &view, 2).await.unwrap();
    check(&copy.view().eq_scalar(0.).await.unwrap(), &[0, 0, 0, 0]).await;
    assert_eq!(view.sum_all().await.unwrap(), 0.);
    let empty = source(&[0f32; 4], shape![2, 2], true, 2).await;
    check(
        &empty
            .view()
            .matmul(&empty.view())
            .await
            .unwrap()
            .exp()
            .await
            .unwrap(),
        &[0.; 4],
    )
    .await;
    let cancel = source(&[1f32, -1.], shape![1, 2], true, 2).await;
    let ones = source(&[1f32, 1.], shape![2, 1], true, 1).await;
    check(
        &cancel
            .view()
            .matmul(&ones.view())
            .await
            .unwrap()
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
        let a = source(&[0f32], shape![1, 1], true, 1).await;
        let b = source(&[value], shape![1, 1], true, 1).await;
        check(&a.view().matmul(&b.view()).await.unwrap(), &[f32::NAN]).await;
    }

    let a = source(&[255u8; 129], shape![1, 129], false, 17).await;
    let b = source(&[2u8; 129], shape![129, 1], false, 19).await;
    check(&a.view().matmul(&b.view()).await.unwrap(), &[254]).await;
    let a = source(&[-0f64], shape![1, 1], false, 1).await;
    let b = source(&[1f64], shape![1, 1], false, 1).await;
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
    let a = source(
        &[1f32, 2., 3., 4., 5., 6., 7., 8.],
        shape![2, 2, 2],
        false,
        3,
    )
    .await;
    let identity = source(&[1f32, 0., 0., 1.], shape![1, 2, 2], false, 2).await;
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
    let cast = product.cast().await.unwrap();
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
    let wrong = source(&[1f32; 6], shape![2, 3], false, 2).await;
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
                let a = source(&av, shape![1 as u64, k as u64], false, 31).await;
                let b = source(&bv, shape![k as u64, 1 as u64], false, 17).await;
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
                    let (_, dir) = new_dir("adaptive_accuracy_copy").await;
                    let copy: Tensor<FsEntry, $t> =
                        Tensor::copy_from(dir, &product, 17).await.unwrap();
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
async fn huge_selected_outputs_and_corruption_boundaries() {
    use fensor::TensorSparseIndex;
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (_, dir) = new_dir("huge_matmul").await;
        let a = Tensor::<FsEntry, f32>::create(
            dir.clone(),
            TensorSchema::new(f32::dtype(), shape![1_000_000_000, 2]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();
        a.write_value(&[999_999_999, 1], 2.).await.unwrap();
        a.write_value(&[0, 0], 1.).await.unwrap();
        let id = a.lookup_block_id(&[0, 0]).await.unwrap().unwrap();
        dir.read()
            .await
            .get_dir("blocks")
            .unwrap()
            .write()
            .await
            .delete(&id.to_string())
            .await;
        let b = source(&[1f32, 2., 3., 4.], shape![2, 2], true, 2).await;
        let view = a.view().matmul(&b.view()).await.unwrap();
        let entries: Vec<_> = view
            .read_sparse_elements_in_order(
                range![AxisRange::At(999_999_999), AxisRange::In(0, 2, 1)],
                axes![0, 1],
            )
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(
            entries,
            vec![(vec![999_999_999, 0], 6.), (vec![999_999_999, 1], 8.)]
        );
        assert!(view.read_value(&[0, 0]).await.is_err());
        assert!(
            view.read_sparse_elements_in_order(
                range![AxisRange::At(1_000_000_000), AxisRange::At(0)],
                axes![0, 1]
            )
            .await
            .is_err()
        );
        // The same corrupt source on the right is only read for selected columns.
        let right = a.view().transpose(None).unwrap();
        let left = b.view().transpose(None).unwrap();
        let reversed = left.matmul(&right).await.unwrap();
        assert_eq!(reversed.read_value(&[0, 999_999_999]).await.unwrap(), 6.);
        assert!(reversed.read_value(&[0, 0]).await.is_err());
    })
    .await
    .expect("selected product must not traverse unrelated output tiles");
}

#[tokio::test]
async fn repeated_concurrent_cancelled_and_cache_pressure_reads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("matmul_cache").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let a = Tensor::<FsEntry, u8>::create(
            cache.load(root).unwrap(),
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

        let b = source(&[1u8; 130], shape![2, 65], false, 17).await;
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
        let dir = cache.load(root.clone()).unwrap();
        let copy: Tensor<FsEntry, u8> = Tensor::copy_from(dir.clone(), &view, 32).await.unwrap();
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
    let a = source(&[1f32; 32], shape![32, 1], false, 7).await;
    let b = source(&[2f32; 129], shape![1, 129], false, 31).await;
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
    let zero = source(&vec![0f32; 32 * 129], shape![32, 129], true, 31).await;
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
    let twos = source(&vec![2f32; 32 * 129], shape![32, 129], false, 31).await;
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
