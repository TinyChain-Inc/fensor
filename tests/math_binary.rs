//! Filesystem-backed binary expression parity and source-support regressions.
use fensor::{
    AxisRange, Layout, Tensor, TensorArray, TensorBoolean, TensorBooleanScalar, TensorCast,
    TensorCompare, TensorCompareScalar, TensorElement, TensorFileEntry, TensorGeometry, TensorMath,
    TensorMathScalar, TensorNumeric, TensorRead, TensorSchema, TensorTransform, TensorUnary,
    TensorUnaryBoolean, TensorWhere, TensorWrite,
};

use futures::{StreamExt, TryStreamExt};
use ha_ndarray::{
    Array, ArrayAccess, Buffer, NDArrayBoolean, NDArrayBooleanScalar, NDArrayCompare,
    NDArrayCompareScalar, NDArrayMath, NDArrayMathScalar, NDArrayRead, NDArrayWhere, Number, axes,
    range, shape,
};

use number_general::DType;

use common::{FsEntry, new_dir};

mod common;

trait TestElement: TensorElement {
    fn matches(self, other: Self) -> bool;
}

impl TestElement for f32 {
    fn matches(self, other: Self) -> bool {
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}

impl TestElement for f64 {
    fn matches(self, other: Self) -> bool {
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}

impl TestElement for u8 {
    fn matches(self, other: Self) -> bool {
        self == other
    }
}

async fn source<T: TensorElement>(
    values: &[T],
    layout: Layout,
    capacity: usize,
) -> Tensor<FsEntry, T>
where
    FsEntry: TensorFileEntry<T>,
{
    let (_, dir) = new_dir("binary_source").await;
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(T::dtype(), shape![values.len() as u64]).unwrap(),
        layout,
        capacity,
    )
    .await
    .unwrap();

    for (i, value) in values.iter().enumerate() {
        tensor.write_value(&[i as u64], *value).await.unwrap();
    }

    tensor
}

async fn check<V>(view: V, expected: Vec<V::DType>)
where
    V: TensorRead,
    V::DType: TestElement,
    FsEntry: TensorFileEntry<V::DType>,
{
    let blocks: Vec<Vec<_>> = view.read_blocks().unwrap().try_collect().await.unwrap();

    assert!(blocks.iter().all(|b| b.len() <= 4096));
    let actual: Vec<_> = blocks.into_iter().flatten().collect();

    assert_eq!(actual.len(), expected.len());
    let (root, dir) = new_dir("binary_copy").await;
    let copy: Tensor<FsEntry, V::DType> = Tensor::copy_from(dir.clone(), &view, 3).await.unwrap();
    copy.sync().await.unwrap();
    drop(copy);
    drop(dir);
    let copy = Tensor::<FsEntry, V::DType>::load(common::open_dir(&root).unwrap())
        .await
        .unwrap();

    for ((i, (&a, &e)), coord) in actual
        .iter()
        .zip(&expected)
        .enumerate()
        .zip(common::iter_coords(view.shape()))
    {
        assert!(a.matches(e), "stream {i}: {a:?} != {e:?}");
        assert!(
            view.read_value(&coord).await.unwrap().matches(e),
            "point {i}"
        );
        let copied = copy.read_value(&coord).await.unwrap();

        assert!(
            copied.matches(e)
                || (matches!(view.layout(), Layout::Sparse { .. })
                    && copied == V::DType::ZERO
                    && e == V::DType::ZERO),
            "copy {i}"
        );
    }

    if matches!(view.layout(), Layout::Sparse { .. }) {
        let entries: Vec<_> = view
            .read_sparse_elements_in_order(
                view.shape()
                    .iter()
                    .map(|d| AxisRange::In(0, *d, 1))
                    .collect(),
                (0..view.ndim()).collect(),
            )
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let wanted: Vec<_> = expected
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != V::DType::ZERO)
            .collect();

        assert_eq!(entries.len(), wanted.len());

        for ((coord, value), (i, expected)) in entries.into_iter().zip(wanted) {
            assert_eq!(coord, vec![i as u64]);
            assert!(value.matches(*expected));
        }
    }
}

macro_rules! matrix {
    ($name:ident, $t:ty, $left:expr, $right:expr, [$($op:ident),+]) => {
        #[tokio::test]
        async fn $name() {
            let left: Vec<$t> = $left;
            let right: Vec<$t> = $right;

            for ls in [false, true] {
                for rs in [false, true] {
                    let layout = |s| if s {
                        Layout::Sparse { axis: Some(0) }
                    } else {
                        Layout::Dense
                    };

                    let a = source(&left, layout(ls), 3).await;
                    let b = source(&right, layout(rs), 2).await;

                    $(
                        let canonical = |values: &Vec<$t>, sparse: bool| values
                            .iter()
                            .map(|v| if sparse && *v == 0 as $t {
                                0 as $t
                            } else {
                                *v
                            })
                            .collect::<Vec<_>>();

                        let reference: ArrayAccess<'_, _> = ArrayAccess::from(
                            Array::new(Buffer::from(canonical(&left, ls)), shape![left.len()])
                                .unwrap()
                                .$op(
                                    Array::new(
                                        Buffer::from(canonical(&right, rs)),
                                        shape![right.len()]
                                    )
                                    .unwrap()
                                )
                                .unwrap()
                        );
                        let mut expected = reference
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec();

                        if ls && rs {
                            for (i, value) in expected.iter_mut().enumerate() {
                                if left[i] == 0 as $t && right[i] == 0 as $t {
                                    *value = Default::default();
                                }
                            }
                        }

                        let expression = a.view().$op(&b.view()).await.unwrap();

                        assert_eq!(
                            expression.layout(),
                            if ls && rs {
                                Layout::Sparse { axis: None }
                            } else {
                                Layout::Dense
                            }
                        );
                        check(expression, expected).await;
                    )+
                }
            }
        }
    };
}

matrix!(
    f32_matrix,
    f32,
    vec![
        0.,
        -0.,
        5.5,
        -5.5,
        0.,
        2.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY
    ],
    vec![0., 2., 2., -2., 3., 0., 1., 2., 0.],
    [add, sub, mul, div, pow, log, rem]
);
matrix!(
    f64_matrix,
    f64,
    vec![
        0.,
        -0.,
        5.5,
        -5.5,
        0.,
        2.,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY
    ],
    vec![0., 2., 2., -2., 3., 0., 1., 2., 0.],
    [add, sub, mul, div, pow, log, rem]
);
matrix!(
    u8_matrix,
    u8,
    vec![0, 0, 255, 127, 2, 3, 0],
    vec![0, 1, 255, 255, 8, 6, 255],
    [add, sub, mul, div, pow, rem]
);

#[tokio::test]
async fn integer_expected_results() {
    let a = source(&[255u8, 0, 2, 3], Layout::Dense, 2).await;
    let b = source(&[255u8, 0, 8, 6], Layout::Dense, 3).await;
    check(a.view().add(&b.view()).await.unwrap(), vec![254, 0, 10, 9]).await;
    check(a.view().sub(&b.view()).await.unwrap(), vec![0, 0, 250, 253]).await;
    check(a.view().mul(&b.view()).await.unwrap(), vec![1, 0, 16, 18]).await;
    check(a.view().pow(&b.view()).await.unwrap(), vec![255, 1, 0, 217]).await;
    check(a.view().div(&b.view()).await.unwrap(), vec![1, 0, 0, 0]).await;
    check(a.view().rem(&b.view()).await.unwrap(), vec![0, 0, 2, 3]).await;
}

#[tokio::test]
async fn sparse_support_survives_zero_intermediates_but_not_copying() {
    let sparse = Layout::Sparse { axis: None };
    let a = source(&[0.2f32, 0., 2., 0.], sparse, 2).await;
    let b = source(&[0f32, 3., 2., 0.], sparse, 3).await;
    let zero = a.view().sub(&a.view()).await.unwrap();
    check(zero.exp().await.unwrap(), vec![1., 0., 1., 0.]).await;
    check(
        zero.div(&zero).await.unwrap(),
        vec![f32::NAN, 0., f32::NAN, 0.],
    )
    .await;
    check(
        a.view()
            .round()
            .await
            .unwrap()
            .exp()
            .await
            .unwrap()
            .add(&b.view())
            .await
            .unwrap(),
        vec![1., 3., 2f32.exp() + 2., 0.],
    )
    .await;
    check(
        a.view().sub(&b.view()).await.unwrap().exp().await.unwrap(),
        vec![0.2f32.exp(), (-3f32).exp(), 1., 0.],
    )
    .await;
    let (_, dir) = new_dir("binary_support_boundary").await;
    let copied: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &zero, 2).await.unwrap();
    check(copied.view().exp().await.unwrap(), vec![0.; 4]).await;
    let empty = source(&[0f32; 4], sparse, 2).await;
    check(
        empty
            .view()
            .div(&empty.view())
            .await
            .unwrap()
            .exp()
            .await
            .unwrap(),
        vec![0.; 4],
    )
    .await;
}

#[tokio::test]
async fn nested_transforms_casts_predicates_and_explicit_broadcast() {
    let a = source(&[1f32, 2., 3., 4.], Layout::Dense, 2).await;
    let b = source(&[2f32, 0., 1., 2.], Layout::Dense, 3).await;
    let expression = a
        .view()
        .flip(0)
        .unwrap()
        .sub(&b.view())
        .await
        .unwrap()
        .flip(0)
        .unwrap()
        .cast()
        .await
        .unwrap()
        .is_nan()
        .await
        .unwrap()
        .not()
        .await
        .unwrap();
    check(expression, vec![1u8; 4]).await;
    let transformed = a
        .view()
        .reshape(shape![2, 2])
        .unwrap()
        .add(&b.view().reshape(shape![2, 2]).unwrap())
        .await
        .unwrap()
        .transpose(Some(axes![1, 0]))
        .unwrap();
    check(transformed, vec![3., 4., 2., 6.]).await;
    let scalar = source(&[2f32], Layout::Dense, 1).await;

    assert!(a.view().add(&scalar.view()).await.is_err());
    check(
        a.view()
            .mul(&scalar.view().broadcast(shape![4]).unwrap())
            .await
            .unwrap(),
        vec![2., 4., 6., 8.],
    )
    .await;
}

#[tokio::test]
async fn bounded_batches_independent_consumption_and_live_sources() {
    let a = source(&vec![0.2f32; 4101], Layout::Dense, 101).await;
    let b = source(&vec![0.2f32; 4101], Layout::Dense, 127).await;
    let expression = a.view().sub(&b.view()).await.unwrap().exp().await.unwrap();
    let mut dropped = expression.read_blocks().unwrap();

    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let (first, second) = futures::join!(
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
        expression.read_blocks().unwrap().try_collect::<Vec<_>>()
    );
    let first = first.unwrap();

    assert_eq!(first, second.unwrap());
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4096, 5]
    );

    assert!(first.iter().flatten().all(|v| *v == 1.));
    a.write_value(&[4100], 1.2).await.unwrap();

    assert_eq!(expression.read_value(&[4100]).await.unwrap(), 1f32.exp());
    assert!(expression.read_value(&[4101]).await.is_err());
}

#[tokio::test]
async fn far_end_sparse_range_is_bounded_and_ordered() {
    let (_, dir) = new_dir("binary_far_left").await;
    let a = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), shape![1_000_000_000]).unwrap(),
        Layout::Sparse { axis: None },
        100,
    )
    .await
    .unwrap();
    let (_, dir) = new_dir("binary_far_right").await;
    let b =
        Tensor::<FsEntry, f32>::create(dir, a.schema().clone(), Layout::Sparse { axis: None }, 99)
            .await
            .unwrap();
    a.write_value(&[999_999_998], 2.).await.unwrap();
    b.write_value(&[999_999_999], 3.).await.unwrap();
    let expression = a.view().add(&b.view()).await.unwrap().exp().await.unwrap();
    let run = async {
        expression
            .read_sparse_elements_in_order(
                range![AxisRange::Of(vec![999_999_999, 999_999_998, 999_999_999])],
                axes![0],
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    };

    let result = tokio::time::timeout(std::time::Duration::from_secs(15), run)
        .await
        .unwrap();

    assert_eq!(
        result,
        vec![
            (vec![999_999_998], 2f32.exp()),
            (vec![999_999_999], 3f32.exp())
        ]
    );

    assert!(
        expression
            .read_sparse_elements_in_order(
                range![AxisRange::In(999_999_999, 1_000_000_001, 1)],
                axes![0]
            )
            .await
            .is_err()
    );

    assert!(
        expression
            .read_sparse_elements_in_order(range![], axes![1])
            .await
            .is_err()
    );

    assert!(
        expression
            .read_sparse_elements_in_order(range![AxisRange::In(5, 5, 1)], axes![0])
            .await
            .unwrap()
            .next()
            .await
            .is_none()
    );
}

#[tokio::test]
async fn selected_range_propagates_corruption_only_when_read() {
    use fensor::TensorSparseIndex;

    let (_, dir) = new_dir("binary_corruption").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(f32::dtype(), shape![8]).unwrap(),
        Layout::Sparse { axis: None },
        2,
    )
    .await
    .unwrap();
    tensor.write_value(&[0], 0.2).await.unwrap();
    tensor.write_value(&[7], 0.2).await.unwrap();
    let id = tensor.lookup_block_id(&[7, 3]).await.unwrap().unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete(&id.to_string()).await;
    let empty = source(&[0f32; 8], Layout::Sparse { axis: None }, 3).await;

    for reversed in [false, true] {
        let (left, right) = if reversed {
            (&empty, &tensor)
        } else {
            (&tensor, &empty)
        };

        let expression = left
            .view()
            .add(&right.view())
            .await
            .unwrap()
            .exp()
            .await
            .unwrap();
        let values: Vec<_> = expression
            .read_sparse_elements_in_order(range![AxisRange::At(0)], axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(values, vec![(vec![0], 0.2f32.exp())]);
        let mut stream = expression
            .read_sparse_elements_in_order(range![AxisRange::At(7)], axes![0])
            .await
            .unwrap();

        assert!(stream.try_next().await.is_err());
        assert!(expression.read_value(&[7]).await.is_err());
        assert!(
            expression
                .read_blocks()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn binary_copy_under_cache_pressure_reloads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("binary_cache").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root).unwrap();
        let a = Tensor::<FsEntry, u8>::create(
            dir.clone(),
            TensorSchema::new(u8::dtype(), shape![1024]).unwrap(),
            Layout::Dense,
            32,
        )
        .await
        .unwrap();
        a.write_value(&[1023], 255).await.unwrap();
        let b = source(&vec![1u8; 1024], Layout::Dense, 31).await;
        let (out_root, _) = new_dir("binary_cache_copy").await;
        let out_cache =
            freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let out_dir = out_cache.load(out_root.clone()).unwrap();
        let expression = a.view().add(&b.view()).await.unwrap();
        let output: Tensor<FsEntry, u8> = Tensor::copy_from(out_dir.clone(), &expression, 32)
            .await
            .unwrap();

        assert_eq!(output.read_value(&[1023]).await.unwrap(), 0);
        output.sync().await.unwrap();
        drop(output);
        drop(out_dir);
        let reloaded = Tensor::<FsEntry, u8>::load(common::open_dir(&out_root).unwrap())
            .await
            .unwrap();

        assert_eq!(reloaded.read_value(&[0]).await.unwrap(), 1);
        assert_eq!(reloaded.read_value(&[1023]).await.unwrap(), 0);
    })
    .await
    .expect("cache pressure must make progress");
}

#[tokio::test]
async fn sparse_batch_boundary_masks_children_and_retains_support() {
    let mut values = vec![0f32; 4101];
    values[0] = 0.2;
    values[4095] = 0.2;
    values[4096] = 0.2;
    values[4100] = 0.2;
    let a = source(&values, Layout::Sparse { axis: None }, 100).await;
    let b = source(&vec![0f32; 4101], Layout::Sparse { axis: None }, 127).await;
    let expression = a
        .view()
        .round()
        .await
        .unwrap()
        .add(&b.view())
        .await
        .unwrap()
        .exp()
        .await
        .unwrap();
    let read = || async {
        expression
            .read_sparse_elements_in_order(range![AxisRange::In(0, 4101, 1)], axes![0])
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    };

    let mut dropped = expression.read_blocks().unwrap();

    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let (first, second) = futures::join!(read(), read());

    assert_eq!(first, second);
    assert_eq!(
        first,
        vec![
            (vec![0], 1.),
            (vec![4095], 1.),
            (vec![4096], 1.),
            (vec![4100], 1.)
        ]
    );

    assert_eq!(read().await, first);
    let selected: Vec<_> = expression
        .read_sparse_elements_in_order(range![AxisRange::In(4094, 4101, 2)], axes![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    assert_eq!(selected, vec![(vec![4096], 1.), (vec![4100], 1.)]);
}

#[tokio::test]
async fn binary_tree_transforms_preserve_operand_order_and_scalar_limits() {
    let a = source(&[1f64, 2., 3., 4.], Layout::Dense, 2).await;
    let b = source(&[2f64, 3., 4., 5.], Layout::Dense, 3).await;
    let difference = a.view().sub(&b.view()).await.unwrap();
    let sum = a.view().add(&b.view()).await.unwrap();
    let tree = sum
        .div(&difference)
        .await
        .unwrap()
        .unsqueeze(axes![0])
        .unwrap()
        .slice(range![AxisRange::At(0), AxisRange::Of(vec![3, 1, 3])])
        .unwrap()
        .unsqueeze(axes![0])
        .unwrap()
        .squeeze(axes![0])
        .unwrap();
    check(tree, vec![-9., -5., -9.]).await;
    let scalar = sum.slice(range![AxisRange::At(0)]).unwrap();

    assert_eq!(scalar.read_value(&[]).await.unwrap(), 3.);
    assert!(scalar.read_blocks().is_err());
    let (_, dir) = new_dir("binary_scalar_copy").await;

    assert!(
        Tensor::<FsEntry, f64>::copy_from(dir, &scalar, 2)
            .await
            .is_err()
    );
}

fn backend<T: TensorElement>(values: &[T], sparse: bool) -> ArrayAccess<'static, T> {
    let values: Vec<_> = values
        .iter()
        .map(|v| if sparse && *v == T::ZERO { T::ZERO } else { *v })
        .collect();
    let len = values.len();

    ArrayAccess::from(Array::new(Buffer::from(values), shape![len]).unwrap())
}

macro_rules! scalar_cases {
    ($name:ident, $t:ty, $values:expr, $scalars:expr, [$($op:ident),+]) => {
        #[tokio::test]
        async fn $name() {
            let values: Vec<$t> = $values;

            for sparse in [false, true] {
                let layout = if sparse { Layout::Sparse { axis: None } } else { Layout::Dense };
                let tensor = source(&values, layout, 3).await;

                for scalar in $scalars {
                    $(
                        let reference = backend(&values, sparse).$op(scalar).unwrap();
                        let mut expected = reference.buffer().unwrap().to_slice().unwrap().into_vec();

                        if sparse {
                            for (input, output) in values.iter().zip(&mut expected) {
                                if *input == 0 as $t { *output = Default::default(); }
                            }
                        }

                        check(tensor.view().$op(scalar).await.unwrap(), expected).await;
                    )+
                }
            }
        }
    };
}
scalar_cases!(
    f32_scalar_math,
    f32,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY
    ],
    [0., -1., 2.],
    [
        add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, rem_scalar, log_scalar
    ]
);
scalar_cases!(
    f32_scalar_predicates,
    f32,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY
    ],
    [0., -0., 1., f32::NAN, f32::INFINITY],
    [
        eq_scalar, ne_scalar, gt_scalar, ge_scalar, lt_scalar, le_scalar, and_scalar, or_scalar,
        xor_scalar
    ]
);
scalar_cases!(
    f64_scalar_math,
    f64,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY
    ],
    [0., -1., 2.],
    [
        add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, rem_scalar, log_scalar
    ]
);
scalar_cases!(
    f64_scalar_predicates,
    f64,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY
    ],
    [0., -0., 1., f64::NAN, f64::INFINITY],
    [
        eq_scalar, ne_scalar, gt_scalar, ge_scalar, lt_scalar, le_scalar, and_scalar, or_scalar,
        xor_scalar
    ]
);
scalar_cases!(
    u8_scalar_math,
    u8,
    vec![0, 1, 2, 127, 255],
    [0, 1, 2, 127, 255],
    [
        add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar, rem_scalar
    ]
);
scalar_cases!(
    u8_scalar_predicates,
    u8,
    vec![0, 1, 2, 127, 255],
    [0, 1, 127, 255],
    [
        eq_scalar, ne_scalar, gt_scalar, ge_scalar, lt_scalar, le_scalar, and_scalar, or_scalar,
        xor_scalar
    ]
);
matrix!(
    f32_comparisons_and_booleans,
    f32,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY
    ],
    vec![
        -0.,
        1.,
        2.,
        0.,
        1.,
        f32::NAN,
        0.,
        f32::INFINITY,
        f32::NEG_INFINITY
    ],
    [eq, ne, gt, ge, lt, le, and, or, xor]
);
matrix!(
    f64_comparisons_and_booleans,
    f64,
    vec![
        0.,
        -0.,
        -2.,
        0.2,
        1.,
        2.,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY
    ],
    vec![
        -0.,
        1.,
        2.,
        0.,
        1.,
        f64::NAN,
        0.,
        f64::INFINITY,
        f64::NEG_INFINITY
    ],
    [eq, ne, gt, ge, lt, le, and, or, xor]
);
matrix!(
    u8_comparisons_and_booleans,
    u8,
    vec![0, 0, 1, 127, 255],
    vec![0, 255, 1, 0, 127],
    [eq, ne, gt, ge, lt, le, and, or, xor]
);

#[tokio::test]
async fn scalar_independent_arithmetic_and_truth_tables() {
    let tensor = source(&[0u8, 1, 127, 255], Layout::Dense, 2).await;
    check(
        tensor.view().add_scalar(1).await.unwrap(),
        vec![1, 2, 128, 0],
    )
    .await;
    check(
        tensor.view().sub_scalar(1).await.unwrap(),
        vec![255, 0, 126, 254],
    )
    .await;
    check(
        tensor.view().mul_scalar(2).await.unwrap(),
        vec![0, 2, 254, 254],
    )
    .await;
    check(tensor.view().pow_scalar(2).await.unwrap(), vec![0, 1, 1, 1]).await;
    check(tensor.view().div_scalar(0).await.unwrap(), vec![0; 4]).await;
    check(tensor.view().rem_scalar(0).await.unwrap(), vec![0; 4]).await;
    check(
        tensor.view().and_scalar(127).await.unwrap(),
        vec![0, 1, 1, 1],
    )
    .await;
    check(
        tensor.view().xor_scalar(255).await.unwrap(),
        vec![1, 0, 0, 0],
    )
    .await;
    check(
        tensor.view().gt_scalar(127).await.unwrap(),
        vec![0, 0, 0, 1],
    )
    .await;

    let floats = source(
        &[0f32, -0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        Layout::Dense,
        2,
    )
    .await;
    check(
        floats.view().eq_scalar(0.).await.unwrap(),
        vec![1, 1, 0, 0, 0],
    )
    .await;
    check(
        floats.view().ne_scalar(0.).await.unwrap(),
        vec![0, 0, 1, 1, 1],
    )
    .await;
    check(
        floats.view().and_scalar(1.).await.unwrap(),
        vec![0, 0, 1, 1, 1],
    )
    .await;
}

macro_rules! selection_cases {
    ($name:ident, $t:ty, $left:expr, $right:expr) => {
        #[tokio::test]
        async fn $name() {
            let left: Vec<$t> = $left;
            let right: Vec<$t> = $right;
            let conditions = [0u8, 1, 127, 255, 0, 0];

            for cs in [false, true] {
                for ls in [false, true] {
                    for rs in [false, true] {
                        let layout = |s| {
                            if s {
                                Layout::Sparse { axis: None }
                            } else {
                                Layout::Dense
                            }
                        };
                        let c = source(&conditions, layout(cs), 2).await;
                        let a = source(&left, layout(ls), 3).await;
                        let b = source(&right, layout(rs), 4).await;
                        let expression = c.view().cond(&a.view(), &b.view()).await.unwrap();
                        let expected: Vec<$t> = conditions
                            .iter()
                            .enumerate()
                            .map(|(i, c)| {
                                let (v, sparse) = if *c == 0 {
                                    (right[i], rs)
                                } else {
                                    (left[i], ls)
                                };
                                if sparse && v == 0 as $t { 0 as $t } else { v }
                            })
                            .collect();

                        let reference = backend(&conditions, cs)
                            .cond(backend(&left, ls), backend(&right, rs))
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec();
                        assert!(reference.iter().zip(&expected).all(|(a, b)| a.matches(*b)));
                        assert_eq!(expression.layout(), layout(cs && ls && rs));
                        check(expression, expected).await;
                    }
                }
            }
        }
    };
}
selection_cases!(
    f32_selection,
    f32,
    vec![0., -0., f32::NAN, f32::INFINITY, 2., 0.],
    vec![-0., f32::NAN, 1., 0., f32::NEG_INFINITY, 0.]
);
selection_cases!(
    f64_selection,
    f64,
    vec![0., -0., f64::NAN, f64::INFINITY, 2., 0.],
    vec![-0., f64::NAN, 1., 0., f64::NEG_INFINITY, 0.]
);
selection_cases!(
    u8_selection,
    u8,
    vec![0, 1, 127, 255, 2, 0],
    vec![0, 255, 1, 0, 127, 0]
);

#[tokio::test]
async fn sparse_scalar_comparison_and_selection_support() {
    let layout = Layout::Sparse { axis: None };
    let a = source(&[0f32, 2., 0., 0.], layout, 2).await;
    let b = source(&[0f32, 0., 3., 0.], layout, 3).await;
    let c = source(&[0u8, 0, 0, 1], layout, 2).await;

    check(a.view().add_scalar(1.).await.unwrap(), vec![0., 3., 0., 0.]).await;
    check(a.view().eq_scalar(0.).await.unwrap(), vec![0; 4]).await;
    check(a.view().eq(&b.view()).await.unwrap(), vec![0; 4]).await;
    check(
        a.view().eq(&b.view()).await.unwrap().not().await.unwrap(),
        vec![0, 1, 1, 0],
    )
    .await;

    let selected = c.view().cond(&a.view(), &b.view()).await.unwrap();
    check(selected.clone(), vec![0., 0., 3., 0.]).await;
    // Retain support from an unselected branch and from the condition itself.
    check(selected.eq_scalar(0.).await.unwrap(), vec![0, 1, 0, 1]).await;
    let (_, dir) = new_dir("selected_support_boundary").await;
    let copy: Tensor<FsEntry, f32> = Tensor::copy_from(dir, &selected, 2).await.unwrap();
    check(copy.view().eq_scalar(0.).await.unwrap(), vec![0; 4]).await;
}

#[tokio::test]
async fn mixed_elementwise_transforms_and_broadcast() {
    let a = source(&[0f32, 1., 2., 3., 4., 5.], Layout::Dense, 2).await;
    let one = source(&[1f64], Layout::Dense, 1).await;
    let a = a
        .view()
        .reshape(shape![2, 3])
        .unwrap()
        .flip(1)
        .unwrap()
        .cast()
        .await
        .unwrap()
        .add_scalar(1f64)
        .await
        .unwrap()
        .transpose(None)
        .unwrap();
    let rhs = one.view().broadcast(shape![3, 2]).unwrap();
    let condition = a.gt(&rhs).await.unwrap().and_scalar(255).await.unwrap();
    let expression = condition
        .cond(&a, &rhs)
        .await
        .unwrap()
        .transpose(None)
        .unwrap();
    check(expression, vec![3., 2., 1., 6., 5., 4.]).await;

    assert!(a.eq(&one.view()).await.is_err());
    assert!(a.and(&one.view()).await.is_err());
    assert!(condition.cond(&one.view(), &rhs).await.is_err());
    assert!(condition.cond(&rhs, &one.view()).await.is_err());
    let small_condition = source(&[1u8], Layout::Dense, 1).await;
    assert!(small_condition.view().cond(&a, &rhs).await.is_err());
}

#[tokio::test]
async fn conditional_batch_boundary_repeated_concurrent_and_dropped_streams() {
    let mut values = vec![0f32; 4101];
    for i in [0, 4095, 4096, 4100] {
        values[i] = 0.2;
    }
    let a = source(&values, Layout::Sparse { axis: None }, 31).await;
    let zeros = source(&vec![0f32; 4101], Layout::Sparse { axis: None }, 32).await;
    let condition = a.view().round().await.unwrap().eq_scalar(0.).await.unwrap();
    let expression = condition
        .cond(&zeros.view(), &a.view())
        .await
        .unwrap()
        .eq_scalar(0.)
        .await
        .unwrap()
        .xor_scalar(0)
        .await
        .unwrap();
    let consume = || async {
        expression
            .read_blocks()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    };
    let mut dropped = expression.read_blocks().unwrap();
    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let (first, second) = futures::join!(consume(), consume());
    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4096, 5]
    );
    assert_eq!(
        first.into_iter().flatten().collect::<Vec<_>>(),
        values
            .iter()
            .map(|v| u8::from(*v != 0.))
            .collect::<Vec<_>>()
    );
    let entries: Vec<_> = expression
        .read_sparse_elements_in_order(range![AxisRange::In(0, 4101, 1)], axes![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(
        entries,
        vec![
            (vec![0], 1),
            (vec![4095], 1),
            (vec![4096], 1),
            (vec![4100], 1)
        ]
    );
}

#[tokio::test]
async fn conditional_selected_range_propagates_each_operand_error() {
    use fensor::TensorSparseIndex;

    let (_, dir) = new_dir("conditional_corruption").await;
    let corrupt = Tensor::<FsEntry, u8>::create(
        dir.clone(),
        TensorSchema::new(u8::dtype(), shape![8]).unwrap(),
        Layout::Sparse { axis: None },
        2,
    )
    .await
    .unwrap();
    corrupt.write_value(&[0], 1).await.unwrap();
    corrupt.write_value(&[7], 1).await.unwrap();
    let id = corrupt.lookup_block_id(&[7, 3]).await.unwrap().unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete(&id.to_string()).await;
    let empty = source(&[0u8; 8], Layout::Sparse { axis: None }, 3).await;

    for position in 0..3 {
        let operands =
            std::array::from_fn::<_, 3, _>(|i| if i == position { &corrupt } else { &empty });
        let expression = operands[0]
            .view()
            .cond(&operands[1].view(), &operands[2].view())
            .await
            .unwrap()
            .or_scalar(1)
            .await
            .unwrap();
        let values: Vec<_> = expression
            .read_sparse_elements_in_order(range![AxisRange::At(0)], axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(values, vec![(vec![0], 1)]);
        assert!(expression.read_value(&[7]).await.is_err());
        let mut stream = expression
            .read_sparse_elements_in_order(range![AxisRange::At(7)], axes![0])
            .await
            .unwrap();
        assert!(stream.try_next().await.is_err());
        assert!(
            expression
                .read_blocks()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn conditional_far_end_range_is_bounded_and_validated() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (_, dir) = new_dir("conditional_far_end").await;
        let a = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![1_000_000_000]).unwrap(),
            Layout::Sparse { axis: None },
            32,
        )
        .await
        .unwrap();
        a.write_value(&[999_999_999], 127).await.unwrap();
        let condition = a.view().eq_scalar(0).await.unwrap();
        let expression = condition
            .cond(&a.view(), &a.view())
            .await
            .unwrap()
            .add_scalar(1)
            .await
            .unwrap();
        let entries: Vec<_> = expression
            .read_sparse_elements_in_order(
                range![AxisRange::Of(vec![999_999_999, 999_999_998, 999_999_999])],
                axes![0],
            )
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(entries, vec![(vec![999_999_999], 128)]);
        let stepped: Vec<_> = expression
            .read_sparse_elements_in_order(
                range![AxisRange::In(999_999_997, 1_000_000_000, 2)],
                axes![0],
            )
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(stepped, entries);
        assert!(
            expression
                .read_sparse_elements_in_order(range![AxisRange::At(1_000_000_000)], axes![0])
                .await
                .is_err()
        );
        assert!(
            expression
                .read_sparse_elements_in_order(range![], axes![1])
                .await
                .is_err()
        );
        assert!(
            expression
                .read_sparse_elements_in_order(range![AxisRange::In(1, 1, 1)], axes![0])
                .await
                .unwrap()
                .next()
                .await
                .is_none()
        );
    })
    .await
    .expect("a tiny selected range must not scan a billion coordinates");
}

#[tokio::test]
async fn conditional_live_sources_transforms_and_scalar_limit() {
    let a = source(&[1f32, 2., 3., 4.], Layout::Dense, 2).await;
    let b = source(&[5f32, 6., 7., 8.], Layout::Dense, 3).await;
    let condition = a.view().gt_scalar(2.).await.unwrap();
    let selected = condition.cond(&a.view(), &b.view()).await.unwrap();
    check(
        selected
            .clone()
            .unsqueeze(axes![0])
            .unwrap()
            .slice(range![AxisRange::At(0), AxisRange::Of(vec![3, 1, 3])])
            .unwrap()
            .unsqueeze(axes![0])
            .unwrap()
            .squeeze(axes![0])
            .unwrap(),
        vec![4., 6., 4.],
    )
    .await;
    // Construction does not capture either condition values or branch values.
    a.write_value(&[1], 9.).await.unwrap();
    b.write_value(&[0], 10.).await.unwrap();
    check(selected.clone(), vec![10., 9., 3., 4.]).await;
    let scalar = selected.slice(range![AxisRange::At(1)]).unwrap();
    assert_eq!(scalar.read_value(&[]).await.unwrap(), 9.);
    assert!(scalar.read_blocks().is_err());
    let (_, dir) = new_dir("conditional_scalar_copy").await;
    assert!(
        Tensor::<FsEntry, f32>::copy_from(dir, &scalar, 2)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn conditional_copy_under_cache_pressure_reloads() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("conditional_cache").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root).unwrap();
        let a = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![1024]).unwrap(),
            Layout::Dense,
            32,
        )
        .await
        .unwrap();
        a.write_value(&[1023], 255).await.unwrap();
        let condition = a.view().eq_scalar(0).await.unwrap();
        let expression = condition
            .cond(&a.view().add_scalar(1).await.unwrap(), &a.view())
            .await
            .unwrap();
        let (out_root, _) = new_dir("conditional_cache_copy").await;
        let out_cache =
            freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let out_dir = out_cache.load(out_root.clone()).unwrap();
        let output: Tensor<FsEntry, u8> = Tensor::copy_from(out_dir.clone(), &expression, 32)
            .await
            .unwrap();
        assert_eq!(output.read_value(&[1023]).await.unwrap(), 255);
        output.sync().await.unwrap();
        drop(output);
        drop(out_dir);
        let reloaded = Tensor::<FsEntry, u8>::load(common::open_dir(&out_root).unwrap())
            .await
            .unwrap();
        assert_eq!(reloaded.read_value(&[0]).await.unwrap(), 1);
        assert_eq!(reloaded.read_value(&[1023]).await.unwrap(), 255);
    })
    .await
    .expect("conditional copying must make progress under cache pressure");
}
