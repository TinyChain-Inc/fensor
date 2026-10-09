#![cfg(feature = "complex")]

use fensor::{
    AxisRange, Error, Layout, Tensor, TensorElement, TensorExpression, TensorFileEntry,
    TensorFourier, TensorGeometry, TensorMatMul, TensorMath, TensorMatrixUnaryComplex, TensorRead,
    TensorReduce, TensorReduceAll, TensorSchema, TensorTransform, TensorUnary, TensorViewSemantics,
    TensorWrite,
    complex::{Complex32, Complex64},
    fft::{Fft, FourierOp, Ifft, fft2, ifft2},
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, ArrayAccess, Buffer, Complex, NDArrayFourier, NDArrayRead, NDArrayTransform, shape,
};
use number_general::DType;
use safecast::CastFrom;

use common::{FsEntry, cleanup, fixture, new_dir, numbers::same};

mod common;

fn value<T: TensorElement>(re: f64, im: f64) -> T {
    T::cast_from(Complex64::new(re, im).into())
}

fn components<T: Complex>(v: T) -> (f64, f64) {
    (f64::cast_from(v.re().into()), f64::cast_from(v.im().into()))
}

fn close<T: Complex>(a: T, b: T, bound: f64) -> bool {
    let (ar, ai) = components(a);
    let (br, bi) = components(b);
    (ar - br).abs() <= bound && (ai - bi).abs() <= bound
}

async fn consumers<V: TensorRead>(view: &V, expected: &[V::DType], bound: f64)
where
    V::DType: TensorElement + Complex,
    FsEntry: TensorFileEntry<V::DType>,
{
    fixture::consumers(
        view,
        expected,
        |a, b| close(a, b, bound),
        |a, b| close(a, b, bound),
    )
    .await;
    let mut stream = view.read_coordinate_blocks().unwrap();
    let mut seen = vec![false; expected.len()];
    while let Some((coords, values)) = stream.try_next().await.unwrap() {
        assert_eq!(coords.len(), values.len());
        for (coord, actual) in coords.into_iter().zip(values) {
            let flat = coord
                .iter()
                .zip(view.shape())
                .fold(0, |n, (i, d)| n * d + i) as usize;
            assert!(!seen[flat]);
            seen[flat] = true;
            assert!(close(actual, expected[flat], bound), "coordinate {coord:?}");
        }
    }
    assert!(seen.into_iter().all(|v| v));
}

async fn transforms<T: TensorElement + Complex>(u: f64)
where
    T::Real: TensorElement,
    FsEntry: TensorFileEntry<T>,
    Fft: FourierOp<T>,
    Ifft: FourierOp<T>,
    ArrayAccess<'static, T>: NDArrayFourier<DType = T, Platform = ha_ndarray::Platform>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        for n in [1, 3, 8, 17] {
            // Distinct batches with nonzero real and imaginary components.
            let input: Vec<T> = (0..3 * n)
                .map(|i| value((i % 7) as f64 - 3., (i % 5) as f64 - 2.))
                .collect();
            let (root, tensor) = fixture::source(
                "fourier",
                shape![3, n as u64],
                layout,
                7,
                if matches!(layout, Layout::Dense) {
                    // Admit typed complex blocks and file envelopes, not encoded bytes.
                    1024
                } else {
                    1_000_000
                },
                input.clone(),
            )
            .await;
            let source = tensor.view();
            let backend =
                ArrayAccess::from(Array::new(Buffer::from(input.clone()), shape![3, n]).unwrap());
            let forward = source.fft().await.unwrap();
            let inverse = source.ifft().await.unwrap();
            let expected = backend
                .clone()
                .fft()
                .unwrap()
                .buffer()
                .unwrap()
                .to_slice()
                .unwrap()
                .to_vec();
            let inverse_expected = backend
                .ifft()
                .unwrap()
                .buffer()
                .unwrap()
                .to_slice()
                .unwrap()
                .to_vec();
            let gamma = (32. * n as f64 * u) / (1. - 32. * n as f64 * u);
            let scale = input
                .chunks(n)
                .map(|group| {
                    group
                        .iter()
                        .map(|&v| {
                            let (r, i) = components(v);
                            r.hypot(i)
                        })
                        .sum::<f64>()
                })
                .fold(0., f64::max);
            consumers(&forward, &expected, gamma * scale).await;
            consumers(&inverse, &inverse_expected, gamma * scale).await;

            // Independently compute each DFT, using f64 only for the tiny reference.
            // Its rounding error is covered by the much larger backend gamma bound.
            for (batch, values) in input.chunks(n).enumerate() {
                for k in 0..n {
                    let reference = |sign: f64| {
                        let sum = values.iter().enumerate().fold(
                            Complex64::new(0., 0.),
                            |sum, (j, &v)| {
                                let (r, i) = components(v);
                                let phase =
                                    sign * std::f64::consts::TAU * (j * k) as f64 / n as f64;
                                sum + Complex64::new(r, i) * Complex64::from_polar(1., phase)
                            },
                        );
                        value::<T>(sum.re, sum.im)
                    };
                    assert!(close(
                        forward.read_value(&[batch as u64, k as u64]).await.unwrap(),
                        reference(-1.),
                        gamma * scale
                    ));
                    assert!(close(
                        inverse.read_value(&[batch as u64, k as u64]).await.unwrap(),
                        reference(1.),
                        gamma * scale
                    ));
                }
            }

            // Forward error propagates through the inverse's absolute row sum N.
            let roundtrip = forward.ifft().await.unwrap();
            let expected_roundtrip: Vec<_> = input
                .iter()
                .map(|&v| {
                    let (r, i) = components(v);
                    value::<T>(r * n as f64, i * n as f64)
                })
                .collect();
            let bound = n as f64 * gamma * scale * (2. + gamma);
            fixture::blocks(&roundtrip, &expected_roundtrip, |a, b| close(a, b, bound)).await;

            let mut dropped = forward.read_blocks().unwrap();
            dropped.try_next().await.unwrap();
            drop(dropped);
            let (a, b) = futures::join!(forward.read_value(&[0, 0]), forward.read_value(&[0, 0]));
            assert!(same(a.unwrap(), b.unwrap()));

            let gathered = forward
                .clone()
                .slice(smallvec::smallvec![
                    AxisRange::Of(vec![2, 0, 2]),
                    AxisRange::Of(vec![(n - 1) as u64, 0, 0])
                ])
                .unwrap()
                .flip(1)
                .unwrap()
                .transpose(None)
                .unwrap();
            let mut gathered_expected = Vec::new();
            for frequency in [0, 0, n - 1] {
                for batch in [2, 0, 2] {
                    gathered_expected.push(expected[batch * n + frequency]);
                }
            }
            fixture::blocks(&gathered, &gathered_expected, |a, b| {
                close(a, b, gamma * scale)
            })
            .await;
            cleanup(&root).await;
        }
    }
}

#[tokio::test]
async fn complex32_transforms() {
    transforms::<Complex32>(f32::EPSILON as f64 / 2.).await;
}

#[tokio::test]
async fn complex64_transforms() {
    transforms::<Complex64>(f64::EPSILON / 2.).await;
}

async fn matrix_helpers<T: TensorElement + Complex>(u: f64)
where
    T::Real: TensorElement,
    FsEntry: TensorFileEntry<T>,
    Fft: FourierOp<T>,
    Ifft: FourierOp<T>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let input: Vec<T> = (0..12).map(|i| value(i as f64, 1. - i as f64)).collect();
        let (root, tensor) = fixture::source(
            "fourier_matrix",
            shape![2, 2, 3],
            layout,
            3,
            if matches!(layout, Layout::Dense) {
                512
            } else {
                1_000_000
            },
            input.clone(),
        )
        .await;
        let v = tensor.view();
        let hermitian = v.mh().await.unwrap();
        assert_eq!(hermitian.shape(), &[2, 3, 2]);
        assert!(!hermitian.supports_write_through());
        let mut expected = Vec::new();
        for b in 0..2 {
            for j in 0..3 {
                for i in 0..2 {
                    expected.push(input[b * 6 + i * 3 + j].conj());
                }
            }
        }
        consumers(&hermitian, &expected, 0.).await;

        let forward = fft2(&v).await.unwrap();
        let inverse = ifft2(&v).await.unwrap();
        let gamma = |n: f64| 32. * n * u / (1. - 32. * n * u);
        let delta = gamma(2.) + gamma(3.) + gamma(2.) * gamma(3.);
        let scale = input
            .chunks(6)
            .map(|group| {
                group
                    .iter()
                    .map(|&v| {
                        let (r, i) = components(v);
                        r.hypot(i)
                    })
                    .sum::<f64>()
            })
            .fold(0., f64::max);
        let bound = delta * scale;

        let backend =
            ArrayAccess::from(Array::new(Buffer::from(input.clone()), shape![2, 2, 3]).unwrap());
        let axes = ha_ndarray::axes![0, 2, 1];
        let first = Fft
            .apply(backend.clone())
            .unwrap()
            .transpose(axes.clone())
            .unwrap();
        let expected_forward = Fft
            .apply(ArrayAccess::from(first))
            .unwrap()
            .transpose(axes.clone())
            .unwrap()
            .buffer()
            .unwrap()
            .to_slice()
            .unwrap()
            .to_vec();
        let first = Ifft
            .apply(ArrayAccess::from(backend.transpose(axes.clone()).unwrap()))
            .unwrap()
            .transpose(axes)
            .unwrap();
        let expected_inverse = Ifft
            .apply(ArrayAccess::from(first))
            .unwrap()
            .buffer()
            .unwrap()
            .to_slice()
            .unwrap()
            .to_vec();
        let reference = |sign: f64| {
            let mut values = Vec::new();
            for b in 0..2 {
                for r in 0..2 {
                    for c in 0..3 {
                        let mut sum = Complex64::new(0., 0.);
                        for i in 0..2 {
                            for j in 0..3 {
                                let (re, im) = components(input[b * 6 + i * 3 + j]);
                                let phase = sign
                                    * std::f64::consts::TAU
                                    * ((r * i) as f64 / 2. + (c * j) as f64 / 3.);
                                sum += Complex64::new(re, im) * Complex64::from_polar(1., phase);
                            }
                        }
                        values.push(value::<T>(sum.re, sum.im));
                    }
                }
            }
            values
        };
        consumers(&forward, &expected_forward, bound).await;
        consumers(&inverse, &expected_inverse, bound).await;
        fixture::blocks(&forward, &reference(-1.), |a, b| close(a, b, bound)).await;
        fixture::blocks(&inverse, &reference(1.), |a, b| close(a, b, bound)).await;
        let roundtrip = ifft2(&forward).await.unwrap();
        let expected: Vec<_> = input
            .iter()
            .map(|&v| {
                let (r, i) = components(v);
                value::<T>(6. * r, 6. * i)
            })
            .collect();
        fixture::blocks(&roundtrip, &expected, |a, b| {
            close(a, b, 6. * bound * (2. + delta))
        })
        .await;

        assert!(forward.clone().reshape(shape![12]).is_err());
        let reshaped = inverse
            .clone()
            .reshape(shape![12])
            .unwrap()
            .unsqueeze(smallvec::smallvec![0])
            .unwrap()
            .squeeze(smallvec::smallvec![0])
            .unwrap()
            .flip(0)
            .unwrap();
        let mut expected = reference(1.);
        expected.reverse();
        fixture::blocks(&reshaped, &expected, |a, b| close(a, b, bound)).await;
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn complex32_matrix_helpers() {
    matrix_helpers::<Complex32>(f32::EPSILON as f64 / 2.).await;
}

#[tokio::test]
async fn complex64_matrix_helpers() {
    matrix_helpers::<Complex64>(f64::EPSILON / 2.).await;
}

#[tokio::test]
async fn sparse_group_support_and_live_sources() {
    let z = Complex64::ZERO;
    let one = Complex64::ONE;
    let (root, tensor) = fixture::source(
        "fourier_support",
        shape![2, 4],
        Layout::Sparse { axis: None },
        2,
        1024,
        [one, z, z, z, z, z, z, z],
    )
    .await;
    let v = tensor.view();
    let forward = v.fft().await.unwrap();
    consumers(&forward, &[one, one, one, one, z, z, z, z], 0.).await;
    let zero = v.sub(&v).await.unwrap();
    let transformed = zero.fft().await.unwrap();
    assert!(matches!(
        transformed.exp().await,
        Err(Error::WouldDensify { operation: "exp" })
    ));
    let retained = TensorExpression::new(transformed)
        .unwrap()
        .into_dense()
        .exp()
        .await
        .unwrap();
    consumers(&retained, &[one; 8], 0.).await;
    let (copy_root, dir) = new_dir("fourier_support_copy").await;
    let copy = Tensor::<FsEntry, Complex64>::copy_from(dir, &zero, zero.layout(), 2)
        .await
        .unwrap();
    let copied = TensorExpression::new(copy.view().fft().await.unwrap())
        .unwrap()
        .into_dense()
        .exp()
        .await
        .unwrap();
    fixture::blocks(&copied, &[one; 8], same).await;

    // Both transform axes retain ordinary zeros before dense conversion.
    let both_axes = TensorExpression::new(fft2(&zero).await.unwrap())
        .unwrap()
        .into_dense()
        .exp()
        .await
        .unwrap();
    fixture::blocks(&both_axes, &[one; 8], same).await;
    tensor.write_value(&[0, 0], z).await.unwrap();
    tensor.write_value(&[1, 1], one).await.unwrap();
    assert_eq!(forward.read_value(&[0, 0]).await.unwrap(), z);
    assert_eq!(forward.read_value(&[1, 0]).await.unwrap(), one);
    assert_eq!(forward.sum_all().await.unwrap(), z);
    cleanup(&copy_root).await;
    cleanup(&root).await;
}

#[tokio::test]
async fn bounds_corruption_and_exceptional_values() {
    for shape in [shape![4097], shape![4097, 1], shape![1, 4097]] {
        let (root, dir) = new_dir("fourier_limit").await;
        let tensor = Tensor::<FsEntry, Complex64>::create(
            dir,
            TensorSchema::new(Complex64::dtype(), shape.clone()).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        if *shape.last().unwrap() > 4096 {
            assert!(
                matches!(tensor.view().fft().await, Err(Error::Unsupported(message)) if message.contains("4097") && message.contains("4096"))
            );
        }
        assert!(fft2(&tensor.view()).await.is_err());
        cleanup(&root).await;
    }

    let (root, tensor) = fixture::source(
        "fourier_corrupt",
        shape![2, 4],
        Layout::Dense,
        4,
        1024,
        [Complex64::ONE; 8],
    )
    .await;
    tensor.sync().await.unwrap();
    drop(tensor);
    let dir = common::open_dir(&root).unwrap();
    let tensor = Tensor::<FsEntry, Complex64>::load(dir.clone())
        .await
        .unwrap();
    let v = tensor.view();
    assert!(v.clone().reshape(shape![8]).unwrap().mh().await.is_err());
    assert!(fft2(&v.clone().reshape(shape![8]).unwrap()).await.is_err());
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let file = blocks.read().await.get_file("1").unwrap().clone();
    file.write::<Vec<Complex64>>(0).await.unwrap().clear();
    let fft = v.fft().await.unwrap();
    assert_eq!(
        fft.read_value(&[0, 0]).await.unwrap(),
        Complex64::new(4., 0.)
    );
    assert!(fft.read_value(&[1, 0]).await.is_err());
    drop(dir);
    cleanup(&root).await;

    for x in [
        Complex64::new(f64::NAN, 0.),
        Complex64::new(f64::INFINITY, -0.),
        Complex64::new(-0., 0.),
    ] {
        let (root, tensor) = fixture::source(
            "fourier_exceptional",
            shape![1],
            Layout::Dense,
            1,
            1024,
            [x],
        )
        .await;
        let expected = Array::new(Buffer::from(vec![x]), shape![1])
            .unwrap()
            .fft()
            .unwrap()
            .buffer()
            .unwrap()
            .to_slice()
            .unwrap()
            .to_vec();
        fixture::blocks(&tensor.view().fft().await.unwrap(), &expected, same).await;
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn expression_boundaries_and_tiny_far_end_selection() {
    let (root, tensor) = fixture::source(
        "fft_composition",
        shape![2, 2],
        Layout::Dense,
        2,
        1024,
        [Complex64::ONE; 4],
    )
    .await;
    let v = tensor.view();
    let product = v.matmul(&v).await.unwrap().fft().await.unwrap();
    fixture::blocks(
        &product,
        &[
            Complex64::new(4., 0.),
            Complex64::ZERO,
            Complex64::new(4., 0.),
            Complex64::ZERO,
        ],
        same,
    )
    .await;
    let reduced = v
        .sum(ha_ndarray::axes![0], true)
        .await
        .unwrap()
        .fft()
        .await
        .unwrap();
    fixture::blocks(&reduced, &[Complex64::new(4., 0.), Complex64::ZERO], same).await;
    let before = v.clone().flip(0).unwrap().fft().await.unwrap();
    fixture::blocks(
        &before,
        &[
            Complex64::new(2., 0.),
            Complex64::ZERO,
            Complex64::new(2., 0.),
            Complex64::ZERO,
        ],
        same,
    )
    .await;
    cleanup(&root).await;

    let (root, dir) = new_dir("fft_huge").await;
    let n = 1_000_000_000;
    let tensor = Tensor::<FsEntry, Complex64>::create(
        dir,
        TensorSchema::new(Complex64::dtype(), shape![n, 4]).unwrap(),
        Layout::Sparse { axis: None },
        4,
    )
    .await
    .unwrap();
    tensor
        .write_value(&[n - 1, 0], Complex64::ONE)
        .await
        .unwrap();
    let view = tensor.view().fft().await.unwrap();
    let values: Vec<_> = view
        .read_sparse_elements_in_order(
            smallvec::smallvec![AxisRange::At(n - 1), AxisRange::In(0, 4, 1)],
            ha_ndarray::axes![0, 1],
        )
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(
        values,
        (0..4)
            .map(|i| (vec![n - 1, i], Complex64::ONE))
            .collect::<Vec<_>>()
    );
    cleanup(&root).await;
}

#[tokio::test]
async fn largest_axis_impulse_and_multiple_packs() {
    // Large sparse source has a complete, bounded transform axis.
    let (root, dir) = new_dir("fourier_boundary").await;
    let tensor = Tensor::<FsEntry, Complex32>::create(
        dir,
        TensorSchema::new(Complex32::dtype(), shape![2, 4096]).unwrap(),
        Layout::Sparse { axis: None },
        31,
    )
    .await
    .unwrap();
    tensor.write_value(&[1, 0], Complex32::ONE).await.unwrap();
    let view = tensor.view().fft().await.unwrap();
    let expected: Vec<_> = (0..8192)
        .map(|i| {
            if i < 4096 {
                Complex32::ZERO
            } else {
                Complex32::ONE
            }
        })
        .collect();
    fixture::blocks(&view, &expected, same).await;
    cleanup(&root).await;
}
