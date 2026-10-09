//! Shared sparse consumption through operation-specific, transformed expressions.
use fensor::{
    AxisRange, Layout, Tensor, TensorElement, TensorMatMul, TensorMatrixUnary, TensorRead,
    TensorSchema, TensorTransform, TensorTrig, TensorUnary, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{Number, shape};
use number_general::DType;

mod common;
use common::{FsEntry, cleanup, iter_coords, new_dir, numbers::same};

async fn parity<V: TensorRead>(view: &V, empty: bool, intermediate_zero: bool)
where
    V::DType: TensorElement,
{
    let values = view.read_blocks().unwrap().try_concat().await.unwrap();
    assert!(
        values.len() > 4096,
        "fixture spans multiple execution batches"
    );
    if empty {
        assert!(values.iter().all(|value| *value == V::DType::ZERO));
    } else if intermediate_zero {
        // Zero-preserving composition treats intermediate and implicit zeros alike.
        assert!(values.contains(&V::DType::ZERO));
    }
    let full: fensor::Range = view
        .shape()
        .iter()
        .map(|&dim| AxisRange::In(0, dim, 1))
        .collect();
    let mut selected = full.clone();
    let last = view.ndim() - 1;
    selected[last] = AxisRange::Of(vec![view.shape()[last] - 1, 0, view.shape()[last] - 1, 1]);
    let mut empty_range = full.clone();
    empty_range[0] = AxisRange::In(0, 0, 1);
    for range in [full, selected, empty_range] {
        let expected: Vec<_> = iter_coords(view.shape())
            .zip(values.iter().copied())
            .filter(|(coord, value)| {
                *value != V::DType::ZERO
                    && coord.iter().zip(&range).all(|(&c, axis)| match axis {
                        AxisRange::At(at) => c == *at,
                        AxisRange::In(start, end, step) => {
                            c >= *start && c < *end && (c - start) % step == 0
                        }
                        AxisRange::Of(indices) => indices.contains(&c),
                    })
            })
            .collect();
        let actual: Vec<_> = view
            .read_sparse_elements_in_order(range, (0..view.ndim()).collect())
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(actual.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for ((coord, actual), (expected_coord, expected)) in actual.into_iter().zip(expected) {
            assert_eq!(coord, expected_coord);
            assert!(
                same(actual, expected),
                "at {coord:?}: {actual:?} != {expected:?}"
            );
        }
    }
}

#[tokio::test]
async fn matrix_and_diagonal_sparse_consumers_match_row_major_values() {
    for empty in [false, true] {
        let (left_root, left_dir) = new_dir("sparse_consumer_left").await;
        let left = Tensor::<FsEntry, f64>::create(
            left_dir,
            TensorSchema::new(f64::dtype(), shape![4101, 2]).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        let (right_root, right_dir) = new_dir("sparse_consumer_right").await;
        let right = Tensor::<FsEntry, f64>::create(
            right_dir,
            TensorSchema::new(f64::dtype(), shape![2, 2]).unwrap(),
            Layout::Sparse { axis: None },
            3,
        )
        .await
        .unwrap();
        if !empty {
            for (coord, value) in [
                ([0, 0], 1.0),
                ([0, 1], -1.0),
                ([2048, 0], f64::NAN),
                ([4099, 0], f64::INFINITY),
                ([4100, 0], 3.0),
            ] {
                left.write_value(&coord, value).await.unwrap();
            }
            for row in 0..2 {
                right.write_value(&[row, 0], 1.0).await.unwrap();
                right.write_value(&[row, 1], 2.0).await.unwrap();
            }
        }
        let product = left
            .view()
            .matmul(&right.view())
            .await
            .unwrap()
            .transpose(None)
            .unwrap()
            .flip(1)
            .unwrap();
        parity(&product, empty, false).await;
        assert!(matches!(
            product.exp().await,
            Err(fensor::Error::WouldDensify { .. })
        ));
        parity(&product.sin().await.unwrap(), empty, true).await;
        cleanup(&left_root).await;
        cleanup(&right_root).await;

        let (root, dir) = new_dir("sparse_consumer_diagonal").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), shape![2, 4097, 4097]).unwrap(),
            Layout::Sparse { axis: None },
            1,
        )
        .await
        .unwrap();
        if !empty {
            for (coord, value) in [
                ([0, 0, 0], 0.2),
                ([0, 4096, 4096], f64::NAN),
                ([1, 0, 0], f64::INFINITY),
                ([1, 4096, 4096], f64::NEG_INFINITY),
            ] {
                tensor.write_value(&coord, value).await.unwrap();
            }
        }
        let diagonal = tensor
            .view()
            .diag()
            .await
            .unwrap()
            .transpose(None)
            .unwrap()
            .flip(0)
            .unwrap();
        parity(&diagonal, empty, false).await;
        assert!(matches!(
            diagonal.exp().await,
            Err(fensor::Error::WouldDensify { .. })
        ));
        parity(
            &diagonal.round().await.unwrap().sin().await.unwrap(),
            empty,
            true,
        )
        .await;
        cleanup(&root).await;
    }
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn fourier_sparse_consumers_match_row_major_values() {
    use fensor::{TensorFourier, complex::Complex64};
    for empty in [false, true] {
        let (root, dir) = new_dir("sparse_consumer_fourier").await;
        let tensor = Tensor::<FsEntry, Complex64>::create(
            dir,
            TensorSchema::new(Complex64::dtype(), shape![4101, 2]).unwrap(),
            Layout::Sparse { axis: None },
            7,
        )
        .await
        .unwrap();
        if !empty {
            for (coord, value) in [
                ([0, 0], 0.5),
                ([0, 1], 0.5),
                ([2048, 0], f64::NAN),
                ([4099, 0], f64::INFINITY),
                ([4100, 0], 0.5),
            ] {
                tensor
                    .write_value(&coord, Complex64::new(value, 0.0))
                    .await
                    .unwrap();
            }
        }
        let spectrum = tensor
            .view()
            .fft()
            .await
            .unwrap()
            .transpose(None)
            .unwrap()
            .flip(1)
            .unwrap();
        parity(&spectrum, empty, false).await;
        assert!(matches!(
            spectrum.exp().await,
            Err(fensor::Error::WouldDensify { .. })
        ));
        parity(&spectrum.sin().await.unwrap(), empty, true).await;
        cleanup(&root).await;
    }
}
