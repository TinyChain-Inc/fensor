//! Every concrete cast pair, with small fixtures rather than repeated persistence tests.

use std::cell::Cell;

use fensor::{
    Layout, Tensor, TensorCast, TensorElement, TensorFileEntry, TensorGeometry, TensorRead,
};
use ha_ndarray::{Array, Buffer, NDArrayCast, NDArrayRead, shape};

use common::{FsEntry, cleanup, fixture, numbers::same};

mod common;

// Numerical fixtures still retain typed index rows and file envelopes, even
// when their scalar values encode to only a few bytes.
const CACHE_BYTES: usize = 16 * 1024;

async fn check_cast<From: TensorElement, To: TensorElement>(
    tensor: &Tensor<FsEntry, From>,
    expected: &[To],
) where
    FsEntry: TensorFileEntry<From>,
{
    let view = TensorCast::<To>::cast(&tensor.view()).await.unwrap();
    let context = format!(
        "cast {} -> {} ({:?})",
        std::any::type_name::<From>(),
        std::any::type_name::<To>(),
        tensor.layout()
    );

    for (i, &expected) in expected.iter().enumerate() {
        let actual = view.read_value(&[i as u64]).await.unwrap();
        assert!(
            same(actual, expected),
            "{context} at [{i}]: actual {actual:?}, expected {expected:?}"
        );
    }

    let coordinate = Cell::new(0);
    fixture::blocks(&view, expected, |actual, expected| {
        let i = coordinate.get();
        assert!(
            same(actual, expected),
            "{context} at [{i}]: actual {actual:?}, expected {expected:?}"
        );
        coordinate.set(i + 1);
        true
    })
    .await;
}

async fn pair<From: TensorElement, To: TensorElement>(
    tensor: &Tensor<FsEntry, From>,
    input: &[From],
) where
    FsEntry: TensorFileEntry<From>,
{
    let backend = Array::new(Buffer::from(input.to_vec()), shape![input.len()]).unwrap();
    let cast = NDArrayCast::<To>::cast(backend).unwrap();
    let mut expected = cast.buffer().unwrap().to_slice().unwrap().to_vec();

    for (i, (&value, converted)) in input.iter().zip(&mut expected).enumerate() {
        let reference = To::cast_from(value.into());
        assert!(
            same(*converted, reference),
            "cast {} -> {} ({:?}) at [{i}]: backend {converted:?}, scalar {reference:?}",
            std::any::type_name::<From>(),
            std::any::type_name::<To>(),
            tensor.layout()
        );

        if matches!(tensor.layout(), Layout::Sparse { .. }) && value == From::ZERO {
            *converted = To::ZERO;
        }
    }

    check_cast(tensor, &expected).await;
}

async fn destinations<T: TensorElement>(input: &[T])
where
    FsEntry: TensorFileEntry<T>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, tensor) = fixture::source(
            "cast_source",
            shape![input.len() as u64],
            layout,
            3,
            CACHE_BYTES,
            input.iter().copied(),
        )
        .await;

        pair::<T, u8>(&tensor, input).await;
        pair::<T, u16>(&tensor, input).await;
        pair::<T, u32>(&tensor, input).await;
        pair::<T, u64>(&tensor, input).await;
        pair::<T, i8>(&tensor, input).await;
        pair::<T, i16>(&tensor, input).await;
        pair::<T, i32>(&tensor, input).await;
        pair::<T, i64>(&tensor, input).await;
        pair::<T, f32>(&tensor, input).await;
        pair::<T, f64>(&tensor, input).await;
        #[cfg(feature = "complex")]
        {
            pair::<T, fensor::complex::Complex32>(&tensor, input).await;
            pair::<T, fensor::complex::Complex64>(&tensor, input).await;
        }

        drop(tensor);
        cleanup(&root).await;
    }
}

macro_rules! integer_source {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            destinations::<$t>(&[0, 1, 127, <$t>::MIN, <$t>::MAX]).await;
        }
    };
}

integer_source!(u8_casts, u8);
integer_source!(u16_casts, u16);
integer_source!(u32_casts, u32);
integer_source!(u64_casts, u64);
integer_source!(i8_casts, i8);
integer_source!(i16_casts, i16);
integer_source!(i32_casts, i32);
integer_source!(i64_casts, i64);

#[tokio::test]
async fn f32_casts() {
    destinations(&[
        0f32,
        -0.,
        -1.5,
        1.5,
        255.5,
        f32::MAX,
        f32::MIN,
        f32::from_bits(1),
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ])
    .await;
}

#[tokio::test]
async fn f64_casts() {
    destinations(&[
        0f64,
        -0.,
        -1.5,
        1.5,
        255.5,
        16_777_217.,
        9_007_199_254_740_993u64 as f64,
        f64::MAX,
        f64::MIN,
        f64::from_bits(1),
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ])
    .await;
}

async fn literal_cast<From: TensorElement, To: TensorElement>(input: &[From], expected: &[To])
where
    FsEntry: TensorFileEntry<From>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, tensor) = fixture::source(
            "literal_cast",
            shape![input.len() as u64],
            layout,
            3,
            CACHE_BYTES,
            input.iter().copied(),
        )
        .await;

        check_cast(&tensor, expected).await;
        drop(tensor);
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn literal_cast_boundaries() {
    literal_cast(&[-1i64], &[255u8]).await;
    literal_cast(&[-1f64, 256.], &[0u8, 0]).await;
    literal_cast(
        &[16_777_217i32, -16_777_217],
        &[16_777_216f64, -16_777_216.],
    )
    .await;
    literal_cast(&[-1i8], &[255u64]).await;
    literal_cast(&[u32::MAX], &[-1i64]).await;
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex32_casts() {
    use fensor::complex::Complex32 as C;
    destinations(&[
        C::new(0., -0.),
        C::new(1.5, -2.5),
        C::new(0., 1.),
        C::new(f32::NAN, 2.),
        C::new(2., f32::NAN),
        C::new(f32::INFINITY, f32::NEG_INFINITY),
    ])
    .await;
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex64_casts() {
    use fensor::complex::Complex64 as C;
    destinations(&[
        C::new(-0., 0.),
        C::new(1.5, -2.5),
        C::new(0., 1.),
        C::new(f64::NAN, 2.),
        C::new(2., f64::NAN),
        C::new(f64::MAX, f64::MIN),
    ])
    .await;
}
