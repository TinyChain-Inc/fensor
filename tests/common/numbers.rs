//! Comparators preserve floating classifications and signed zeros, including components.

use fensor::TensorElement;
use number_general::{Complex, Float, Number};

fn real(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan())
        || (a == b && (a != 0. || a.is_sign_negative() == b.is_sign_negative()))
}

pub fn same<T: TensorElement>(actual: T, expected: T) -> bool {
    match (actual.into(), expected.into()) {
        (Number::Float(Float::F32(a)), Number::Float(Float::F32(b))) => real(a.into(), b.into()),
        (Number::Float(Float::F64(a)), Number::Float(Float::F64(b))) => real(a, b),
        (Number::Complex(Complex::C32(a)), Number::Complex(Complex::C32(b))) => {
            real(a.re.into(), b.re.into()) && real(a.im.into(), b.im.into())
        }
        (Number::Complex(Complex::C64(a)), Number::Complex(Complex::C64(b))) => {
            real(a.re, b.re) && real(a.im, b.im)
        }
        _ => actual == expected,
    }
}
