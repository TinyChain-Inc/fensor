use fensor::{
    Error, Layout, TensorAbs, TensorBoolean, TensorBooleanScalar, TensorCompare,
    TensorCompareScalar, TensorExpression, TensorGeometry, TensorMath, TensorMathScalar,
    TensorTrig, TensorUnary, TensorUnaryBoolean,
};

mod common;
use common::{fixture, numbers};

fn rejects_densification(result: fensor::Result<()>, expected: &'static str) {
    match result {
        Err(Error::WouldDensify { operation }) => assert_eq!(operation, expected),
        Err(error) => panic!("{expected}: expected WouldDensify, got {error}"),
        Ok(()) => panic!("{expected}: accepted a nonzero sparse background"),
    }
}

#[tokio::test]
async fn sparse_operation_families_validate_the_zero_background() {
    let (_dense_root, dense) = fixture::source(
        "sparse_policy_dense",
        vec![1, 4].into(),
        Layout::Dense,
        4,
        1 << 20,
        [2.0_f32, 0., 0., 0.],
    )
    .await;

    for axis in [None, Some(0)] {
        let (_root, sparse) = fixture::source(
            "sparse_policy",
            vec![1, 4].into(),
            Layout::Sparse { axis },
            4,
            1 << 20,
            [2.0_f32, 0., 0., 0.],
        )
        .await;
        let view = sparse.view();

        for (operation, result) in [
            ("exp", view.exp().await.map(|_| ())),
            ("ln", view.ln().await.map(|_| ())),
            ("cos", view.cos().await.map(|_| ())),
            ("acos", view.acos().await.map(|_| ())),
            ("cosh", view.cosh().await.map(|_| ())),
            ("not", view.not().await.map(|_| ())),
            ("add_scalar", view.add_scalar(1.).await.map(|_| ())),
            ("sub_scalar", view.sub_scalar(1.).await.map(|_| ())),
            ("div_scalar", view.div_scalar(0.).await.map(|_| ())),
            ("rem_scalar", view.rem_scalar(0.).await.map(|_| ())),
            ("pow_scalar", view.pow_scalar(0.).await.map(|_| ())),
            ("pow_scalar", view.pow_scalar(-1.).await.map(|_| ())),
            ("log_scalar", view.log_scalar(2.).await.map(|_| ())),
            ("eq_scalar", view.eq_scalar(0.).await.map(|_| ())),
            ("ne_scalar", view.ne_scalar(1.).await.map(|_| ())),
            ("gt_scalar", view.gt_scalar(-1.).await.map(|_| ())),
            ("ge_scalar", view.ge_scalar(0.).await.map(|_| ())),
            ("lt_scalar", view.lt_scalar(1.).await.map(|_| ())),
            ("le_scalar", view.le_scalar(0.).await.map(|_| ())),
            ("or_scalar", view.or_scalar(1.).await.map(|_| ())),
            ("xor_scalar", view.xor_scalar(1.).await.map(|_| ())),
            ("div", view.div(&view).await.map(|_| ())),
            ("rem", view.rem(&view).await.map(|_| ())),
            ("pow", view.pow(&view).await.map(|_| ())),
            ("log", view.log(&view).await.map(|_| ())),
            ("eq", view.eq(&view).await.map(|_| ())),
            ("ge", view.ge(&view).await.map(|_| ())),
            ("le", view.le(&view).await.map(|_| ())),
        ] {
            rejects_densification(result, operation);
        }

        for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            rejects_densification(view.mul_scalar(scalar).await.map(|_| ()), "mul_scalar");
        }
        for (operation, result) in [
            ("div_scalar", view.div_scalar(f32::NAN).await.map(|_| ())),
            ("rem_scalar", view.rem_scalar(f32::NAN).await.map(|_| ())),
            ("pow_scalar", view.pow_scalar(f32::NAN).await.map(|_| ())),
        ] {
            rejects_densification(result, operation);
        }

        let actual = view
            .abs()
            .await
            .unwrap()
            .mul_scalar(3.)
            .await
            .unwrap()
            .div_scalar(2.)
            .await
            .unwrap()
            .pow_scalar(2.)
            .await
            .unwrap();
        let expected = dense
            .view()
            .abs()
            .await
            .unwrap()
            .mul_scalar(3.)
            .await
            .unwrap()
            .div_scalar(2.)
            .await
            .unwrap()
            .pow_scalar(2.)
            .await
            .unwrap();
        fixture::reads(&actual, &[9., 0., 0., 0.], numbers::same).await;
        fixture::blocks(&expected, &[9., 0., 0., 0.], numbers::same).await;
        fixture::reads(
            &view.add(&view).await.unwrap(),
            &[4., 0., 0., 0.],
            numbers::same,
        )
        .await;
        fixture::reads(
            &view.mul(&view).await.unwrap(),
            &[4., 0., 0., 0.],
            numbers::same,
        )
        .await;
        fixture::reads(&view.gt_scalar(1.).await.unwrap(), &[1, 0, 0, 0], |a, b| {
            a == b
        })
        .await;
        fixture::reads(&view.ne(&view).await.unwrap(), &[0; 4], |a, b| a == b).await;
        fixture::reads(&view.or(&view).await.unwrap(), &[1, 0, 0, 0], |a, b| a == b).await;
        fixture::reads(
            &view.and_scalar(1.).await.unwrap(),
            &[1, 0, 0, 0],
            |a, b| a == b,
        )
        .await;

        fixture::reads(
            &view.div_scalar(f32::INFINITY).await.unwrap(),
            &[0.; 4],
            numbers::same,
        )
        .await;
        fixture::reads(&view.eq_scalar(2.).await.unwrap(), &[1, 0, 0, 0], |a, b| {
            a == b
        })
        .await;
        let mixed = view.pow(&dense.view()).await.unwrap();
        assert_eq!(mixed.layout(), Layout::Dense);
        fixture::reads(&mixed, &[4., 1., 1., 1.], numbers::same).await;

        let sparse = TensorExpression::new(sparse.clone()).unwrap();
        let explicit_dense = sparse.clone().into_dense();
        assert!(matches!(sparse.layout(), Layout::Sparse { .. }));
        assert_eq!(explicit_dense.layout(), Layout::Dense);
        rejects_densification(sparse.exp().await.map(|_| ()), "exp");
        let exponentials = [2.0_f32.exp(), 1., 1., 1.];
        fixture::reads(
            &explicit_dense.exp().await.unwrap(),
            &exponentials,
            numbers::same,
        )
        .await;
        fixture::blocks(
            &dense.view().exp().await.unwrap(),
            &exponentials,
            numbers::same,
        )
        .await;
    }
}

#[tokio::test]
async fn integer_zero_denominators_follow_native_zero_semantics() {
    let (_root, sparse) = fixture::source(
        "sparse_integer_policy",
        vec![1, 4].into(),
        Layout::Sparse { axis: None },
        4,
        1 << 20,
        [2_i32, 0, 0, 0],
    )
    .await;
    let view = sparse.view();
    fixture::reads(&view.div_scalar(0).await.unwrap(), &[0; 4], |a, b| a == b).await;
    fixture::reads(&view.rem_scalar(0).await.unwrap(), &[0; 4], |a, b| a == b).await;
    fixture::reads(&view.div(&view).await.unwrap(), &[1, 0, 0, 0], |a, b| {
        a == b
    })
    .await;
    fixture::reads(&view.rem(&view).await.unwrap(), &[0; 4], |a, b| a == b).await;
    fixture::reads(&view.div_scalar(2).await.unwrap(), &[1, 0, 0, 0], |a, b| {
        a == b
    })
    .await;
    fixture::reads(&view.rem_scalar(2).await.unwrap(), &[0; 4], |a, b| a == b).await;
}

#[tokio::test]
async fn sparse_multiplication_visits_both_operands_for_nonfinite_values() {
    let (_left_root, left) = fixture::source(
        "sparse_ieee_left",
        vec![1, 4].into(),
        Layout::Sparse { axis: None },
        4,
        1 << 20,
        [2.0_f32, 0., 0., 0.],
    )
    .await;
    let (_right_root, right) = fixture::source(
        "sparse_ieee_right",
        vec![1, 4].into(),
        Layout::Sparse { axis: None },
        4,
        1 << 20,
        [0., f32::INFINITY, f32::NAN, 0.],
    )
    .await;
    let expected = [0., f32::NAN, f32::NAN, 0.];
    fixture::reads(
        &left.view().mul(&right.view()).await.unwrap(),
        &expected,
        numbers::same,
    )
    .await;

    let left = TensorExpression::new(left).unwrap().into_dense();
    let right = TensorExpression::new(right).unwrap().into_dense();
    fixture::reads(&left.mul(&right).await.unwrap(), &expected, numbers::same).await;
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn complex_scalar_guards_preserve_zero_and_nonfinite_classifications() {
    use fensor::complex::Complex32 as C;

    let zero = C::new(0., 0.);
    let (_root, sparse) = fixture::source(
        "sparse_complex_policy",
        vec![1, 4].into(),
        Layout::Sparse { axis: None },
        4,
        1 << 20,
        [C::new(2., 1.), zero, zero, zero],
    )
    .await;
    let view = sparse.view();
    for (operation, result) in [
        (
            "add_scalar",
            view.add_scalar(C::new(1., 0.)).await.map(|_| ()),
        ),
        ("div_scalar", view.div_scalar(zero).await.map(|_| ())),
        ("pow_scalar", view.pow_scalar(zero).await.map(|_| ())),
        ("eq_scalar", view.eq_scalar(zero).await.map(|_| ())),
        ("div", view.div(&view).await.map(|_| ())),
    ] {
        rejects_densification(result, operation);
    }
    for scalar in [C::new(f32::INFINITY, 0.), C::new(f32::NAN, 0.)] {
        rejects_densification(view.mul_scalar(scalar).await.map(|_| ()), "mul_scalar");
    }
    let expected = [C::new(4., 2.), zero, zero, zero];
    fixture::reads(
        &view.mul_scalar(C::new(2., 0.)).await.unwrap(),
        &expected,
        numbers::same,
    )
    .await;
    let dense = TensorExpression::new(sparse.clone()).unwrap().into_dense();
    fixture::blocks(
        &dense.mul_scalar(C::new(2., 0.)).await.unwrap(),
        &expected,
        numbers::same,
    )
    .await;
}
