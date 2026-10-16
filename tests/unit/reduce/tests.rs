use std::cell::RefCell;

use number_general::DType;

use super::*;
use crate::read_metrics::{CURRENT, Metrics};
use crate::test_support::{FsEntry, new_dir};
use crate::{Tensor, TensorExpression, TensorMath, TensorMathScalar, TensorSchema, TensorWrite};

#[tokio::test]
async fn zero_gap_products_preserve_expression_signed_zeros() {
    for len in [3, expression::MAX_BATCH_ELEMENTS as u64 + 1] {
        let (_root, dir) = new_dir("reduce_signed_zero").await;
        let tensor = Tensor::<FsEntry, f64>::create(
            dir,
            TensorSchema::new(f64::dtype(), vec![1, len].into()).unwrap(),
            Layout::Sparse { axis: None },
            31,
        )
        .await
        .unwrap();

        for first in [0., 1.] {
            tensor.write_value(&[0, 0], first).await.unwrap();
            let sparse = tensor.view().mul_scalar(-1.).await.unwrap();
            let owned = TensorExpression::new(sparse.clone()).unwrap();
            let dense = owned.clone().into_dense();
            let expected = dense.product_all().await.unwrap().to_bits();
            assert_eq!(expected, (-0.0_f64).to_bits());
            assert_eq!(sparse.product_all().await.unwrap().to_bits(), expected);
            assert_eq!(owned.product_all().await.unwrap().to_bits(), expected);
            assert_eq!(
                owned
                    .product(ha_ndarray::axes![1], false)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap()
                    .to_bits(),
                expected
            );
        }
    }
}

#[tokio::test]
async fn indexed_reductions_account_for_implicit_zero_gaps() {
    let (_root, dir) = new_dir("reduce_zero_gaps").await;
    let len = 1_000_000_007;
    let tensor = Tensor::<FsEntry, f64>::create(
        dir,
        TensorSchema::new(f64::dtype(), vec![1, len].into()).unwrap(),
        Layout::Sparse { axis: None },
        31,
    )
    .await
    .unwrap();
    tensor.write_value(&[0, 1], 2.).await.unwrap();
    tensor.write_value(&[0, len - 1], 3.).await.unwrap();

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        CURRENT.scope(RefCell::new(Metrics::default()), async {
            assert_eq!(tensor.sum_all().await.unwrap(), 5.);
            assert_eq!(tensor.product_all().await.unwrap(), 0.);
            assert_eq!(tensor.min_all().await.unwrap(), 0.);
            assert_eq!(tensor.max_all().await.unwrap(), 3.);
            assert_eq!(tensor.mean_all().await.unwrap(), 5. / len as f64);
            let variance = 13. / len as f64 - (5. / len as f64).powi(2);
            assert!((tensor.std_all().await.unwrap() - variance.sqrt()).abs() < 1e-12);
            assert_eq!(tensor.norm_all().await.unwrap(), 13.0_f64.sqrt());
            assert_eq!(
                tensor
                    .view()
                    .sum(ha_ndarray::axes![1], false)
                    .await
                    .unwrap()
                    .read_value(&[0])
                    .await
                    .unwrap(),
                5.
            );
            let zeros = tensor.view().sub(&tensor.view()).await.unwrap();
            assert_eq!(zeros.mean_all().await.unwrap(), 0.);
            CURRENT.with(|metrics| {
                let metrics = metrics.borrow();
                assert!(metrics.requested > 0 && metrics.requested <= 24);
                assert!(metrics.reduction_calls <= 18);
            });
        }),
    )
    .await
    .expect("indexed reductions must not scan the implicit-zero gap");

    tensor.write_value(&[0, 1], f64::INFINITY).await.unwrap();
    assert!(tensor.product_all().await.unwrap().is_nan());
    assert!(tensor.std_all().await.unwrap().is_nan());
    assert_eq!(tensor.mean_all().await.unwrap(), f64::INFINITY);
    assert_eq!(tensor.norm_all().await.unwrap(), f64::INFINITY);

    let mut state = None;
    let mut remaining = 1;
    assert!(accumulate(&Sum, &mut state, &mut remaining, vec![1., 2.]).is_err());
    assert_eq!(remaining, 1);
    assert_eq!(state, None);
}
