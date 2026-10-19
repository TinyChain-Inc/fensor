use std::cell::Cell;
use std::sync::Arc;

use super::coordinate_batches;
use crate::expression::MAX_BATCH_ELEMENTS;

#[test]
fn batching_only_consumes_its_bounded_prefix() {
    let consumed = Cell::new(0);
    let coords = (0..1_000_000_000u64).map(|i| {
        consumed.set(consumed.get() + 1);
        vec![i]
    });
    let mut batches = coordinate_batches(coords);

    assert_eq!(consumed.get(), 0);
    assert_eq!(batches.next().unwrap().len(), MAX_BATCH_ELEMENTS);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);

    drop(batches);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
}

// Observe constructor work without involving filesystem storage or evaluation.
struct ConstructorProbe {
    clones: Arc<crate::test_support::counters::Counter>,
    inspections: Arc<crate::test_support::counters::Counter>,
}

impl Clone for ConstructorProbe {
    fn clone(&self) -> Self {
        self.clones.increment();
        Self {
            clones: Arc::clone(&self.clones),
            inspections: Arc::clone(&self.inspections),
        }
    }
}

impl super::TensorGeometry for ConstructorProbe {
    type DType = f64;

    fn dtype(&self) -> crate::NumberType {
        <f64 as number_general::DType>::dtype()
    }

    fn shape(&self) -> &[u64] {
        self.inspections.increment();
        &[1, 1]
    }

    fn layout(&self) -> crate::Layout {
        self.inspections.increment();
        crate::Layout::Dense
    }
}

impl crate::expression::traversal::Plan for ConstructorProbe {}

impl crate::expression::Expression for ConstructorProbe {
    fn build<'a>(
        &'a self,
        _: crate::expression::Context<'a>,
        _: std::sync::Arc<crate::request::BatchRequest>,
    ) -> crate::BoxFuture<'a, crate::Result<crate::expression::Batch<f64>>> {
        panic!("constructors must not evaluate their source")
    }
}

#[tokio::test]
async fn constructor_futures_are_send_and_defer_work_until_polled() {
    use crate::test_support::counters::Counter;
    use crate::{TensorCast, TensorUnary};

    fn send(future: impl std::future::Future + Send) {
        drop(future);
    }

    // Check the public generic contracts, not just inferred concrete futures.
    fn real<S>(source: &S)
    where
        S: super::TensorGeometry<DType = f64>
            + super::TensorUnary
            + super::TensorTrig
            + super::TensorAbs
            + super::TensorMath<S>
            + super::TensorMathScalar
            + super::TensorCompare<S>
            + super::TensorCompareScalar
            + super::TensorBoolean<S>
            + super::TensorBooleanScalar
            + super::TensorCast<u8>
            + super::TensorReduce
            + crate::TensorStatistics
            + super::TensorMatrixUnary
            + super::TensorMatMul<S>,
    {
        send(source.exp());
        send(source.round());
        send(source.sin());
        send(source.abs());
        send(source.add(source));
        send(source.add_scalar(1.));
        send(source.gt(source));
        send(source.gt_scalar(1.));
        send(source.and(source));
        send(source.and_scalar(1.));
        send(source.cast());
        send(source.sum(ha_ndarray::axes![0], false));
        send(source.min(ha_ndarray::axes![0], true));
        send(source.mean(ha_ndarray::axes![0], false));
        send(source.std(ha_ndarray::axes![0], false));
        send(source.norm(ha_ndarray::axes![0], false));
        send(source.mt());
        send(source.diag());
        send(source.matmul(source));
    }

    fn boolean<S: super::TensorUnaryBoolean + super::TensorWhere<S, S>>(source: &S) {
        send(source.not());
        send(source.cond(source, source));
    }

    let clones = Arc::new(Counter::new());
    let inspections = Arc::new(Counter::new());
    let source = ConstructorProbe {
        clones: Arc::clone(&clones),
        inspections: Arc::clone(&inspections),
    };
    let future = source.exp();
    assert_eq!((clones.read(), inspections.read()), (0, 0));
    send(future);
    assert_eq!((clones.read(), inspections.read()), (0, 0));

    let view = source.exp().await.unwrap();
    assert_eq!(clones.read(), 1);
    assert!(inspections.read() > 0);
    let view = crate::TensorExpression::new(view).unwrap();
    clones.reset();
    inspections.reset();
    real(&view);
    assert_eq!((clones.read(), inspections.read()), (0, 0));

    let mask = <_ as TensorCast<u8>>::cast(&view).await.unwrap();
    boolean(&mask);

    #[cfg(feature = "complex")]
    {
        fn complex<S>(source: &S)
        where
            S: super::TensorFourier + super::TensorMatrixUnaryComplex + super::TensorComplex,
            S::DType: ha_ndarray::Complex,
        {
            send(source.fft());
            send(source.ifft());
            send(source.mh());
            send(source.conj());
        }

        let value = <_ as TensorCast<ha_ndarray::complex::Complex64>>::cast(&view)
            .await
            .unwrap();
        complex(&value);
    }
}
