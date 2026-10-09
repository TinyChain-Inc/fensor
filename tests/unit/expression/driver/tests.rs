use super::*;

#[test]
fn poisoned_driver_rejects_polling_and_releases_pending_work() {
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
        budget: Arc::new(Semaphore::new(MAX_LIVE_BATCH_BYTES)),
    };
    let lease = Arc::new(());
    let retained = Arc::downgrade(&lease);
    let child_context = context.clone();
    let reservation = context.reserve::<f64>(1).unwrap();
    *context.pending.lock().unwrap() = Some(Box::pin(async move {
        let _owned = (lease, child_context, reservation);
        std::future::pending::<()>().await;
    }));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = context.pending.lock().unwrap();
            panic!("injected driver failure");
        }))
        .is_err()
    );
    let (_send, receive) = oneshot::channel();
    let mut driver = Driver::<f64> {
        context: context.clone(),
        frames: vec![],
        receive,
    };
    let waker = futures::task::noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut driver).poll(&mut cx),
        Poll::Ready(Err(Error::InvalidLayout(_)))
    ));
    assert!(retained.upgrade().is_some());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _driver = driver;
            panic!("injected outer unwind");
        }))
        .is_err()
    );
    assert!(retained.upgrade().is_none());
    assert_eq!(context.budget.available_permits(), MAX_LIVE_BATCH_BYTES);
    assert!(context.pending.is_poisoned());
    assert_eq!(Arc::strong_count(&context.pending), 1);
}

#[test]
fn batch_admission_follows_retained_values_and_refunds_on_drop() {
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
        budget: Arc::new(Semaphore::new(MAX_LIVE_BATCH_BYTES)),
    };

    let capacity = MAX_LIVE_BATCH_BYTES / (std::mem::size_of::<f64>());
    let reservation = context.reserve::<f64>(capacity).unwrap();
    for len in [1, capacity + 1] {
        assert!(matches!(
            context.reserve::<f64>(len),
            Err(Error::Unsupported(message)) if message == "expression live batch limit exceeded"
        ));
        assert_eq!(context.budget.available_permits(), 0);
    }
    assert!(matches!(
        context.reserve::<f64>(usize::MAX),
        Err(Error::Unsupported(message)) if message == "expression batch allocation overflow"
    ));
    drop(context.reserve::<f64>(0).unwrap());
    assert_eq!(context.budget.available_permits(), 0);
    drop(reservation);
    assert_eq!(context.budget.available_permits(), MAX_LIVE_BATCH_BYTES);

    let mut batch = Batch::from_array(super::super::batch_array(vec![1_f64]).unwrap()).unwrap();
    batch._reservation = Some(context.reserve::<f64>(1).unwrap());

    let values = batch.into_evaluated().unwrap();
    assert_eq!(context.budget.available_permits(), MAX_LIVE_BATCH_BYTES - 8);
    assert_eq!(values.values, [1.]);
    drop(values);
    assert_eq!(context.budget.available_permits(), MAX_LIVE_BATCH_BYTES);
}

#[test]
fn admission_refunds_during_unwind_and_rejects_closed_budget() {
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
        budget: Arc::new(Semaphore::new(MAX_LIVE_BATCH_BYTES)),
    };
    let reservation = context.reserve::<f64>(1).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _reservation = reservation;
            panic!("injected outer unwind");
        }))
        .is_err()
    );
    assert_eq!(context.budget.available_permits(), MAX_LIVE_BATCH_BYTES);
    drop(context.reserve::<f64>(MAX_LIVE_BATCH_BYTES / 8).unwrap());
    assert_eq!(Arc::strong_count(&context.budget), 1);

    context.budget.close();
    for len in [0, 1] {
        assert!(matches!(
            context.reserve::<f64>(len),
            Err(Error::InvalidLayout(message)) if message == "expression admission closed"
        ));
    }
}
