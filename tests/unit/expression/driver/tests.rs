use super::*;

#[test]
fn poisoned_driver_rejects_polling_and_releases_pending_work() {
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
    };
    let lease = Arc::new(());
    let retained = Arc::downgrade(&lease);
    let child_context = context.clone();
    *context.pending.lock().unwrap() = Some(Box::pin(async move {
        let _owned = (lease, child_context);
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
    assert!(context.pending.is_poisoned());
    assert_eq!(Arc::strong_count(&context.pending), 1);
}
