//! User-controlled composition must not become native worker-stack depth.

use std::future::Future;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fensor::{
    BoxFuture, Layout, NumberType, StorageGeometry, Tensor, TensorAbs, TensorArray,
    TensorExpression, TensorGeometry, TensorMath, TensorMathScalar, TensorRead, TensorReduce,
    TensorSchema, TensorSource, TensorTransform, TensorView,
};
use futures::TryStreamExt;

mod common;

const CASE_ENV: &str = "FENSOR_EXPRESSION_DEPTH_CASE";
const TEST: &str = "owned_expression_depth_is_independent_of_the_worker_stack";

#[derive(Clone)]
struct Source {
    tensor: Tensor<common::FsEntry, f64>,
    lease: Arc<AtomicUsize>,
    behavior: &'static str,
}

impl TensorGeometry for Source {
    type DType = f64;

    fn dtype(&self) -> NumberType {
        self.tensor.dtype()
    }

    fn layout(&self) -> Layout {
        self.tensor.layout()
    }

    fn shape(&self) -> &[u64] {
        self.tensor.shape()
    }
}

impl TensorArray for Source {
    fn schema(&self) -> &TensorSchema {
        self.tensor.schema()
    }

    fn strides(&self) -> &[u64] {
        self.tensor.strides()
    }
}

impl TensorSource for Source {
    fn storage_geometry(&self) -> StorageGeometry {
        self.tensor.storage_geometry()
    }

    fn read_logical_block(&self, id: u64) -> BoxFuture<'_, fensor::Result<Vec<f64>>> {
        Box::pin(async move {
            self.lease.fetch_add(1, Ordering::Relaxed);
            match self.behavior {
                "cancel" => std::future::pending().await,
                "ready" => Ok(vec![0., -2.]),
                "error" => Err(fensor::Error::InvalidLayout(
                    "injected deep leaf failure".into(),
                )),
                _ => self.tensor.read_logical_block(id).await,
            }
        })
    }

    fn occupied_blocks(
        &self,
        range: std::ops::Range<u64>,
    ) -> futures::stream::BoxStream<'static, fensor::Result<u64>> {
        self.tensor.occupied_blocks(range)
    }
}

async fn exercise(case: &str) {
    let layout = if matches!(case, "sparse" | "mixed-sparse") {
        Layout::Sparse { axis: None }
    } else {
        Layout::Dense
    };
    let (_root, tensor) = common::fixture::source(
        "expression_depth",
        vec![2].into(),
        layout,
        2,
        1_000_000,
        [0., -2.],
    )
    .await;
    let dtype = tensor.dtype();
    let lease = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&lease);
    let behavior = match case {
        "cancel" => "cancel",
        "ready-cancel" => "ready",
        "error" => "error",
        _ => "read",
    };
    let source = Source {
        tensor,
        lease,
        behavior,
    };
    let mut expression = TensorExpression::new(TensorView::new(source)).unwrap();
    if case == "wide" {
        let leaf = expression
            .reshape(vec![1, 2].into())
            .unwrap()
            .broadcast(vec![2048, 2].into())
            .unwrap();
        expression = leaf.clone();
        for _ in 0..1024 {
            expression = TensorExpression::new(leaf.add(&expression).await.unwrap()).unwrap();
        }
        drop(leaf);
        let mut stream = expression.into_blocks().unwrap();
        assert!(matches!(stream.try_next().await,
            Err(fensor::Error::Unsupported(message)) if message.contains("live batch limit")));
        drop(stream);
        assert!(
            weak.upgrade().is_none(),
            "batch admission failure retained its source"
        );
        eprintln!("completed {case}");
        return;
    }
    let depth = match case {
        "drop" | "unpolled" | "concurrent-drop" => 4096,
        "limit" => 20_000,
        "construction-limit" => 40_000,
        _ => 1024,
    };
    for i in 0..depth {
        let next = if case == "reduce" {
            TensorExpression::new(expression.sum(ha_ndarray::axes![0], true).await.unwrap())
        } else if case.starts_with("mixed") && i % 2 == 0 {
            TensorExpression::new(
                expression
                    .mul_scalar(2.)
                    .await
                    .unwrap()
                    .div_scalar(2.)
                    .await
                    .unwrap(),
            )
            .and_then(|value| value.flip(0)?.flip(0))
        } else {
            TensorExpression::new(expression.abs().await.unwrap())
        };
        match next {
            Ok(next) => expression = next,
            Err(fensor::Error::Unsupported(_)) if case == "construction-limit" => {
                drop(expression);
                assert!(
                    weak.upgrade().is_none(),
                    "rejected construction retained its source"
                );
                eprintln!("completed {case} after {i} operations");
                return;
            }
            Err(error) => panic!("construct {case} at {i}: {error}"),
        }
    }
    assert_ne!(
        case, "construction-limit",
        "expression resource bound was not enforced"
    );
    eprintln!("constructed {case} depth={depth}");
    assert_eq!(
        expression.shape(),
        if case == "reduce" { &[1][..] } else { &[2][..] }
    );
    assert_eq!(expression.layout(), layout);
    assert_eq!(expression.dtype(), dtype);
    eprintln!("metadata complete");
    match case {
        "reduce" => {
            assert_eq!(expression.read_value(&[0]).await.unwrap(), -2.);
            assert_eq!(
                expression
                    .into_blocks()
                    .unwrap()
                    .try_concat()
                    .await
                    .unwrap(),
                [-2.]
            );
        }
        "drop" => drop(expression),
        "unpolled" => drop(expression.into_blocks().unwrap()),
        "concurrent-drop" => {
            let barrier = Arc::new(tokio::sync::Barrier::new(2));
            let other_barrier = Arc::clone(&barrier);
            let other = expression.clone();
            let task = tokio::spawn(async move {
                other_barrier.wait().await;
                drop(other);
            });
            barrier.wait().await;
            drop(expression);
            task.await.unwrap();
        }
        "ready-cancel" => {
            // This leaf never awaits storage. A pending poll must therefore be
            // a cooperative yield, including while numerical ancestors are ready.
            let mut stream = expression.into_blocks().unwrap();
            {
                let mut next = std::pin::pin!(stream.try_next());
                let mut polls = 0;
                futures::future::poll_fn(|cx| {
                    polls += 1;
                    assert!(next.as_mut().poll(cx).is_pending());
                    if weak.upgrade().unwrap().load(Ordering::Relaxed) > 0 {
                        Poll::Ready(())
                    } else {
                        Poll::Pending
                    }
                })
                .await;
                assert!(polls > 1, "ready descent monopolized one poll");
            }
            drop(stream);
        }
        "cancel" => {
            let mut stream = expression.into_blocks().unwrap();
            {
                let mut next = std::pin::pin!(stream.try_next());
                futures::future::poll_fn(|cx| {
                    assert!(next.as_mut().poll(cx).is_pending());
                    if weak.upgrade().unwrap().load(Ordering::Relaxed) > 0 {
                        Poll::Ready(())
                    } else {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                })
                .await;
            }
            drop(stream);
        }
        "limit" => {
            assert!(matches!(
                expression.read_value(&[1]).await,
                Err(fensor::Error::Unsupported(_))
            ));
            assert_eq!(weak.upgrade().unwrap().load(Ordering::Relaxed), 0);
            drop(expression);
        }
        "error" => {
            let mut stream = expression.into_blocks().unwrap();
            assert!(matches!(stream.try_next().await,
                Err(fensor::Error::InvalidLayout(message)) if message == "injected deep leaf failure"));
            drop(stream);
        }
        "sparse" => {
            // A deep intermediate can become zero; another operand supplies its
            // ordinary value without turning the source's implicit zero nonzero.
            let zeros = expression.sub(&expression).await.unwrap();
            let transformed = TensorExpression::new(
                zeros
                    .add(&expression.mul_scalar(1.5).await.unwrap())
                    .await
                    .unwrap(),
            )
            .unwrap();
            drop(zeros);
            drop(expression);
            assert_eq!(transformed.read_value(&[0]).await.unwrap(), 0.);
            let entries: Vec<_> = transformed
                .into_sparse_elements()
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(entries, [(vec![1], 3.)]);
        }
        "dense" | "mixed" | "mixed-sparse" => {
            let (expression, expected) = if case.starts_with("mixed") {
                // Both operands retain the same deep graph.
                let shared =
                    TensorExpression::new(expression.add(&expression).await.unwrap()).unwrap();
                drop(expression);
                (shared, 4.)
            } else {
                (expression, 2.)
            };
            assert_eq!(expression.read_value(&[1]).await.unwrap(), expected);
            if case == "mixed-sparse" {
                let entries: Vec<_> = expression
                    .into_sparse_elements()
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                assert_eq!(entries, [(vec![1], expected)]);
            } else {
                assert_eq!(
                    expression
                        .into_blocks()
                        .unwrap()
                        .try_concat()
                        .await
                        .unwrap(),
                    [0., expected]
                );
            }
        }
        _ => panic!("unknown depth case {case}"),
    }
    assert!(
        weak.upgrade().is_none(),
        "source lease retained after {case}"
    );
    eprintln!("completed {case}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn owned_expression_depth_is_independent_of_the_worker_stack() {
    assert!(
        std::env::var_os("RUST_MIN_STACK").is_none(),
        "normal worker stacks required"
    );
    if let Ok(case) = std::env::var(CASE_ENV) {
        tokio::spawn(async move { exercise(&case).await })
            .await
            .unwrap();
        return;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let logs = std::env::temp_dir().join(format!(
        "fensor-expression-depth-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&logs).unwrap();
    for case in [
        "dense",
        "sparse",
        "mixed",
        "mixed-sparse",
        "reduce",
        "wide",
        "drop",
        "concurrent-drop",
        "unpolled",
        "cancel",
        "ready-cancel",
        "error",
        "limit",
        "construction-limit",
    ] {
        let log = logs.join(format!("{case}.log"));
        let output = std::fs::File::create(&log).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env(CASE_ENV, case)
            .stdin(Stdio::null())
            .stderr(output.try_clone().unwrap())
            .stdout(output)
            .spawn()
            .unwrap();
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > Duration::from_secs(60) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("{case} timed out; log: {}", log.display());
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let transcript = std::fs::read_to_string(&log).unwrap();
        assert!(
            status.success()
                && transcript.contains(&format!("completed {case}"))
                && transcript.contains("1 passed"),
            "{case}: {status}; log: {}\n{}",
            log.display(),
            transcript
        );
    }
    std::fs::remove_dir_all(logs).unwrap();
}
