//! One explicit stack of native evaluation continuations. Child requests suspend
//! their caller; only this driver polls them. No nested executor or spawned task.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use futures::channel::oneshot;

use super::{Batch, EvaluatedBatch, Expression};
use crate::request::BatchRequest;
use crate::{BoxFuture, Error, Result, TensorElement};

// Bounds continuation storage independently of the numerical batch bound.
const MAX_FRAMES: usize = 16_384;
const MAX_LIVE_BATCH_BYTES: usize = 16 * 1024 * 1024;

// Bound synchronous progress even when every child and numerical batch is ready.
const READY_WORK_STEPS: usize = 128;

pub(super) struct WorkBudget {
    remaining: usize,
}

impl WorkBudget {
    pub(super) fn new() -> Self {
        Self {
            remaining: READY_WORK_STEPS,
        }
    }

    pub(super) fn poll(&mut self, cx: &mut TaskContext<'_>) -> Poll<()> {
        if self.remaining == 0 {
            self.remaining = READY_WORK_STEPS;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            self.remaining -= 1;
            Poll::Ready(())
        }
    }
}

pub(crate) struct Allocation {
    bytes: usize,
    live: Arc<AtomicUsize>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

type Frame<'a> = BoxFuture<'a, ()>;

#[derive(Clone)]
pub struct Context<'a> {
    pending: Arc<Mutex<Option<Frame<'a>>>>,
    live: Arc<AtomicUsize>,
}

impl<'a> Context<'a> {
    fn reserve<T: TensorElement>(&self, len: usize) -> Result<Allocation> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>() + 1)
            .ok_or_else(|| Error::Unsupported("expression batch allocation overflow".into()))?;
        self.live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes)
                    .filter(|&total| total <= MAX_LIVE_BATCH_BYTES)
            })
            .map_err(|_| Error::Unsupported("expression live batch limit exceeded".into()))?;
        Ok(Allocation {
            bytes,
            live: Arc::clone(&self.live),
        })
    }

    pub async fn batch<E>(
        &self,
        source: &'a E,
        request: std::sync::Arc<BatchRequest>,
    ) -> Result<Batch<E::DType>>
    where
        E: Expression + ?Sized,
        E::DType: TensorElement,
    {
        request.validate(source.shape())?;
        let expected = request.len();
        let allocation = self.reserve::<E::DType>(expected)?;
        let (send, receive) = oneshot::channel();
        let context = self.clone();
        let frame = Box::pin(async move {
            let result = async {
                let mut batch = source.build(context, request).await?;
                batch.validate(expected)?;
                batch._allocation = Some(allocation);
                Ok(batch)
            }
            .await;
            let _ = send.send(result);
        });
        {
            let mut pending = self.pending.lock().expect("evaluation driver poisoned");
            if pending.is_some() {
                return Err(Error::InvalidLayout("concurrent child evaluation".into()));
            }
            *pending = Some(frame as Frame<'a>);
        }
        receive
            .await
            .map_err(|_| Error::InvalidLayout("evaluation driver closed".into()))?
    }

    pub async fn evaluate<E>(
        &self,
        source: &'a E,
        request: std::sync::Arc<BatchRequest>,
    ) -> Result<EvaluatedBatch<E::DType>>
    where
        E: Expression + ?Sized,
        E::DType: TensorElement,
    {
        #[cfg(test)]
        crate::read_metrics::record(|m| m.slice_requests += 1);
        self.batch(source, request).await?.into_evaluated()
    }
}

pub(super) async fn evaluate<E>(
    source: &E,
    request: std::sync::Arc<BatchRequest>,
) -> Result<EvaluatedBatch<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
        live: Arc::new(AtomicUsize::new(0)),
    };

    let root_context = context.clone();
    let (send, receive) = oneshot::channel();
    let root = Box::pin(async move {
        let _ = send.send(root_context.evaluate(source, request).await);
    });
    Driver {
        context,
        frames: vec![root],
        receive,
        work: WorkBudget::new(),
    }
    .await
}

struct Driver<'a, T: TensorElement> {
    context: Context<'a>,
    frames: Vec<Frame<'a>>,
    receive: oneshot::Receiver<Result<EvaluatedBatch<T>>>,
    work: WorkBudget,
}

impl<T: TensorElement> Future for Driver<'_, T> {
    type Output = Result<EvaluatedBatch<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        loop {
            if this.work.poll(cx).is_pending() {
                return Poll::Pending;
            }

            let pending = this
                .context
                .pending
                .lock()
                .expect("evaluation driver poisoned")
                .take();
            if let Some(frame) = pending {
                if this.frames.len() == MAX_FRAMES {
                    return Poll::Ready(Err(Error::Unsupported(
                        "expression evaluation exceeds frame limit".into(),
                    )));
                }

                if let Err(error) = this.frames.try_reserve(1) {
                    return Poll::Ready(Err(Error::Unsupported(format!(
                        "expression frame allocation: {error}"
                    ))));
                }
                this.frames.push(frame);
            }

            let Some(frame) = this.frames.last_mut() else {
                return Pin::new(&mut this.receive).poll(cx).map(|result| {
                    result.unwrap_or_else(|_| {
                        Err(Error::InvalidLayout("evaluation result missing".into()))
                    })
                });
            };

            match frame.as_mut().poll(cx) {
                Poll::Ready(()) => {
                    this.frames.pop();
                }
                Poll::Pending => {
                    if this
                        .context
                        .pending
                        .lock()
                        .expect("evaluation driver poisoned")
                        .is_none()
                    {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

impl<T: TensorElement> Drop for Driver<'_, T> {
    fn drop(&mut self) {
        // A queued child owns a Context clone. Detach before dropping it to break
        // that temporary cycle, including cancellation before the child is polled.
        let pending = self
            .context
            .pending
            .lock()
            .expect("evaluation driver poisoned")
            .take();
        drop(pending);

        while self.frames.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_admission_follows_retained_values_and_refunds_on_drop() {
        let context = Context {
            pending: Arc::new(Mutex::new(None)),
            live: Arc::new(AtomicUsize::new(0)),
        };

        let capacity = MAX_LIVE_BATCH_BYTES / (std::mem::size_of::<f64>() + 1);
        let allocation = context.reserve::<f64>(capacity).unwrap();
        assert!(context.reserve::<f64>(1).is_err());
        assert!(context.reserve::<f64>(usize::MAX).is_err());
        drop(allocation);
        assert_eq!(context.live.load(Ordering::Relaxed), 0);

        let batch = Batch {
            _allocation: Some(context.reserve::<f64>(1).unwrap()),
            array: super::super::batch_array(vec![1_f64]).unwrap(),
            support: Some(vec![1]),
        };

        let values = batch.realize().unwrap().into_evaluated().unwrap();
        assert_eq!(context.live.load(Ordering::Relaxed), 9);
        assert_eq!(values.values, [1.]);
        drop(values);
        assert_eq!(context.live.load(Ordering::Relaxed), 0);
    }
}
