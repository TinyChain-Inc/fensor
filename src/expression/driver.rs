//! A trampoline using an explicit continuation stack. Child requests suspend
//! their caller; only this driver polls them. No nested executor or spawned task.
//! Concrete operations still delegate to operands, as in v1; the trampoline prevents
//! user-controlled expression depth from becoming call-stack depth. Boxing alone
//! does not flatten nested future polling.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, ready};

use futures::channel::oneshot;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::task::coop;

use super::{Batch, EvaluatedBatch, Expression};
use crate::request::BatchRequest;
use crate::{BoxFuture, Error, Result, TensorElement};

// Bounds continuation storage independently of the numerical batch bound.
const MAX_FRAMES: usize = 16_384;
const MAX_LIVE_BATCH_BYTES: usize = 16 * 1024 * 1024;

type Frame<'a> = BoxFuture<'a, ()>;

#[derive(Clone)]
pub struct Context<'a> {
    pending: Arc<Mutex<Option<Frame<'a>>>>,
    // Intermediate arrays are outside the filesystem cache and may outlive frames.
    // Owned permits bound their combined size and refund capacity on cancellation.
    budget: Arc<Semaphore>,
}

impl<'a> Context<'a> {
    fn reserve<T: TensorElement>(&self, len: usize) -> Result<OwnedSemaphorePermit> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::Unsupported("expression batch allocation overflow".into()))?;

        if bytes > MAX_LIVE_BATCH_BYTES {
            return Err(Error::Unsupported(
                "expression live batch limit exceeded".into(),
            ));
        }

        let permits = u32::try_from(bytes)
            .map_err(|_| Error::Unsupported("expression batch allocation overflow".into()))?;
        // Waiting could deadlock: retained operands may be released only after
        // this operation finishes, so admission must reject immediately.
        Arc::clone(&self.budget)
            .try_acquire_many_owned(permits)
            .map_err(|cause| match cause {
                TryAcquireError::NoPermits => {
                    Error::Unsupported("expression live batch limit exceeded".into())
                }
                TryAcquireError::Closed => {
                    Error::InvalidLayout("expression admission closed".into())
                }
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
        let reservation = self.reserve::<E::DType>(expected)?;
        let (send, receive) = oneshot::channel();
        let context = self.clone();
        let frame = Box::pin(async move {
            let result = source.build(context, request).await.and_then(|mut batch| {
                batch.validate(expected)?;
                batch._reservation = Some(reservation);
                Ok(batch)
            });
            let _ = send.send(result);
        });
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| Error::InvalidLayout("evaluation driver poisoned".into()))?;
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

pub(crate) async fn evaluate_batch<E>(
    source: &E,
    request: &BatchRequest,
) -> Result<EvaluatedBatch<E::DType>>
where
    E: Expression + ?Sized,
    E::DType: TensorElement,
{
    let request = Arc::new(request.clone());
    let context = Context {
        pending: Arc::new(Mutex::new(None)),
        budget: Arc::new(Semaphore::new(MAX_LIVE_BATCH_BYTES)),
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
    }
    .await
}

struct Driver<'a, T: TensorElement> {
    context: Context<'a>,
    frames: Vec<Frame<'a>>,
    receive: oneshot::Receiver<Result<EvaluatedBatch<T>>>,
}

impl<T: TensorElement> Future for Driver<'_, T> {
    type Output = Result<EvaluatedBatch<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        loop {
            let progress = ready!(coop::poll_proceed(cx));

            let pending = this
                .context
                .pending
                .lock()
                .map_err(|_| Error::InvalidLayout("evaluation driver poisoned".into()))?
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
                progress.made_progress();
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
                    progress.made_progress();
                }
                Poll::Pending => {
                    if this
                        .context
                        .pending
                        .lock()
                        .map_err(|_| Error::InvalidLayout("evaluation driver poisoned".into()))?
                        .is_none()
                    {
                        return Poll::Pending;
                    }
                    progress.made_progress();
                }
            }
        }
    }
}

impl<T: TensorElement> Drop for Driver<'_, T> {
    fn drop(&mut self) {
        // A queued child owns a Context clone. Detach before dropping it to break
        // that temporary cycle, including cancellation before the child is polled.
        // Poison recovery here only permits destruction; never resume evaluation.
        let pending = self
            .context
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(pending);

        while self.frames.pop().is_some() {}
    }
}

#[cfg(test)]
#[path = "../../tests/unit/expression/driver/tests.rs"]
mod tests;
