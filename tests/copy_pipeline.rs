//! Controlled stream readiness over real filesystem-backed tensors.

use std::sync::Mutex;
use std::task::Poll;

use fensor::{
    CoordinateBlockStream, Layout, Tensor, TensorGeometry, TensorRead, TensorSchema, TensorWrite,
};
use freqfs::{DirLock, FileWriteGuardOwned};
use futures::{FutureExt, StreamExt, channel::oneshot};
use ha_ndarray::shape;
use number_general::{DType, NumberType};

mod common;
use common::{FsEntry, counters::Counter};

type GuardSender = oneshot::Sender<FileWriteGuardOwned<FsEntry, Vec<u8>>>;

struct Reader<'a> {
    source: &'a Tensor<FsEntry, u8>,
    destination: DirLock<FsEntry>,
    guard: Mutex<Option<GuardSender>>,
    reads: Counter,
    dropped: Counter,
    fail_read: bool,
    stall_read: bool,
}

struct OnDrop<'a>(&'a Counter);

impl Drop for OnDrop<'_> {
    fn drop(&mut self) {
        self.0.increment();
    }
}

impl TensorGeometry for Reader<'_> {
    type DType = u8;

    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[usize] {
        self.source.shape()
    }

    fn layout(&self) -> Layout {
        self.source.layout()
    }
}

impl TensorRead for Reader<'_> {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> fensor::BoxFuture<'a, fensor::Result<u8>> {
        self.source.read_value(coord)
    }

    fn read_coordinate_blocks(&self) -> fensor::Result<CoordinateBlockStream<'_, u8>> {
        Ok(futures::stream::try_unfold(
            (0, OnDrop(&self.dropped)),
            move |(index, drop)| async move {
                self.reads.increment();
                if index == 3 {
                    return Ok(None);
                }
                if index == 0 {
                    // Creation is complete before the first source poll. Hold the real
                    // destination block so the first update must wait on freqfs.
                    let blocks = self
                        .destination
                        .read()
                        .await
                        .get_dir("blocks")
                        .unwrap()
                        .clone();
                    let file = blocks.read().await.get_file("0").unwrap().clone();
                    let guard = file.write_owned::<Vec<u8>>().await.unwrap();
                    self.guard
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap()
                        .send(guard)
                        .unwrap();
                } else if self.fail_read {
                    return Err(fensor::Error::Unsupported("injected source error".into()));
                } else if self.stall_read {
                    futures::future::pending::<()>().await;
                }
                let value = self.source.read_value(&[index]).await?;
                Ok(Some(((vec![vec![index]], vec![value]), (index + 1, drop))))
            },
        )
        .boxed())
    }
}

async fn source() -> (std::path::PathBuf, Tensor<FsEntry, u8>) {
    let (root, dir) = common::new_dir("pipeline_source").await;
    let tensor = Tensor::create(
        dir,
        TensorSchema::new(u8::dtype(), shape![3]).unwrap(),
        Layout::Dense,
        1,
    )
    .await
    .unwrap();
    for i in 0..3 {
        tensor.write_value(&[i], i as u8 + 1).await.unwrap();
    }
    (root, tensor)
}

#[tokio::test]
async fn lookahead_progresses_but_writes_and_buffering_stay_bounded() {
    let (root, source) = source().await;
    let (out_root, dir) = common::new_dir("pipeline_output").await;
    let (send, receive) = oneshot::channel();
    let reader = Reader {
        source: &source,
        destination: dir.clone(),
        guard: Mutex::new(Some(send)),
        reads: Counter::new(),
        dropped: Counter::new(),
        fail_read: false,
        stall_read: false,
    };
    let mut copy = Box::pin(Tensor::copy_from(dir.clone(), &reader, 1));
    let guard = futures::select! {
        guard = receive.fuse() => guard.unwrap(),
        _ = copy.as_mut().fuse() => panic!("copy completed before releasing its block"),
    };
    assert!(matches!(futures::poll!(copy.as_mut()), Poll::Pending));
    assert_eq!(reader.reads.read(), 2, "one completed lookahead only");
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let second = blocks.read().await.get_file("1").unwrap().clone();
    assert_eq!(&*second.read::<Vec<u8>>().await.unwrap(), &[0]);
    drop(guard);
    let output = copy.await.unwrap();
    assert_eq!(
        reader.reads.read(),
        4,
        "three blocks and exactly one EOF poll"
    );
    assert_eq!(reader.dropped.read(), 1);
    output.sync().await.unwrap();
    drop(output);
    let loaded = Tensor::<FsEntry, u8>::load(common::open_dir(&out_root).unwrap())
        .await
        .unwrap();
    for i in 0..3 {
        assert_eq!(loaded.read_value(&[i]).await.unwrap(), i as u8 + 1);
    }
    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}

#[tokio::test]
async fn cancellation_and_errors_drop_the_other_side() {
    let (root, source) = source().await;
    // Drop while reading/writing, source error while writing, destination error
    // while the next read is pending. No timing or global-counter assertions.
    for (fail_read, stall_read, corrupt) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (false, true, true),
    ] {
        let (out_root, dir) = common::new_dir("pipeline_cancel").await;
        let (send, receive) = oneshot::channel();
        let reader = Reader {
            source: &source,
            destination: dir.clone(),
            guard: Mutex::new(Some(send)),
            reads: Counter::new(),
            dropped: Counter::new(),
            fail_read,
            stall_read,
        };
        let mut copy = Box::pin(Tensor::copy_from(dir.clone(), &reader, 1));
        // Drive only until the first block has been locked; the source error may
        // be returned by the same poll, so retain that result for the error case.
        let (mut guard, first) = match futures::future::select(receive, copy.as_mut()).await {
            futures::future::Either::Left((guard, _)) => (guard.unwrap(), None),
            futures::future::Either::Right((result, receive)) => {
                (receive.await.unwrap(), Some(result))
            }
        };
        if fail_read {
            let result = match first {
                Some(result) => result,
                None => copy.as_mut().await,
            };
            assert!(matches!(result, Err(fensor::Error::Unsupported(_))));
        } else {
            assert!(first.is_none());
            assert!(futures::poll!(copy.as_mut()).is_pending());
            assert_eq!(reader.reads.read(), 2);
        }
        if corrupt {
            guard.clear();
            drop(guard);
            assert!(matches!(
                copy.as_mut().await,
                Err(fensor::Error::InvalidLayout(_))
            ));
        } else {
            drop(guard);
        }
        drop(copy);
        assert_eq!(reader.dropped.read(), 1);
        let reads = reader.reads.read();
        let output = Tensor::<FsEntry, u8>::load(dir).await.unwrap();
        assert_eq!(output.read_value(&[1]).await.unwrap(), 0);
        assert_eq!(reader.reads.read(), reads);
        common::cleanup(&out_root).await;
    }
    common::cleanup(&root).await;
}
