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

    fn shape(&self) -> &[u64] {
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
                    // Inject a valid construction block and hold it so the first
                    // update waits on freqfs before another source batch is requested.
                    let blocks = self
                        .destination
                        .read()
                        .await
                        .get_dir("blocks")
                        .unwrap()
                        .clone();
                    blocks
                        .write()
                        .await
                        .create_file(
                            "0".into(),
                            vec![0u8],
                            std::mem::size_of::<FsEntry>() + std::mem::size_of::<Vec<u64>>() + 1,
                        )
                        .await
                        .unwrap();
                    let file = blocks.read().await.get_file("0").unwrap().clone();
                    let guard = file.write_owned::<Vec<u8>>(0).await.unwrap();
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

async fn source() -> (common::Directory, Tensor<FsEntry, u8>) {
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
async fn destination_backpressure_stops_source_consumption() {
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
    assert_eq!(reader.reads.read(), 1, "no source lookahead while writing");
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    assert!(blocks.read().await.get_file("1").is_none());
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
async fn cancellation_and_errors_release_the_active_operation_and_source() {
    let (root, source) = source().await;
    // Cancel while writing or reading; propagate a source error only after the
    // prior write completes; a destination error prevents the next source poll.
    for (fail_read, stall_read, corrupt) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (true, false, true),
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
        let mut guard = Some(futures::select! {
            guard = receive.fuse() => guard.unwrap(),
            _ = copy.as_mut().fuse() => panic!("copy completed before releasing its block"),
        });
        assert!(futures::poll!(copy.as_mut()).is_pending());
        assert_eq!(reader.reads.read(), 1);
        if corrupt {
            guard.as_mut().unwrap().clear();
        }
        if fail_read || stall_read {
            drop(guard.take());
            if corrupt {
                assert!(matches!(
                    copy.as_mut().await,
                    Err(fensor::Error::InvalidLayout(_))
                ));
                assert_eq!(
                    reader.reads.read(),
                    1,
                    "destination failure stops ingestion"
                );
            } else {
                if fail_read {
                    assert!(matches!(
                        copy.as_mut().await,
                        Err(fensor::Error::Unsupported(_))
                    ));
                } else {
                    assert!(futures::poll!(copy.as_mut()).is_pending());
                }
                assert_eq!(reader.reads.read(), 2);
            }
        }
        // The write-cancellation case still owns its destination guard here.
        drop(copy);
        drop(guard);
        assert_eq!(reader.dropped.read(), 1);
        let reads = reader.reads.read();
        assert!(Tensor::<FsEntry, u8>::load(dir.clone()).await.is_err());
        let blocks = dir.read().await.get_dir("blocks").cloned().unwrap();
        assert!(blocks.read().await.get_file("1").is_none());
        let file = blocks.read().await.get_file("0").cloned().unwrap();
        assert!(file.try_write::<Vec<u8>>(0).is_ok());
        assert_eq!(reader.reads.read(), reads);
        common::cleanup(&out_root).await;
    }
    common::cleanup(&root).await;
}
