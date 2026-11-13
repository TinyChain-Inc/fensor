//! One storage constructor consumes base tensors and geometric views as readers.

use common::{FsEntry, new_dir};
use fensor::{
    AxisRange, Error, Layout, Tensor, TensorArray, TensorGeometry, TensorRead, TensorSchema,
    TensorSource, TensorTransform, TensorWrite,
};
use ha_ndarray::{axes, range, shape};
use number_general::{FloatType, NumberType, UIntType};

mod common;

#[tokio::test]
async fn copy_base_and_geometric_readers_into_independent_storage() {
    for layout in [Layout::Dense, Layout::Sparse { axis: Some(1) }] {
        let (root, dir) = new_dir("copy_source").await;
        let source = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(NumberType::UInt(UIntType::U8), shape![2, 3]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();

        source.write_value(&[1, 2], 255).await.unwrap();
        let (base_root, base_dir) = new_dir("copy_base").await;
        let copied = Tensor::copy_from(base_dir, &source, source.layout(), 3)
            .await
            .unwrap();

        assert_eq!(copied.schema(), source.schema());
        assert_eq!(copied.layout(), layout);
        let view = source.view().transpose(Some(axes![1, 0])).unwrap();
        let (view_root, view_dir) = new_dir("copy_view").await;
        let destination_layout = match layout {
            Layout::Dense => Layout::Dense,
            Layout::Sparse { .. } => Layout::Sparse { axis: Some(0) },
        };

        let transposed = Tensor::copy_from(view_dir, &view, destination_layout, 2)
            .await
            .unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.layout(), destination_layout);
        assert_eq!(transposed.storage_geometry().block_capacity(), 2);

        for coord in common::iter_coords(&[3, 2]) {
            assert_eq!(
                transposed.read_value(&coord).await.unwrap(),
                view.read_value(&coord).await.unwrap()
            );
        }

        source.write_value(&[1, 2], 7).await.unwrap();
        assert_eq!(copied.read_value(&[1, 2]).await.unwrap(), 255);
        assert_eq!(transposed.read_value(&[2, 1]).await.unwrap(), 255);
        copied.write_value(&[1, 2], 3).await.unwrap();
        assert_eq!(source.read_value(&[1, 2]).await.unwrap(), 7);

        for root in [&root, &base_root, &view_root] {
            common::cleanup(root).await;
        }
    }
}

#[tokio::test]
async fn copy_propagates_source_and_destination_errors() {
    let (root, dir) = new_dir("copy_errors").await;
    let source = Tensor::<FsEntry, f32>::create(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();

    let (out_root, out_dir) = new_dir("copy_errors_out").await;
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &source, Layout::Sparse { axis: None }, 2).await,
        Err(Error::Unsupported(_))
    ));

    assert!(out_dir.try_read().unwrap().is_empty());
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &source, source.layout(), 0).await,
        Err(Error::InvalidSchema(_))
    ));

    let scalar = source.view().slice(range![AxisRange::At(0)]).unwrap();
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &scalar, scalar.layout(), 2).await,
        Err(Error::InvalidSchema(_))
    ));

    assert!(!out_root.join("blocks").exists());
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete("0").await;
    assert!(
        Tensor::copy_from(out_dir, &source, source.layout(), 2)
            .await
            .is_err()
    );

    common::cleanup(&root).await;
    common::cleanup(&out_root).await;
}

// Deliberately malformed public reader over real filesystem storage, not a
// substitute tensor implementation: exercise the untrusted stream boundary.
struct CoordinateReader<'a> {
    tensor: &'a Tensor<FsEntry, u8>,
    blocks: Vec<(Vec<Vec<u64>>, Vec<u8>)>,
}

impl TensorGeometry for CoordinateReader<'_> {
    type DType = u8;

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

impl TensorRead for CoordinateReader<'_> {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> fensor::BoxFuture<'a, fensor::Result<u8>> {
        self.tensor.read_value(coord)
    }

    fn read_coordinate_blocks(&self) -> fensor::Result<fensor::CoordinateBlockStream<'_, u8>> {
        use futures::StreamExt;
        Ok(futures::stream::iter(self.blocks.clone().into_iter().map(Ok)).boxed())
    }
}

#[tokio::test]
async fn coordinate_copy_validates_blocks_and_accepts_unordered_coverage() {
    let (_dir_root, dir) = new_dir("coordinate_copy_source").await;
    let tensor = Tensor::<FsEntry, u8>::create(
        dir,
        TensorSchema::new(NumberType::UInt(UIntType::U8), shape![2]).unwrap(),
        Layout::Dense,
        1,
    )
    .await
    .unwrap();

    for blocks in [
        vec![(vec![vec![0]], vec![])],
        vec![(vec![vec![2]], vec![1])],
        vec![(vec![vec![0]; 4097], vec![1; 4097])],
        vec![(vec![vec![0]], vec![1])],
        vec![(vec![vec![0], vec![1], vec![0]], vec![1, 2, 3])],
    ] {
        let (_dir_root, dir) = new_dir("coordinate_copy_invalid").await;
        assert!(
            Tensor::<FsEntry, u8>::copy_from(
                dir,
                &CoordinateReader {
                    tensor: &tensor,
                    blocks
                },
                Layout::Dense,
                1
            )
            .await
            .is_err()
        );
    }

    let (_dir_root, dir) = new_dir("coordinate_copy_valid").await;
    let copy = Tensor::<FsEntry, u8>::copy_from(
        dir,
        &CoordinateReader {
            tensor: &tensor,
            blocks: vec![(vec![vec![1], vec![0]], vec![3, 2])],
        },
        Layout::Dense,
        1,
    )
    .await
    .unwrap();

    assert_eq!(copy.read_value(&[0]).await.unwrap(), 2);
    assert_eq!(copy.read_value(&[1]).await.unwrap(), 3);
}

async fn copy_boundary_values<T: fensor::TensorElement>(pattern: &[T], equal: impl Fn(T, T) -> bool)
where
    FsEntry: fensor::TensorFileEntry<T>,
{
    use futures::TryStreamExt;

    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, dir) = new_dir("copy_boundary_source").await;
        let source = Tensor::<FsEntry, T>::create(
            dir,
            TensorSchema::new(<T as number_general::DType>::dtype(), shape![17, 241]).unwrap(),
            layout,
            31,
        )
        .await
        .unwrap();

        for i in 0..4097 {
            let value = pattern[i as usize % pattern.len()];
            if matches!(layout, Layout::Dense) || value != T::ZERO {
                source
                    .write_value(&[i / 241, i % 241], value)
                    .await
                    .unwrap();
            }
        }

        let view = source.view().transpose(None).unwrap();
        let (out_root, out_dir) = new_dir("copy_boundary_out").await;
        drop(out_dir);
        let out_dir =
            freqfs::Cache::<FsEntry>::new(65_536, None, 0, std::time::Duration::from_secs(1))
                .load(out_root.to_path_buf())
                .unwrap();

        let (other_root, other_dir) = new_dir("copy_concurrent_out").await;
        let (copied, other) = if matches!(layout, Layout::Sparse { .. }) {
            let (_unsupported_root, unsupported_dir) = new_dir("copy_unsupported_order").await;
            assert!(matches!(
                Tensor::copy_from(unsupported_dir, &view, view.layout(), 128).await,
                Err(Error::Unsupported(_))
            ));

            futures::try_join!(
                Tensor::copy_from_values(out_dir, &view, view.layout(), 128),
                Tensor::copy_from_values(other_dir, &view, view.layout(), 7),
            )
        } else {
            futures::try_join!(
                Tensor::copy_from(out_dir, &view, view.layout(), 128),
                Tensor::copy_from(other_dir, &view, view.layout(), 7),
            )
        }
        .unwrap();

        copied.sync().await.unwrap();
        drop(copied);
        let dir = freqfs::Cache::<FsEntry>::new(65_536, None, 0, std::time::Duration::from_secs(1))
            .load(out_root.to_path_buf())
            .unwrap();

        let copied = Tensor::<FsEntry, T>::load(dir).await.unwrap();
        let mut blocks = copied.read_blocks().unwrap();
        let mut i = 0;

        while let Some(values) = blocks.try_next().await.unwrap() {
            for value in values {
                let expected = pattern[((i % 17) * 241 + i / 17) % pattern.len()];
                let expected = if matches!(layout, Layout::Sparse { .. }) && expected == T::ZERO {
                    T::ZERO
                } else {
                    expected
                };

                assert!(equal(value, expected), "copy mismatch at {i}");
                i += 1;
            }
        }

        assert_eq!(i, 4097);
        assert!(equal(
            other.read_value(&[240, 16]).await.unwrap(),
            pattern[4096 % pattern.len()]
        ));

        for path in [&root, &out_root, &other_root] {
            common::cleanup(path).await;
        }
    }
}

#[tokio::test]
async fn grouped_copy_preserves_float_special_values_and_u8_boundaries() {
    copy_boundary_values(&[0u8, 1, 127, 255], |a, b| a == b).await;
    copy_boundary_values(
        &[0f32, -0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.],
        |a, b| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
    )
    .await;

    copy_boundary_values(
        &[0f64, -0., f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.],
        |a, b| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
    )
    .await;
}

struct PausedReader<'a>(&'a Tensor<FsEntry, u8>);

impl TensorGeometry for PausedReader<'_> {
    type DType = u8;

    fn dtype(&self) -> NumberType {
        self.0.dtype()
    }

    fn layout(&self) -> Layout {
        self.0.layout()
    }

    fn shape(&self) -> &[u64] {
        self.0.shape()
    }
}

impl TensorRead for PausedReader<'_> {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> fensor::BoxFuture<'a, fensor::Result<u8>> {
        self.0.read_value(coord)
    }

    fn read_coordinate_blocks(&self) -> fensor::Result<fensor::CoordinateBlockStream<'_, u8>> {
        use futures::StreamExt;
        Ok(self
            .0
            .read_coordinate_blocks()?
            .take(1)
            .chain(futures::stream::pending())
            .boxed())
    }
}

#[tokio::test]
async fn dropped_copy_releases_guards_and_allows_source_reuse() {
    let (root, dir) = new_dir("copy_cancel_source").await;
    let source = Tensor::<FsEntry, u8>::create(
        dir,
        TensorSchema::new(NumberType::UInt(UIntType::U8), shape![2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();

    source.write_value(&[1], 255).await.unwrap();
    let paused = PausedReader(&source);
    let (partial_root, partial_dir) = new_dir("copy_cancel_partial").await;
    let mut pending = Box::pin(Tensor::copy_from(partial_dir, &paused, paused.layout(), 2));
    assert!(futures::poll!(&mut pending).is_pending());
    drop(pending);
    source.write_value(&[0], 127).await.unwrap();
    let (out_root, out_dir) = new_dir("copy_cancel_reuse").await;
    let output = Tensor::copy_from(out_dir, &source, source.layout(), 2)
        .await
        .unwrap();

    assert_eq!(output.read_value(&[0]).await.unwrap(), 127);
    assert_eq!(output.read_value(&[1]).await.unwrap(), 255);

    for path in [&root, &partial_root, &out_root] {
        common::cleanup(path).await;
    }
}

struct SparseOnce {
    starts: std::sync::atomic::AtomicUsize,
    drops: std::sync::atomic::AtomicUsize,
    malformed: bool,
}

impl TensorGeometry for SparseOnce {
    type DType = f64;

    fn dtype(&self) -> NumberType {
        NumberType::Float(FloatType::F64)
    }

    fn shape(&self) -> &[u64] {
        &[2, 8193]
    }

    fn layout(&self) -> Layout {
        Layout::Sparse { axis: Some(1) }
    }
}

impl TensorRead for SparseOnce {
    fn read_value<'a>(&'a self, _: &'a [u64]) -> fensor::BoxFuture<'a, fensor::Result<f64>> {
        Box::pin(async { panic!("copy must consume the sparse stream once") })
    }

    fn read_sparse_elements_in_order<'a>(
        &'a self,
        _: fensor::Range,
        _: ha_ndarray::Axes,
    ) -> fensor::BoxFuture<'a, fensor::Result<fensor::SparseElementStream<'a, f64>>> {
        use futures::StreamExt;
        Box::pin(async move {
            self.starts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            struct Lease<'a>(&'a std::sync::atomic::AtomicUsize);

            impl Drop for Lease<'_> {
                fn drop(&mut self) {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }

            let lease = Lease(&self.drops);
            let malformed = self.malformed;
            Ok(futures::stream::iter([
                Ok((vec![0, 0], f64::NAN)),
                Ok((vec![0, 4096], f64::INFINITY)),
                Ok((
                    if malformed {
                        vec![0, 4096]
                    } else {
                        vec![1, 8192]
                    },
                    2.,
                )),
            ])
            .map(move |entry| {
                let _ = &lease;
                entry
            })
            .boxed())
        })
    }
}

#[tokio::test]
async fn sparse_copy_consumes_once_releases_source_and_rejects_duplicates() {
    let source = SparseOnce {
        starts: 0.into(),
        drops: 0.into(),
        malformed: false,
    };

    for layout in [Layout::Dense, Layout::Sparse { axis: Some(2) }] {
        let (_root, dir) = new_dir("copy_reject_layout").await;
        let result = Tensor::<FsEntry, f64>::copy_from(dir.clone(), &source, layout, 4096).await;
        match layout {
            Layout::Dense => assert!(matches!(result, Err(Error::Unsupported(_)))),
            _ => assert!(matches!(result, Err(Error::InvalidSchema(_)))),
        }

        assert!(dir.try_read().unwrap().is_empty());
        assert_eq!(source.starts.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(source.drops.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    for (axis, malformed) in [
        (Some(1), false),
        (Some(0), false),
        (None, false),
        (Some(1), true),
    ] {
        let layout = Layout::Sparse { axis };
        let source = SparseOnce {
            starts: 0.into(),
            drops: 0.into(),
            malformed,
        };

        let (root, dir) = new_dir("copy_once").await;
        let result = Tensor::<FsEntry, f64>::copy_from(dir.clone(), &source, layout, 4096).await;
        assert_eq!(source.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(source.drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        if malformed {
            assert!(result.is_err());
            assert!(Tensor::<FsEntry, f64>::load(dir).await.is_err());
        } else {
            let tensor = result.unwrap();
            assert_eq!(tensor.layout(), layout);
            assert_eq!(
                tensor.storage_geometry().block_capacity(),
                if axis == Some(0) { 4096 } else { 1 }
            );

            assert!(tensor.read_value(&[0, 0]).await.unwrap().is_nan());
            assert_eq!(tensor.read_value(&[0, 4096]).await.unwrap(), f64::INFINITY);
            assert_eq!(tensor.read_value(&[1, 8192]).await.unwrap(), 2.);
            tensor.validate().await.unwrap();
        }

        common::cleanup(&root).await;
    }
}

// Observe the full-value reader boundary independently of indexed sparse copying.
struct ValueReader<'a> {
    tensor: &'a Tensor<FsEntry, u8>,
    blocks: Vec<std::result::Result<Vec<u8>, &'static str>>,
    waiting: Option<std::sync::Arc<tokio::sync::Notify>>,
    starts: std::sync::atomic::AtomicUsize,
    lease: std::sync::Arc<()>,
}

impl TensorGeometry for ValueReader<'_> {
    type DType = u8;

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

impl TensorRead for ValueReader<'_> {
    fn read_value<'a>(&'a self, _: &'a [u64]) -> fensor::BoxFuture<'a, fensor::Result<u8>> {
        Box::pin(async { panic!("full-value copy must consume read_blocks once") })
    }

    fn read_blocks(&self) -> fensor::Result<fensor::ValueBlockStream<'_, u8>> {
        use std::task::Poll;

        use futures::StreamExt;

        self.starts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let lease = self.lease.clone();
        let waiting = self.waiting.clone();
        Ok(futures::stream::iter(self.blocks.clone())
            .chain(futures::stream::poll_fn(move |_| {
                if let Some(waiting) = &waiting {
                    waiting.notify_one();
                    Poll::Pending
                } else {
                    Poll::Ready(None)
                }
            }))
            .map(move |block| {
                let _ = &lease;
                block.map_err(|message| Error::InvalidLayout(message.into()))
            })
            .boxed())
    }
}

#[tokio::test]
async fn full_value_copy_validates_before_zero_filtering_and_releases_sources() {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use futures::TryStreamExt;

    for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
        let (_root, dir) = new_dir("copy_values_source").await;
        let source = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(NumberType::UInt(UIntType::U8), shape![2, 4]).unwrap(),
            layout,
            3,
        )
        .await
        .unwrap();

        for (blocks, expected_error) in [
            (vec![], Some("reader returned fewer values than its shape")),
            (
                vec![Ok(vec![0; 7])],
                Some("reader returned fewer values than its shape"),
            ),
            (
                vec![Ok(vec![0; 9])],
                Some("reader returned more values than its shape"),
            ),
            (
                vec![Ok(vec![0; fensor::MAX_BATCH_ELEMENTS + 1])],
                Some("copy block:"),
            ),
            (
                vec![Ok(vec![0; 4]), Err("source failure")],
                Some("source failure"),
            ),
            (vec![Ok(vec![0; 8])], None),
            (
                vec![Ok(vec![]), Ok(vec![0, 1, 2]), Ok(vec![3, 4, 5, 6, 7])],
                None,
            ),
        ] {
            let reader = ValueReader {
                tensor: &source,
                blocks: blocks.clone(),
                waiting: None,
                starts: 0.into(),
                lease: Arc::new(()),
            };

            let (root, dir) = new_dir("copy_values_result").await;
            let result =
                Tensor::<FsEntry, u8>::copy_from_values(dir.clone(), &reader, layout, 2).await;

            assert_eq!(reader.starts.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&reader.lease), 1);

            if let Some(message) = expected_error {
                assert!(result.err().unwrap().to_string().contains(message));
                assert!(Tensor::<FsEntry, u8>::load(dir).await.is_err());
            } else {
                let copy = result.unwrap();
                let expected: Vec<_> = blocks.into_iter().flat_map(Result::unwrap).collect();
                let values: Vec<_> = copy.read_blocks().unwrap().try_concat().await.unwrap();
                assert_eq!(values, expected);
                source.write_value(&[0, 0], 255).await.unwrap();
                assert_eq!(copy.read_value(&[0, 0]).await.unwrap(), expected[0]);
                copy.sync().await.unwrap();
                drop(copy);
                drop(dir);
                let reopened = Tensor::<FsEntry, u8>::load(common::open_dir(&root).unwrap())
                    .await
                    .unwrap();

                assert_eq!(
                    reopened.read_blocks().unwrap().try_concat().await.unwrap(),
                    expected
                );
            }
        }

        let waiting = Arc::new(tokio::sync::Notify::new());
        let reader = ValueReader {
            tensor: &source,
            blocks: vec![Ok(vec![0; 4])],
            waiting: Some(waiting.clone()),
            starts: 0.into(),
            lease: Arc::new(()),
        };

        let (_partial_root, dir) = new_dir("copy_values_cancel").await;
        let mut pending = Box::pin(Tensor::<FsEntry, u8>::copy_from_values(
            dir.clone(),
            &reader,
            layout,
            2,
        ));

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::select! {
                result = &mut pending => panic!("copy completed before cancellation: {}", result.is_ok()),
                () = waiting.notified() => {}
            }
        }).await.unwrap();

        assert_eq!(reader.starts.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&reader.lease), 2);
        drop(pending);
        assert_eq!(Arc::strong_count(&reader.lease), 1);
        assert!(Tensor::<FsEntry, u8>::load(dir).await.is_err());
        source.write_value(&[1, 3], 127).await.unwrap();
        assert_eq!(source.read_value(&[1, 3]).await.unwrap(), 127);
    }
}
