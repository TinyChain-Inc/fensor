//! One storage constructor consumes base tensors and geometric views as readers.

use common::{FsEntry, new_dir};
use fensor::{
    Error, Layout, Tensor, TensorArray, TensorGeometry, TensorRead, TensorSchema, TensorTransform,
    TensorWrite,
};
use ha_ndarray::{AxisRange, axes, range, shape};
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
        let copied = Tensor::copy_from(base_dir, &source, 3).await.unwrap();
        assert_eq!(copied.schema(), source.schema());
        assert_eq!(
            copied.layout(),
            match layout {
                Layout::Dense => Layout::Dense,
                Layout::Sparse { .. } => Layout::Sparse { axis: None },
            }
        );
        let view = source.view().transpose(Some(axes![1, 0])).unwrap();
        let (view_root, view_dir) = new_dir("copy_view").await;
        let transposed = Tensor::copy_from(view_dir, &view, 2).await.unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);

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
        Tensor::copy_from(out_dir.clone(), &source, 0).await,
        Err(Error::InvalidSchema(_))
    ));
    let scalar = source.view().slice(range![AxisRange::At(0)]).unwrap();
    assert!(matches!(
        Tensor::copy_from(out_dir.clone(), &scalar, 2).await,
        Err(Error::InvalidSchema(_))
    ));
    assert!(!out_root.join("blocks").exists());
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    blocks.write().await.delete("0").await;
    assert!(Tensor::copy_from(out_dir, &source, 2).await.is_err());
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

    fn shape(&self) -> &[usize] {
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
    let (_, dir) = new_dir("coordinate_copy_source").await;
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
        let (_, dir) = new_dir("coordinate_copy_invalid").await;
        assert!(
            Tensor::<FsEntry, u8>::copy_from(
                dir,
                &CoordinateReader {
                    tensor: &tensor,
                    blocks
                },
                1
            )
            .await
            .is_err()
        );
    }

    let (_, dir) = new_dir("coordinate_copy_valid").await;
    let copy = Tensor::<FsEntry, u8>::copy_from(
        dir,
        &CoordinateReader {
            tensor: &tensor,
            blocks: vec![(vec![vec![1], vec![0]], vec![3, 2])],
        },
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
                .load(out_root.clone())
                .unwrap();
        let (other_root, other_dir) = new_dir("copy_concurrent_out").await;
        let (copied, other) = futures::try_join!(
            Tensor::copy_from(out_dir, &view, 128),
            Tensor::copy_from(other_dir, &view, 7),
        )
        .unwrap();
        copied.sync().await.unwrap();
        drop(copied);
        let dir = freqfs::Cache::<FsEntry>::new(65_536, None, 0, std::time::Duration::from_secs(1))
            .load(out_root.clone())
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

    fn shape(&self) -> &[usize] {
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
    let mut pending = Box::pin(Tensor::copy_from(partial_dir, &paused, 2));
    assert!(futures::poll!(&mut pending).is_pending());
    drop(pending);
    source.write_value(&[0], 127).await.unwrap();
    let (out_root, out_dir) = new_dir("copy_cancel_reuse").await;
    let output = Tensor::copy_from(out_dir, &source, 2).await.unwrap();
    assert_eq!(output.read_value(&[0]).await.unwrap(), 127);
    assert_eq!(output.read_value(&[1]).await.unwrap(), 255);

    for path in [&root, &partial_root, &out_root] {
        common::cleanup(path).await;
    }
}
