//! Filesystem parity between compact/affine stream requests and explicit point reads.

use fensor::{
    AxisRange, Layout, Tensor, TensorElement, TensorFileEntry, TensorRead, TensorSchema,
    TensorTransform, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{axes, range, shape};

mod common;
use common::FsEntry;

async fn check<T: TensorElement>(pattern: &[T], same: impl Fn(T, T) -> bool)
where
    FsEntry: TensorFileEntry<T>,
{
    for capacity in [1, 7, 31, 128, 4096] {
        for layout in [
            Layout::Dense,
            Layout::Sparse { axis: None },
            Layout::Sparse { axis: Some(1) },
            Layout::Sparse { axis: Some(2) },
        ] {
            let (root, dir) = common::new_dir("storage_runs").await;
            let source = Tensor::<FsEntry, T>::create(
                dir,
                TensorSchema::new(T::dtype(), shape![3, 5, 7]).unwrap(),
                layout,
                capacity,
            )
            .await
            .unwrap();

            for (i, coord) in common::iter_coords(&[3, 5, 7]).enumerate() {
                let value = pattern[i % pattern.len()];
                if matches!(layout, Layout::Dense) || value != T::ZERO {
                    source.write_value(&coord, value).await.unwrap();
                }
            }
            source.sync().await.unwrap();
            let before = file_names(&root).await;
            let views = [
                source.view(),
                source.view().transpose(Some(axes![2, 0, 1])).unwrap(),
                source.view().flip(0).unwrap().flip(2).unwrap(),
                source
                    .view()
                    .slice(range![
                        AxisRange::In(0, 3, 2),
                        AxisRange::In(1, 5, 2),
                        AxisRange::In(0, 7, 3)
                    ])
                    .unwrap(),
                source
                    .view()
                    .reshape(shape![105])
                    .unwrap()
                    .slice(range![AxisRange::In(0, 105, 11)])
                    .unwrap(),
                source
                    .view()
                    .slice(range![
                        AxisRange::At(1),
                        AxisRange::In(0, 5, 1),
                        AxisRange::At(3)
                    ])
                    .unwrap()
                    .broadcast(shape![4, 5])
                    .unwrap(),
                source
                    .view()
                    .slice(range![
                        AxisRange::Of(vec![2, 0, 2]),
                        AxisRange::In(0, 5, 1),
                        AxisRange::In(0, 7, 1)
                    ])
                    .unwrap()
                    .flip(1)
                    .unwrap(),
            ];

            for view in views {
                use fensor::TensorGeometry;
                let mut expected = Vec::new();

                for coord in common::iter_coords(view.shape()) {
                    expected.push(view.read_value(&coord).await.unwrap());
                }

                let mut stream = view.read_blocks().unwrap();
                let mut i = 0;

                while let Some(values) = stream.try_next().await.unwrap() {
                    for value in values {
                        assert!(same(value, expected[i]), "stream mismatch at {i}");
                        i += 1;
                    }
                }
                assert_eq!(i, expected.len());
                let mut stream = view.read_coordinate_blocks().unwrap();
                let mut count = 0;

                while let Some((coords, values)) = stream.try_next().await.unwrap() {
                    for (coord, value) in coords.iter().zip(values) {
                        assert!(same(value, view.read_value(coord).await.unwrap()));
                        count += 1;
                    }
                }
                assert_eq!(count, expected.len());
            }
            assert_eq!(
                file_names(&root).await,
                before,
                "evaluation must not create computed-result files"
            );
            common::cleanup(&root).await;
        }
    }
}

async fn file_names(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut names = Vec::new();

    while let Some(dir) = pending.pop() {
        let mut entries = tokio::fs::read_dir(dir).await.unwrap();

        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry.file_type().await.unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                names.push(entry.path());
            }
        }
    }
    names.sort();
    names
}

#[tokio::test]
async fn affine_streams_match_explicit_reads_for_supported_types() {
    check(&[0u8, 1, 127, 255], |a, b| a == b).await;
    check(
        &[0f32, -0., -3., f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        |a, b| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
    )
    .await;
    check(
        &[0f64, -0., -3., f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
        |a, b| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
    )
    .await;
}
