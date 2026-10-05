//! Downstream public API smoke coverage, not a transaction or storage-provider test.

use fensor::{
    AxisRange, Layout, Tensor, TensorMathScalar, TensorRead, TensorReduce, TensorSource,
    TensorTransform, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{axes, range, shape};

mod common;

async fn coordinates<R: TensorRead<DType = f32>>(reader: &R, expected: &[f32]) {
    let mut seen = vec![false; expected.len()];
    let mut blocks = reader.read_coordinate_blocks().unwrap();

    while let Some((coordinates, values)) = blocks.try_next().await.unwrap() {
        assert_eq!(coordinates.len(), values.len());

        for (coordinate, value) in coordinates.into_iter().zip(values) {
            assert_eq!(coordinate.len(), 1);
            let index = usize::try_from(coordinate[0]).unwrap();
            assert!(!seen[index], "duplicate coordinate {coordinate:?}");
            assert_eq!(value, expected[index], "coordinate {coordinate:?}");
            seen[index] = true;
        }
    }

    assert!(seen.into_iter().all(|seen| seen));
}

#[tokio::test]
async fn public_composition_storage_access_and_persistence() {
    for layout in [Layout::Dense, Layout::Sparse { axis: Some(1) }] {
        let (root, source) = common::fixture::source(
            "handoff",
            shape![2, 3],
            layout,
            2,
            1_000_000,
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .await;

        // The handoff uses logical geometry; physical payload IDs remain native.
        let geometry = source.storage_geometry();
        let (id, offset) = geometry.block_position(&[0, 0]).unwrap();
        assert_eq!(source.read_logical_block(id).await.unwrap()[offset], 1.0);

        let geometric = source.view().transpose(Some(axes![1, 0])).unwrap();
        geometric.write_value(&[0, 0], 0.0).await.unwrap();
        assert_eq!(geometric.read_value(&[0, 0]).await.unwrap(), 0.0);
        assert_eq!(geometric.read_value(&[0, 1]).await.unwrap(), 4.0);

        let expression = source
            .view()
            .slice(range![AxisRange::In(0, 2, 1), AxisRange::In(1, 3, 1)])
            .unwrap()
            .transpose(Some(axes![1, 0]))
            .unwrap()
            .add_scalar(1.0)
            .await
            .unwrap()
            .sum(axes![1], false)
            .await
            .unwrap();
        coordinates(&expression, &[9.0, 11.0]).await;

        let (copy_root, dir) = common::new_dir("handoff_copy").await;
        let copy: Tensor<common::FsEntry, f32> = Tensor::copy_from(dir.clone(), &expression, 2)
            .await
            .unwrap();
        copy.sync().await.unwrap();
        drop(copy);
        drop(dir);
        let copy = Tensor::<common::FsEntry, f32>::load(common::open_dir(&copy_root).unwrap())
            .await
            .unwrap();
        coordinates(&copy, &[9.0, 11.0]).await;
        drop(copy);
        drop(expression);
        drop(geometric);
        drop(source);
        common::cleanup(&root).await;
        common::cleanup(&copy_root).await;
    }
}
