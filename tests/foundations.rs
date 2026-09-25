use std::collections::HashMap;
use std::path::PathBuf;

use fensor::{
    AxisRange, Error, Layout, Range, Tensor, TensorGeometry, TensorRead, TensorSchema,
    TensorTransform, TensorWrite, contiguous_strides,
};
use ha_ndarray::{axes, range, shape};
use number_general::{FloatType, NumberType};

mod common;

use common::{FsEntry, cleanup, iter_coords, new_dir};

async fn create_tensor(name: &str, layout: Layout) -> (PathBuf, Tensor<FsEntry, f32>) {
    let (root, dir) = new_dir(name).await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3, 4]).expect("schema");
    // Keep blocks small so reads and transforms cross storage boundaries.
    let tensor = Tensor::create(dir, schema, layout, 4)
        .await
        .expect("create tensor");
    (root, tensor)
}

async fn seed_values(tensor: &Tensor<FsEntry, f32>) {
    for coord in iter_coords(tensor.shape()) {
        let value = (coord[0] * 100 + coord[1] * 10 + coord[2]) as f32;
        tensor
            .write_value(&coord, value)
            .await
            .expect("write value");
    }
}

fn transpose_range(range: &Range, permutation: &[usize]) -> Range {
    let mut remapped = Vec::with_capacity(permutation.len());

    for axis in permutation {
        remapped.push(range[*axis].clone());
    }

    remapped.into()
}

#[tokio::test]
async fn accessor_coordinate_offset_and_read_write_dense() {
    let (root, tensor) = create_tensor(
        "accessor_coordinate_offset_and_read_write_dense",
        Layout::Dense,
    )
    .await;

    assert_eq!(tensor.view().flat_offset(&[1, 2, 3]).expect("offset"), 23);

    tensor.write_value(&[1, 2, 3], 7.5).await.expect("write");
    tensor.write_value(&[0, 0, 0], 1.25).await.expect("write");

    let v_a = tensor.read_value(&[1, 2, 3]).await.expect("read");
    let v_b = tensor.read_value(&[0, 0, 0]).await.expect("read");
    let v_c = tensor.read_value(&[1, 1, 1]).await.expect("read");

    assert_eq!(v_a, 7.5);
    assert_eq!(v_b, 1.25);
    assert_eq!(v_c, 0.0);

    cleanup(&root).await;
}

#[tokio::test]
async fn standalone_transpose_arbitrary_permutation() {
    let (root, tensor) =
        create_tensor("standalone_transpose_arbitrary_permutation", Layout::Dense).await;
    seed_values(&tensor).await;

    let perm = axes![2, 0, 1];
    let transposed = tensor
        .view()
        .transpose(Some(perm.clone()))
        .expect("transpose");

    assert_eq!(transposed.shape(), &[4, 2, 3]);

    let mut inverse = vec![0usize; perm.len()];

    for (i, axis) in perm.iter().enumerate() {
        inverse[*axis] = i;
    }

    for t_coord in iter_coords(transposed.shape()) {
        let mut src = vec![0u64; t_coord.len()];

        for old_axis in 0..t_coord.len() {
            src[old_axis] = t_coord[inverse[old_axis]];
        }

        let expected = tensor.read_value(&src).await.expect("read original");
        let actual = transposed
            .read_value(&t_coord)
            .await
            .expect("read transposed");
        assert_eq!(actual, expected, "coord {:?}", t_coord);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn standalone_slice_range_selection() {
    let (root, tensor) = create_tensor("standalone_slice_range_selection", Layout::Dense).await;
    seed_values(&tensor).await;

    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(1, 3, 1),
        AxisRange::In(0, 4, 2)
    ];

    let sliced = tensor.view().slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2, 2]);

    for s_coord in iter_coords(sliced.shape()) {
        let src = vec![s_coord[0], s_coord[1] + 1, s_coord[2] * 2];
        let expected = tensor.read_value(&src).await.expect("read original");
        let actual = sliced.read_value(&s_coord).await.expect("read slice");
        assert_eq!(actual, expected, "coord {:?}", s_coord);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn composition_transpose_slice_and_slice_transpose_consistency() {
    let (root, tensor) = create_tensor(
        "composition_transpose_slice_and_slice_transpose_consistency",
        Layout::Dense,
    )
    .await;
    seed_values(&tensor).await;

    let perm = axes![2, 0, 1];
    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(1, 3, 1),
        AxisRange::In(0, 4, 2)
    ];

    let left = tensor
        .view()
        .slice(r.clone())
        .expect("slice")
        .transpose(Some(perm.clone()))
        .expect("transpose");

    let remapped = transpose_range(&r, &perm);
    let right = tensor
        .view()
        .transpose(Some(perm))
        .expect("transpose")
        .slice(remapped)
        .expect("slice");

    assert_eq!(left.shape(), right.shape());

    for coord in iter_coords(left.shape()) {
        let left_v = left.read_value(&coord).await.expect("left read");
        let right_v = right.read_value(&coord).await.expect("right read");
        assert_eq!(left_v, right_v, "coord {:?}", coord);
    }

    cleanup(&root).await;
}

#[tokio::test]
async fn dense_sparse_parity_for_supported_operations() {
    let (dense_root, dense) = create_tensor("parity_dense", Layout::Dense).await;
    let (sparse_root, sparse) =
        create_tensor("parity_sparse", Layout::Sparse { axis: Some(1) }).await;

    let writes: HashMap<Vec<u64>, f32> = [
        (vec![0, 0, 0], 1.0),
        (vec![1, 2, 3], 9.0),
        (vec![1, 0, 1], 4.0),
        (vec![0, 2, 2], 7.0),
    ]
    .into_iter()
    .collect();

    for (coord, value) in &writes {
        dense.write_value(coord, *value).await.expect("dense write");
        sparse
            .write_value(coord, *value)
            .await
            .expect("sparse write");
    }

    for coord in iter_coords(dense.shape()) {
        let expected = writes.get(&coord).copied().unwrap_or(0.0);
        let d = dense.read_value(&coord).await.expect("dense read");
        let s = sparse.read_value(&coord).await.expect("sparse read");
        assert_eq!(d, expected, "dense base coord {:?}", coord);
        assert_eq!(s, expected, "sparse base coord {:?}", coord);
    }

    let permuted_dense = dense
        .view()
        .transpose(Some(axes![1, 2, 0]))
        .expect("transpose");
    let permuted_sparse = sparse
        .view()
        .transpose(Some(axes![1, 2, 0]))
        .expect("transpose");

    for coord in iter_coords(permuted_dense.shape()) {
        let d = permuted_dense.read_value(&coord).await.expect("dense read");
        let s = permuted_sparse
            .read_value(&coord)
            .await
            .expect("sparse read");
        assert_eq!(d, s, "transposed coord {:?}", coord);
    }

    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(0, 3, 2),
        AxisRange::In(1, 4, 1)
    ];

    let sliced_dense = dense.view().slice(r.clone()).expect("slice");
    let sliced_sparse = sparse.view().slice(r).expect("slice");

    for coord in iter_coords(sliced_dense.shape()) {
        let d = sliced_dense.read_value(&coord).await.expect("dense read");
        let s = sliced_sparse.read_value(&coord).await.expect("sparse read");
        assert_eq!(d, s, "sliced coord {:?}", coord);
    }

    cleanup(&dense_root).await;
    cleanup(&sparse_root).await;
}

#[tokio::test]
async fn public_create_rejects_invalid_sparse_axis_hint() {
    let (root, dir) = common::new_dir("invalid_sparse_axis").await;
    let schema =
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 3]).expect("schema");
    let result = fensor::Tensor::<common::FsEntry, f32>::create(
        dir,
        schema,
        Layout::Sparse { axis: Some(99) },
        1000,
    )
    .await;
    assert!(
        matches!(result, Err(Error::InvalidSchema(_))),
        "out-of-bounds sparse axis must be rejected"
    );
    common::cleanup(&root).await;
}

#[test]
fn public_schema_contiguous_strides_match_expected() {
    assert_eq!(
        contiguous_strides(&[2, 3, 4]).expect("strides").as_slice(),
        &[12, 4, 1]
    );
}

#[test]
fn temporary_directory_names_are_unique_across_threads() {
    let workers: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                (0..32)
                    .map(|_| common::unique_tmp_dir("parallel_names"))
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    let mut names = std::collections::HashSet::new();

    for worker in workers {
        for name in worker.join().unwrap() {
            assert!(names.insert(name), "temporary directory name collision");
        }
    }

    assert_eq!(names.len(), 256);
}
