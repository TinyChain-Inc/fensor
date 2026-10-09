use std::collections::BTreeMap;

use ha_ndarray::shape;
use number_general::FloatType;

use super::*;
use crate::Shape;
use crate::TensorSource;
use crate::test_support::{Directory, FsEntry as TestFE, cleanup, new_dir, open_dir};

async fn create_sparse(
    name: &str,
    shape: Shape,
    max_capacity: usize,
    axis: Option<usize>,
) -> (Directory, Tensor<TestFE, f32>) {
    let (root, dir) = new_dir(name).await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape).expect("schema");
    let tensor = Tensor::<TestFE, f32>::create(dir, schema, Layout::Sparse { axis }, max_capacity)
        .await
        .expect("create sparse");
    (root, tensor)
}

async fn block_id_for_coord(tensor: &Tensor<TestFE, f32>, coord: &[u64]) -> Option<u64> {
    let BlockPosition { block_id, .. } = tensor.block_position_from_base_coord(coord);
    tensor
        .storage
        .sparse()
        .unwrap()
        .occupied_page(block_id..block_id + 1)
        .await
        .expect("lookup")
        .contains(&block_id)
        .then_some(block_id)
}

// Caller-selected sparse block geometry is preserved through loading.
async fn persisted_sparse(
    name: &str,
    shape: Shape,
    block_shape: Shape,
    axis: Option<usize>,
) -> (Directory, Tensor<TestFE, f32>) {
    let (root, dir) = new_dir(name).await;
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape.clone()).unwrap();
    let storage =
        StorageSchema::from_block_shape(&shape, Layout::Sparse { axis }, block_shape).unwrap();
    let tensor = Tensor::<TestFE, f32>::create_with_geometry(
        dir.clone(),
        crate::StorageGeometry { schema, storage },
    )
    .await
    .unwrap();
    tensor.sync().await.unwrap();
    drop(tensor);
    drop(dir);
    let tensor = Tensor::load(open_dir(&root).unwrap()).await.unwrap();
    (root, tensor)
}

#[tokio::test]
async fn clearing_one_value_preserves_the_rest_of_its_sparse_key() {
    let (root, tensor) = create_sparse("shared_key_zero", shape![2, 2], 4, Some(0)).await;
    tensor.write_value(&[0, 0], 3.).await.unwrap();
    tensor.write_value(&[0, 1], 7.).await.unwrap();
    let id = block_id_for_coord(&tensor, &[0, 0]).await.unwrap();
    assert_eq!(block_id_for_coord(&tensor, &[0, 1]).await, Some(id));

    tensor.write_value(&[0, 0], 0.).await.unwrap();
    assert_eq!(tensor.read_value(&[0, 1]).await.unwrap(), 7.);
    assert_eq!(block_id_for_coord(&tensor, &[0, 1]).await, Some(id));
    tensor.write_value(&[0, 1], 0.).await.unwrap();
    assert!(block_id_for_coord(&tensor, &[0, 1]).await.is_none());
    assert_eq!(tensor.read_logical_block(id).await.unwrap(), vec![0.; 2]);
    assert!(
        tensor
            .occupied_blocks(0..tensor.num_blocks())
            .try_next()
            .await
            .unwrap()
            .is_none()
    );
    cleanup(&root).await;
}

#[tokio::test]
async fn zero_writes_respect_persisted_regions_edges_and_shared_pages() {
    for (axis, block_shape) in [
        (None, shape![1, 1, 1]),
        (Some(0), shape![1, 2, 2]),
        (Some(1), shape![1, 1, 2]),
        (Some(2), shape![1, 1, 1]),
    ] {
        let (root, tensor) = persisted_sparse(
            "persisted_regions",
            shape![3, 3, 3],
            block_shape.clone(),
            axis,
        )
        .await;
        let first = [0, 0, 0];
        let neighbor = [0, 0, 1];
        tensor.write_value(&first, 2.).await.unwrap();
        tensor.write_value(&neighbor, f32::NAN).await.unwrap();
        let neighbor_id = block_id_for_coord(&tensor, &neighbor).await.unwrap();
        let expected = tensor
            .storage_geometry()
            .block_position(&neighbor)
            .unwrap()
            .0;
        assert_eq!(neighbor_id, expected);
        tensor.write_value(&first, -0.).await.unwrap();
        assert!(tensor.read_value(&neighbor).await.unwrap().is_nan());
        assert_eq!(
            block_id_for_coord(&tensor, &neighbor).await,
            Some(neighbor_id)
        );

        // A different sparse prefix stays independent, including when its native
        // row shares a filesystem page with the replaced block.
        let independent = [1, 0, 0];
        tensor.write_value(&independent, 5.).await.unwrap();
        tensor.write_value(&neighbor, 0.).await.unwrap();
        assert_eq!(tensor.read_value(&independent).await.unwrap(), 5.);
        assert!(block_id_for_coord(&tensor, &neighbor).await.is_none());
        tensor.write_value(&independent, 0.).await.unwrap();
        assert!(block_id_for_coord(&tensor, &independent).await.is_none());

        let edge = [2, 2, 2];
        tensor.write_value(&edge, 9.).await.unwrap();
        let edge_id = block_id_for_coord(&tensor, &edge).await.unwrap();
        let mut block = tensor.read_logical_block(edge_id).await.unwrap();
        if block.len() > 1 {
            *block.last_mut().unwrap() = 11.;
        } else {
            block.push(11.);
        }
        assert!(tensor.replace_logical_block(edge_id, block).await.is_err());
        assert_eq!(tensor.read_value(&edge).await.unwrap(), 9.);
        tensor.write_value(&edge, 0.).await.unwrap();
        tensor.validate().await.unwrap();
        tensor.sync().await.unwrap();
        drop(tensor);
        let loaded = Tensor::<TestFE, f32>::load(open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(loaded.block_shape(), block_shape.as_slice());
        assert_eq!(loaded.read_value(&edge).await.unwrap(), 0.);
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn cancelled_sparse_replacement_before_mutation_releases_owner() {
    for len in [1, 512] {
        let (root, tensor) = create_sparse(
            "interrupted_publication",
            shape![2, len],
            len as usize,
            Some(0),
        )
        .await;
        tensor
            .replace_logical_block(0, vec![1.; len as usize])
            .await
            .unwrap();
        let clone = tensor.clone();
        let owner = tensor.storage.sparse().unwrap();
        {
            use futures::FutureExt;
            let held = owner.gate.write().await;
            assert!(matches!(
                tensor
                    .replace_logical_block(owner.geometry.block_count(), vec![])
                    .now_or_never(),
                Some(Err(Error::InvalidCoord(_)))
            ));
            assert!(matches!(
                tensor.replace_logical_block(0, vec![]).now_or_never(),
                Some(Err(Error::InvalidLayout(_)))
            ));
            drop(held);
            assert_eq!(
                tensor.read_logical_block(0).await.unwrap(),
                vec![1.; len as usize]
            );
        }
        // Cancellation while waiting for native table access leaves no held guards.
        let guard = owner.values.write().await;
        {
            let mut replacement =
                std::pin::pin!(tensor.replace_logical_block(0, vec![2.; len as usize]));
            assert!(futures::poll!(replacement.as_mut()).is_pending());
        }
        drop(guard);
        assert!(owner.gate.try_write().is_ok());
        // This fixture stopped before any table mutation. Inspect native data
        // directly, then follow the caller contract by discarding both handles.
        assert_eq!(
            owner
                .values
                .read()
                .await
                .get_row(&[crate::SparseCell::Key(0)])
                .await
                .unwrap()
                .unwrap()[1],
            crate::SparseCell::Payload(vec![1.; len as usize])
        );
        drop(clone);
        drop(tensor);
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn sparse_capacity_spans_grids_without_axis_padding() {
    for capacity in [1, 7, 31, 128, MAX_BLOCK_CAPACITY] {
        for axis in [None, Some(0), Some(1), Some(2)] {
            let (root, tensor) = create_sparse(
                "sparse_capacity",
                shape![3, 5, MAX_BLOCK_CAPACITY as u64 + 1],
                capacity,
                axis,
            )
            .await;
            assert_eq!(tensor.block_shape()[axis.unwrap_or(0)], 1);
            assert!(tensor.block_len() <= capacity);

            for (coord, value) in [([0, 0, 0], 1.), ([2, 4, MAX_BLOCK_CAPACITY as u64], 2.)] {
                tensor.write_value(&coord, value).await.unwrap();
                assert_eq!(tensor.read_value(&coord).await.unwrap(), value);
            }
            tensor.sync().await.unwrap();
            drop(tensor);
            let loaded = Tensor::<TestFE, f32>::load(open_dir(&root).unwrap())
                .await
                .unwrap();
            assert_eq!(loaded.block_shape()[axis.unwrap_or(0)], 1);
            assert_eq!(
                loaded
                    .read_value(&[2, 4, MAX_BLOCK_CAPACITY as u64])
                    .await
                    .unwrap(),
                2.
            );
            cleanup(&root).await;
        }
    }
}

#[tokio::test]
async fn occupied_block_pages_resume_after_delivered_ids() {
    use crate::TensorReduceAll;

    let entries = SPARSE_INDEX_PAGE_ENTRIES + 1;

    for axis in [None, Some(1)] {
        let (root, tensor) =
            create_sparse("slice_pages", shape![1_u64, entries as u64], 1, axis).await;

        for col in 0..entries as u64 {
            tensor.write_value(&[0, col], 1.).await.unwrap();
        }

        let owner = tensor.storage.sparse().unwrap();
        let first = owner.occupied_page(0..entries as u64).await.unwrap();
        assert_eq!(first.len(), SPARSE_INDEX_PAGE_ENTRIES);
        let second = owner
            .occupied_page(first.last().copied().unwrap() + 1..entries as u64)
            .await
            .unwrap();
        assert_eq!(second.len(), 1);
        assert!(first.last().unwrap() < second.first().unwrap());
        assert!(
            owner
                .occupied_page(second[0] + 1..entries as u64)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(tensor.sum_all().await.unwrap(), entries as f32);
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn sparse_point_mutation_lifecycle() {
    let (root, tensor) = create_sparse("point_lifecycle", shape![2, 3, 4], 4, Some(1)).await;
    let coord = [0, 1, 2];
    let logical_block = tensor.storage_geometry().block_position(&coord).unwrap().0;
    assert!(block_id_for_coord(&tensor, &coord).await.is_none());

    // Absent zero, insertion, replacement, deletion, reinsertion, and deletion.
    for value in [0., 1., 2., 0., 5., 0.] {
        tensor.write_value(&coord, value).await.unwrap();
        let occupied = block_id_for_coord(&tensor, &coord).await;
        assert_eq!(occupied, (value != 0.).then_some(logical_block));
        assert_eq!(tensor.read_value(&coord).await.unwrap(), value);
        assert_eq!(
            tensor.read_logical_block(logical_block).await.unwrap()[2],
            value
        );
    }
    drop(tensor);
    cleanup(&root).await;
}

#[tokio::test]
async fn oversized_evaluation_is_rejected_before_reading_coordinates() {
    let (root, tensor) = create_sparse("batch_bound", shape![1], 1, None).await;
    let coords = vec![vec![99]; crate::expression::MAX_BATCH_ELEMENTS + 1];
    let error = BatchRequest::explicit(coords).err().unwrap();
    let _ = tensor;
    assert!(matches!(error, Error::InvalidLayout(_)));
    assert!(error.to_string().contains("coordinate batch"));
    cleanup(&root).await;
}

#[tokio::test]
async fn storage_batches_group_blocks_and_preserve_order() {
    for layout in [Layout::Dense, Layout::Sparse { axis: Some(0) }] {
        let (root, dir) = new_dir("read_groups").await;
        let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 4]).unwrap();
        let tensor = Tensor::<TestFE, f32>::create(dir, schema, layout, 4)
            .await
            .unwrap();
        tensor.write_value(&[0, 0], 2.).await.unwrap();
        tensor.write_value(&[0, 3], 3.).await.unwrap();
        tensor.write_value(&[1, 1], 4.).await.unwrap();

        let coords = vec![vec![1, 1], vec![0, 3], vec![0, 0], vec![0, 3]];
        let groups: BTreeMap<_, _> = tensor
            .read_groups(&BatchRequest::explicit(coords.clone()).unwrap(), None)
            .unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups.values().flatten().map(|run| run.len).sum::<usize>(),
            coords.len()
        );
        assert!(groups.values().any(|positions| {
            positions
                .iter()
                .flat_map(|run| {
                    (0..run.len).map(move |i| {
                        (
                            run.output + i,
                            (run.offset as i128 + i as i128 * run.stride as i128) as usize,
                        )
                    })
                })
                .collect::<Vec<_>>()
                == vec![(1, 3), (2, 0), (3, 3)]
        }));
        assert_eq!(
            tensor
                .read_batch(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                .await
                .unwrap(),
            vec![4., 3., 2., 3.]
        );
        assert!(
            tensor
                .read_batch(&BatchRequest::point(&[2, 0]), None)
                .await
                .is_err()
        );
        assert!(
            BatchRequest::explicit(vec![vec![0, 0]; crate::expression::MAX_BATCH_ELEMENTS + 1])
                .is_err()
        );

        // Validate the entire borrowed block, even for one selected value.
        let id = *groups.keys().next().unwrap();
        if matches!(layout, Layout::Dense) {
            let file = tensor
                .storage
                .blocks()
                .read()
                .await
                .get_file(&id.to_string())
                .unwrap()
                .clone();
            file.write::<Vec<f32>>(0).await.unwrap().pop();
        } else {
            crate::tensor::corruption::corrupt_sparse_payload(&tensor, id).await;
        }
        assert!(matches!(
            tensor
                .read_batch(&BatchRequest::explicit(coords.clone()).unwrap(), None)
                .await,
            Err(Error::InvalidLayout(_))
        ));
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn invalid_affine_requests_fail_before_storage_io() {
    let (root, tensor) = create_sparse("invalid_affine", shape![2, 4], 4, None).await;
    let mut map = CoordinateMap::identity(shape![2, 4], tensor.strides());
    map.base_offset = -1;
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            assert!(matches!(
                tensor
                    .read_batch(&BatchRequest::linear(0, 8).unwrap(), Some(&map))
                    .await,
                Err(Error::InvalidCoord(_))
            ));
            crate::read_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!((m.logical_payload_reads, m.borrows), (0, 0));
            });
        })
        .await;
    cleanup(&root).await;
}

#[tokio::test]
async fn storage_batch_groups_logical_blocks_across_sparse_regions() {
    let (root, tensor) = persisted_sparse("batch_keys", shape![3, 2], shape![1, 2], Some(0)).await;
    tensor.write_value(&[0, 0], 2.).await.unwrap();
    tensor.write_value(&[2, 1], 3.).await.unwrap();
    let request =
        BatchRequest::explicit(vec![vec![0, 0], vec![1, 0], vec![2, 1], vec![0, 0]]).unwrap();
    assert_eq!(
        tensor.read_batch(&request, None).await.unwrap(),
        vec![2., 0., 3., 2.]
    );
    tensor.write_value(&[0, 0], 0.).await.unwrap();
    assert_eq!(
        tensor.read_batch(&request, None).await.unwrap(),
        vec![0., 0., 3., 0.]
    );
    tensor.validate().await.unwrap();
    cleanup(&root).await;
}

#[tokio::test]
async fn dense_sparse_matrix_products_include_zero_values() {
    use crate::{TensorMatMul, TensorTransform};
    let (root, dir) = new_dir("dense_matrix_support").await;
    let dense = Tensor::<TestFE, f32>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 2]).unwrap(),
        Layout::Dense,
        2,
    )
    .await
    .unwrap();
    let (sparse_root, sparse) = create_sparse("sparse_matrix_support", shape![2, 2], 2, None).await;
    let left = dense.view().matmul(&sparse.view()).await.unwrap();
    let right = sparse
        .view()
        .matmul(&dense.view().transpose(None).unwrap())
        .await
        .unwrap();
    for values in [
        crate::expression::evaluate_batch(&left, &BatchRequest::point(&[0, 0]))
            .await
            .unwrap()
            .values,
        crate::expression::evaluate_batch(&right, &BatchRequest::point(&[0, 0]))
            .await
            .unwrap()
            .values,
    ] {
        assert_eq!(values, vec![0.]);
    }
    cleanup(&root).await;
    cleanup(&sparse_root).await;
}

#[derive(Clone)]
struct CompactSource<'a>(&'a Tensor<TestFE, f32>);

impl TensorGeometry for CompactSource<'_> {
    type DType = f32;

    fn dtype(&self) -> NumberType {
        self.0.dtype()
    }

    fn shape(&self) -> &[u64] {
        self.0.shape()
    }

    fn layout(&self) -> Layout {
        self.0.layout()
    }
}

impl crate::expression::traversal::Plan for CompactSource<'_> {}

impl crate::expression::Expression for CompactSource<'_> {
    fn build<'a>(
        &'a self,
        context: crate::expression::Context<'a>,
        request: std::sync::Arc<BatchRequest>,
    ) -> BoxFuture<'a, Result<crate::expression::Batch<f32>>> {
        Box::pin(async move {
            assert!(matches!(
                request.kind(),
                crate::request::RequestKind::Rectangles(_)
            ));
            self.0.build(context, request).await
        })
    }
}

#[tokio::test]
async fn compact_requests_survive_elementwise_composition() {
    use crate::{TensorCompareScalar, TensorMath, TensorUnary, TensorWhere};
    let (root, tensor) = create_sparse("compact_composition", shape![2, 2], 2, None).await;
    tensor.write_value(&[1, 1], 2.).await.unwrap();
    let source = CompactSource(&tensor);
    let values = source.round().await.unwrap().add(&source).await.unwrap();
    let condition = values.gt_scalar(0.).await.unwrap();
    let selected = condition.cond(&values, &source).await.unwrap();
    let request = BatchRequest::rectangles(vec![
        crate::request::Cartesian::new(vec![
            crate::request::Axis::range(0, 2),
            crate::request::Axis::range(0, 2),
        ])
        .unwrap(),
    ])
    .unwrap();
    assert_eq!(
        crate::expression::evaluate_batch(&selected, &request)
            .await
            .unwrap()
            .values,
        vec![0., 0., 0., 4.]
    );

    // An invalid mapped position is rejected before any data/index read.
    let mut mapping = crate::mapping::CoordinateMap::identity(shape![2, 2], tensor.strides());
    mapping.base_offset = 4;
    assert!(matches!(
        tensor.read_batch(&request, Some(&mapping)).await,
        Err(Error::InvalidCoord(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn copy_batches_group_logical_blocks() {
    let layout = Layout::Dense;
    let (root, dir) = new_dir("copy_groups").await;
    let tensor = Tensor::<TestFE, f32>::unpublished(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 4]).unwrap(),
        layout,
        8,
    )
    .await
    .unwrap();
    let output = construction::Dense(tensor);
    let coords = vec![vec![1, 3], vec![0, 2], vec![1, 1], vec![0, 0]];
    let expected = 1;
    copy_metrics::CURRENT
        .scope(Default::default(), async {
            for _ in 0..2 {
                output
                    .stage(
                        BatchRequest::explicit(coords.clone()).unwrap(),
                        vec![3., 2., 1., 4.],
                    )
                    .await
                    .unwrap();
            }
            copy_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!(m.groups, 2 * expected);
                assert_eq!(m.block_updates, 2 * expected);
            });
        })
        .await;
    let tensor = output.finish().await.unwrap();

    for (coord, value) in coords.iter().zip([3., 2., 1., 4.]) {
        assert_eq!(tensor.read_value(coord).await.unwrap(), value);
    }
    tensor.sync().await.unwrap();
    drop(tensor);
    drop(dir);
    let loaded = Tensor::<TestFE, f32>::load(open_dir(&root).unwrap())
        .await
        .unwrap();
    assert_eq!(loaded.read_value(&[1, 3]).await.unwrap(), 3.);
    cleanup(&root).await;
}

#[tokio::test]
async fn copy_batches_coalesce_existing_logical_blocks() {
    // Row-major input repeatedly interleaves many destination blocks.
    let (root, source) = create_sparse("copy_interleaved", shape![65, 33], 4096, Some(1)).await;

    for id in 0..source.num_blocks() {
        source
            .replace_logical_block(id, vec![1.; source.block_len()])
            .await
            .unwrap();
    }

    let (out_root, dir) = new_dir("copy_interleaved_output").await;
    let output = copy_metrics::CURRENT
        .scope(Default::default(), async {
            let output = Tensor::<TestFE, f32>::copy_from(dir, &source, source.layout(), 4096)
                .await
                .unwrap();
            copy_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!(m.block_updates, 0);
                assert_eq!(m.constructed_blocks, output.num_blocks() as usize);
                assert_eq!(m.staged_elements, 65 * 33);
                assert!(m.max_staging_batch <= crate::expression::MAX_BATCH_ELEMENTS);
            });
            output
        })
        .await;
    for id in 0..output.num_blocks() {
        assert_eq!(
            output.read_logical_block(id).await.unwrap(),
            vec![1.; output.block_len()]
        );
    }
    output.validate().await.unwrap();
    cleanup(&root).await;
    cleanup(&out_root).await;
}

#[tokio::test]
async fn update_planning_rejects_invalid_input_and_corrupt_blocks() {
    let (root, dir) = new_dir("copy_invalid").await;
    let tensor = Tensor::<TestFE, f32>::create(
        dir.clone(),
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![4]).unwrap(),
        Layout::Dense,
        4,
    )
    .await
    .unwrap();

    for (coords, values) in [
        (vec![vec![0], vec![4]], vec![1., 2.]),
        (vec![vec![0]], vec![]),
        (
            vec![vec![0]; crate::expression::MAX_BATCH_ELEMENTS + 1],
            vec![1.; crate::expression::MAX_BATCH_ELEMENTS + 1],
        ),
    ] {
        assert!(
            BatchRequest::explicit(coords)
                .and_then(|request| crate::storage::plan_updates(
                    &tensor.storage_geometry(),
                    None,
                    &request,
                    values,
                ))
                .is_err()
        );
        assert_eq!(tensor.read_value(&[0]).await.unwrap(), 0.);
    }

    let mut block = vec![0.; 4];
    assert!(apply_block_updates(&mut block, tensor.block_len(), &[(0, 1.), (4, 2.)]).is_err());
    assert_eq!(block, vec![0.; 4]);
    let blocks = dir.read().await.get_dir(BLOCKS).unwrap().clone();
    let file = blocks.read().await.get_file("0").unwrap().clone();
    file.write::<Vec<f32>>(0).await.unwrap().pop();
    assert!(tensor.write_value(&[0], 1.).await.is_err());
    assert_eq!(*file.read::<Vec<f32>>().await.unwrap(), vec![0.; 3]);
    cleanup(&root).await;
}

#[tokio::test]
async fn sparse_copy_zeros_do_not_create_storage() {
    let (root, dir) = new_dir("copy_zeros").await;
    let tensor = Tensor::<TestFE, f32>::unpublished(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 4]).unwrap(),
        Layout::Sparse { axis: None },
        8,
    )
    .await
    .unwrap();
    let tensor = copy_metrics::CURRENT
        .scope(Default::default(), async {
            let mut builder = sparse_storage::Construction::new(tensor);
            builder
                .stage(&[vec![0, 0], vec![1, 1]], vec![0., -0.])
                .await
                .unwrap();
            copy_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!((m.groups, m.block_updates, m.constructed_blocks), (0, 0, 0));
            });
            builder.finish().await.unwrap()
        })
        .await;
    assert!(
        tensor
            .occupied_blocks(0..tensor.num_blocks())
            .try_next()
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(tensor.storage.blocks().read().await.files().count(), 1); // metadata only
    cleanup(&root).await;
}

#[tokio::test]
async fn ordered_sparse_consumers_visit_candidate_blocks_with_intermediate_zeros() {
    use crate::{TensorExpression, TensorMath, TensorStatistics};
    let (root, tensor) = create_sparse("ordered_support", shape![1_000_000, 4], 4, None).await;
    let (baseline_root, baseline) =
        create_sparse("ordered_baseline", shape![1_000_000, 4], 4, None).await;
    tensor.write_value(&[12, 3], 2.0).await.unwrap();
    tensor.write_value(&[999_999, 1], 4.0).await.unwrap();
    baseline.write_value(&[12, 3], 2.0).await.unwrap();
    baseline.write_value(&[999_999, 1], 2.0).await.unwrap();
    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let shifted = tensor.view().sub(&baseline.view()).await.unwrap();
            let expression = TensorExpression::new(shifted.clone()).unwrap();
            let entries: Vec<_> = expression
                .into_sparse_elements()
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(entries, [(vec![999_999, 1], 2.0)]);
            let count = 4_000_000.;
            assert_eq!(shifted.mean_all().await.unwrap(), 2. / count);
            let variance: f64 = 4. / count - (2. / count).powi(2);
            assert!((shifted.std_all().await.unwrap() - variance.sqrt()).abs() < 1e-12);
            let doubled = shifted.add(&shifted).await.unwrap();
            assert_eq!(doubled.mean_all().await.unwrap(), 4. / count);
            let entries: Vec<_> = TensorExpression::new(doubled)
                .unwrap()
                .into_sparse_elements()
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            assert_eq!(entries, [(vec![999_999, 1], 4.0)]);
            crate::read_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert!((1..=40).contains(&m.index_entries), "{m:?}");
                assert!(m.requested <= 128, "{m:?}");
            });
        })
        .await;
    cleanup(&root).await;
    cleanup(&baseline_root).await;
}

#[tokio::test]
async fn reordered_sparse_stream_preserves_row_major_values() {
    use crate::{TensorExpression, TensorTransform};
    let (root, tensor) = create_sparse("ordered_transpose", shape![2, 3], 2, None).await;
    tensor.write_value(&[0, 2], f32::NAN).await.unwrap();
    tensor.write_value(&[1, 0], f32::INFINITY).await.unwrap();
    let expression = TensorExpression::new(tensor.view().transpose(None).unwrap()).unwrap();
    let entries: Vec<_> = expression
        .into_sparse_elements()
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0], (vec![0, 1], f32::INFINITY));
    assert_eq!(entries[1].0, vec![2, 0]);
    assert!(entries[1].1.is_nan());
    cleanup(&root).await;
}

#[tokio::test]
async fn interrupted_construction_never_reopens_and_releases_guards() {
    for metadata in [false, true] {
        let (root, dir) = new_dir("construction_cancel").await;
        let tensor = Tensor::<TestFE, f32>::unpublished(
            dir.clone(),
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![2, 512]).unwrap(),
            Layout::Sparse { axis: Some(0) },
            512,
        )
        .await
        .unwrap();
        let probe = tensor.clone();
        let owner = probe.storage.sparse().unwrap();
        let mut builder = sparse_storage::Construction::new(tensor);
        builder.stage(&[vec![0, 0]], vec![1.]).await.unwrap();

        if metadata {
            let held = owner.blocks.write().await;
            let mut future = builder.finish();
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(10), &mut future)
                    .await
                    .is_err()
            );
            // Completion has persisted its row before waiting for metadata. This
            // exercises cancellation after mutation, independently of table locks.
            assert!(
                owner
                    .values
                    .read()
                    .await
                    .get_row(&[crate::SparseCell::Key(0)])
                    .await
                    .unwrap()
                    .is_some()
            );
            drop(future);
            drop(held);
        } else {
            let held = owner.values.write().await;
            let coords = [vec![1, 0]];
            let mut future = builder.stage(&coords, vec![2.]);
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(10), &mut future)
                    .await
                    .is_err()
            );
            drop(future);
            drop(held);
            drop(builder);
        }

        assert!(owner.gate.try_write().is_ok());
        // The caller discards every handle after cancellation. Metadata was
        // withheld, so incomplete construction cannot be reopened.
        drop(probe);
        assert!(Tensor::<TestFE, f32>::load(dir).await.is_err());
        cleanup(&root).await;
    }
}

#[tokio::test]
async fn construction_preserves_interleaved_blocks_across_batches() {
    let (root, dir) = new_dir("interleaved_construction").await;
    let shape = shape![1, 129, 65];
    let schema = TensorSchema::new(NumberType::Float(FloatType::F32), shape.clone()).unwrap();
    let geometry =
        crate::StorageGeometry::new(schema, Layout::Sparse { axis: Some(0) }, shape![1, 2, 16])
            .unwrap();
    let tensor = Tensor::<TestFE, f32>::unpublished_with_geometry(dir, geometry.clone())
        .await
        .unwrap();
    let mut builder = sparse_storage::Construction::new(tensor);
    let mut input = crate::row_major_coords(&shape).unwrap();
    let tensor = copy_metrics::CURRENT
        .scope(Default::default(), async {
            loop {
                let coords: Vec<_> = input
                    .by_ref()
                    .take(crate::expression::MAX_BATCH_ELEMENTS)
                    .collect();
                if coords.is_empty() {
                    break;
                }
                let len = coords.len();
                builder.stage(&coords, vec![1.; len]).await.unwrap();
            }
            let tensor = builder.finish().await.unwrap();
            copy_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!(m.constructed_blocks as u64, geometry.block_count());
                assert_eq!(m.staged_elements, 129 * 65);
                assert_eq!(m.max_staging_batch, crate::expression::MAX_BATCH_ELEMENTS);
                assert_eq!(m.block_updates, 0);
                assert_eq!(m.replaced_blocks, 0);
            });
            tensor
        })
        .await;
    for id in 0..geometry.block_count() {
        let mut expected = vec![0.; geometry.block_len()];
        for offset in geometry.block_offsets(id).unwrap() {
            expected[offset] = 1.;
        }
        assert_eq!(tensor.read_logical_block(id).await.unwrap(), expected);
    }
    tensor.validate().await.unwrap();
    tensor.sync().await.unwrap();
    drop(tensor);
    let tensor = Tensor::<TestFE, f32>::load(open_dir(&root).unwrap())
        .await
        .unwrap();
    assert_eq!(tensor.read_value(&[0, 128, 64]).await.unwrap(), 1.);
    cleanup(&root).await;
}

#[tokio::test]
async fn tiny_blocks_share_pages_and_preserve_scalar_bits() {
    let (root, _) = new_dir("sparse_bits").await;
    let dir = freqfs::Cache::new(
        4 * crate::schema::SPARSE_NODE_MEMORY,
        None,
        0,
        std::time::Duration::from_secs(3),
    )
    .load(root.to_path_buf())
    .unwrap();
    let tensor = Tensor::<TestFE, f64>::create(
        dir,
        TensorSchema::new(NumberType::Float(FloatType::F64), shape![257]).unwrap(),
        Layout::Sparse { axis: None },
        4096,
    )
    .await
    .unwrap();
    let values = [
        f64::from_bits(0x7ff8000000000123),
        f64::INFINITY,
        f64::NEG_INFINITY,
        3.,
    ];
    for id in 0..257 {
        tensor
            .replace_logical_block(id, vec![values[id as usize % 4]])
            .await
            .unwrap();
    }
    assert_eq!(tensor.storage.blocks().read().await.len(), 1);
    let values_dir = tensor
        .directory
        .read()
        .await
        .get_dir("values")
        .cloned()
        .unwrap();
    let primary = values_dir.read().await.get_dir("primary").cloned().unwrap();
    assert!(
        primary.read().await.len() < 257,
        "logical blocks must share native pages"
    );
    tensor.validate().await.unwrap();
    tensor.replace_logical_block(63, vec![0.]).await.unwrap();
    tensor.replace_logical_block(64, vec![7.]).await.unwrap();
    tensor.validate().await.unwrap();
    tensor.sync_all().await.unwrap();
    drop(tensor);
    let tensor = Tensor::<TestFE, f64>::load(open_dir(&root).unwrap())
        .await
        .unwrap();
    for id in 0..257 {
        let expected = match id {
            63 => 0.,
            64 => 7.,
            _ => values[id as usize % 4],
        };
        assert_eq!(
            tensor.read_logical_block(id).await.unwrap()[0].to_bits(),
            expected.to_bits()
        );
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn strict_reopen_rejects_malformed_sparse_storage() {
    for fault in 0..6 {
        let (root, dir) = new_dir("sparse_invalid").await;
        let geometry = crate::StorageGeometry::new(
            TensorSchema::new(NumberType::Float(FloatType::F32), shape![1, 3]).unwrap(),
            Layout::Sparse { axis: Some(0) },
            shape![1, 2],
        )
        .unwrap();
        let tensor = Tensor::<TestFE, f32>::create_with_geometry(dir.clone(), geometry)
            .await
            .unwrap();
        tensor.write_value(&[0, 2], 1.).await.unwrap();
        tensor.sync().await.unwrap();
        let owner = tensor.storage.sparse().unwrap();

        match fault {
            0 => crate::tensor::corruption::corrupt_sparse_payload(&tensor, 1).await,
            1 | 2 => {
                let values = if fault == 1 {
                    vec![0.; 2]
                } else {
                    vec![1., 2.]
                };
                owner
                    .values
                    .write()
                    .await
                    .upsert(
                        vec![crate::SparseCell::Key(1)],
                        vec![crate::SparseCell::Payload(values)],
                    )
                    .await
                    .unwrap();
            }
            3 => {
                dir.write().await.delete("values").await;
            }
            4 => {
                owner
                    .blocks
                    .write()
                    .await
                    .create_file(
                        "unexpected_payload".into(),
                        vec![1f32],
                        std::mem::size_of::<TestFE>() + std::mem::size_of::<Vec<f32>>() + 4,
                    )
                    .await
                    .unwrap();
            }
            5 => {
                let values = dir.read().await.get_dir("values").cloned().unwrap();
                let primary = values.read().await.get_dir("primary").cloned().unwrap();
                let name = primary.read().await.iter().next().unwrap().0.clone();
                primary.write().await.delete(&name).await;
            }
            _ => unreachable!(),
        }

        if fault < 3 {
            assert!(tensor.read_logical_block(1).await.is_err());
            assert!(tensor.replace_logical_block(1, vec![0.; 2]).await.is_err());
        }
        assert!(
            Tensor::<TestFE, f32>::load(dir).await.is_err(),
            "fault {fault}"
        );
        cleanup(&root).await;
    }
}
