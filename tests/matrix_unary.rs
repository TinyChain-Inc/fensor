//! Matrix-unary projection, support, geometry, and consumer contracts.

use std::collections::BTreeSet;

use fensor::{
    AxisRange, Error, Layout, Tensor, TensorCompareScalar, TensorElement, TensorFileEntry,
    TensorGeometry, TensorMatMul, TensorMath, TensorMatrixUnary, TensorRead, TensorReduce,
    TensorReduceAll, TensorSchema, TensorSparseIndex, TensorTransform, TensorUnary,
    TensorViewSemantics, TensorWhere, TensorWrite,
};
use futures::TryStreamExt;
use ha_ndarray::{Array, Buffer, MatrixUnary, NDArrayRead};
use number_general::DType;
use smallvec::smallvec;

mod common;

use common::{FsEntry, cleanup, fixture, new_dir};

async fn consumers<V: TensorRead>(
    view: &V,
    expected: &[V::DType],
    equal: impl Fn(V::DType, V::DType) -> bool,
) where
    V::DType: TensorElement,
    FsEntry: TensorFileEntry<V::DType>,
{
    fixture::consumers(view, expected, &equal, &equal).await;
    let mut stream = view.read_coordinate_blocks().unwrap();
    let mut seen = BTreeSet::new();
    while let Some((coords, values)) = stream.try_next().await.unwrap() {
        assert_eq!(coords.len(), values.len());
        for (coord, value) in coords.into_iter().zip(values) {
            let flat = coord
                .iter()
                .zip(view.shape())
                .fold(0, |n, (c, d)| n * d + c);
            assert!(equal(value, expected[flat as usize]), "at {coord:?}");
            assert!(seen.insert(coord), "duplicate output");
        }
    }
    assert_eq!(seen.len(), expected.len());
}

macro_rules! parity {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for dims in [vec![1, 1], vec![3, 3], vec![2, 3, 3], vec![2, 2, 3]] {
                let values: Vec<$t> = (0..dims.iter().product::<u64>())
                    .map(|i| (i % 11) as $t)
                    .collect();
                for sparse in [false, true] {
                    let (root, tensor) = fixture::source(
                        "matrix_unary",
                        dims.clone().into(),
                        fixture::layout(sparse),
                        7,
                        4096,
                        values.clone(),
                    )
                    .await;
                    let array = Array::new(
                        Buffer::from(values.clone()),
                        dims.iter().map(|d| *d as usize).collect(),
                    )
                    .unwrap();
                    let expected = array
                        .clone()
                        .mt()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .to_vec();
                    let mt = tensor.view().mt().await.unwrap();
                    consumers(&mt, &expected, |a, b| a == b).await;
                    assert_eq!(mt.shape()[..dims.len() - 2], dims[..dims.len() - 2]);
                    if dims[dims.len() - 2] == dims[dims.len() - 1] {
                        let expected = array
                            .diag()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .to_vec();
                        let diag = tensor.view().diag().await.unwrap();
                        assert!(!diag.is_base_tensor());
                        assert!(!diag.supports_write_through());
                        assert_eq!(diag.layout(), fixture::layout(sparse));
                        consumers(&diag, &expected, |a, b| a == b).await;
                    } else {
                        assert!(matches!(
                            tensor.view().diag().await,
                            Err(Error::InvalidLayout(_))
                        ));
                    }
                    cleanup(&root).await;
                }
            }
        }
    };
}

parity!(u8_parity, u8);
parity!(f32_parity, f32);
parity!(f64_parity, f64);

#[tokio::test]
async fn shape_errors_and_transpose_write_through() {
    let (root, tensor) = fixture::source(
        "matrix_shapes",
        smallvec![2, 3, 4],
        Layout::Dense,
        7,
        4096,
        0u8..24,
    )
    .await;
    let mt = tensor.view().mt().await.unwrap();
    assert_eq!(mt.shape(), &[2, 4, 3]);
    assert!(mt.supports_write_through());
    mt.write_value(&[1, 3, 2], 255).await.unwrap();
    assert_eq!(tensor.read_value(&[1, 2, 3]).await.unwrap(), 255);
    let vector = tensor.view().reshape(smallvec![24]).unwrap();
    assert!(matches!(vector.mt().await, Err(Error::InvalidLayout(_))));
    assert!(matches!(vector.diag().await, Err(Error::InvalidLayout(_))));
    let scalar = vector.slice(smallvec![AxisRange::At(0)]).unwrap();
    assert!(matches!(scalar.mt().await, Err(Error::InvalidLayout(_))));
    assert!(matches!(scalar.diag().await, Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn support_and_expression_composition() {
    let (root, tensor) = fixture::source(
        "diag_support",
        smallvec![2, 3, 3],
        Layout::Sparse { axis: Some(2) },
        3,
        4096,
        [
            0.2f32, 8., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.2, 0., 0., 0., 0.,
        ],
    )
    .await;
    let rounded = tensor.view().round().await.unwrap();
    let diagonal = rounded.diag().await.unwrap();
    let restored = diagonal.exp().await.unwrap();
    consumers(&restored, &[1., 0., 2f32.exp(), 0., 1., 0.], |a, b| a == b).await;
    let transposed = diagonal.mt().await.unwrap();
    fixture::blocks(&transposed, &[0., 0., 0., 0., 2., 0.], |a, b| a == b).await;
    assert_eq!(diagonal.sum_all().await.unwrap(), 2.);
    assert_eq!(diagonal.layout(), Layout::Sparse { axis: None });
    let selected = diagonal
        .clone()
        .slice(smallvec![AxisRange::At(0), AxisRange::Of(vec![2, 0, 2])])
        .unwrap()
        .flip(0)
        .unwrap();
    consumers(&selected, &[2., 0., 2.], |a, b| a == b).await;
    let before = tensor.view().flip(2).unwrap().diag().await.unwrap();
    fixture::blocks(&before, &[0., 0., 0., 0., 0.2, 0.], |a, b| a == b).await;
    let binary = rounded.add(&rounded).await.unwrap().diag().await.unwrap();
    fixture::blocks(&binary, &[0., 0., 4., 0., 0., 0.], |a, b| a == b).await;
    let condition = tensor.view().gt_scalar(1.).await.unwrap();
    let chosen = condition
        .cond(&tensor.view(), &rounded)
        .await
        .unwrap()
        .diag()
        .await
        .unwrap();
    fixture::blocks(&chosen, &[0., 0., 2., 0., 0., 0.], |a, b| a == b).await;
    // Reducing the batch axis produces a square matrix; projection adds no aggregate boundary.
    let reduced = rounded
        .sum(smallvec![0], false)
        .await
        .unwrap()
        .diag()
        .await
        .unwrap();
    fixture::blocks(&reduced, &[0., 0., 2.], |a, b| a == b).await;
    let (identity_root, identity) = fixture::source(
        "diag_identity",
        smallvec![2, 3, 3],
        Layout::Dense,
        7,
        4096,
        [
            1f32, 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
        ],
    )
    .await;
    let product = rounded
        .matmul(&identity.view())
        .await
        .unwrap()
        .diag()
        .await
        .unwrap();
    fixture::blocks(&product, &[0., 0., 2., 0., 0., 0.], |a, b| a == b).await;
    let reshaped = diagonal
        .reshape(smallvec![3, 2])
        .unwrap()
        .transpose(None)
        .unwrap()
        .unsqueeze(smallvec![0])
        .unwrap()
        .squeeze(smallvec![0])
        .unwrap();
    fixture::blocks(&reshaped, &[0., 2., 0., 0., 0., 0.], |a, b| a == b).await;
    cleanup(&identity_root).await;
    cleanup(&root).await;
}

#[tokio::test]
async fn exception_bits_and_off_diagonal_support() {
    for sparse in [false, true] {
        let (root, tensor) = fixture::source(
            "diag_exceptions",
            smallvec![4, 4],
            fixture::layout(sparse),
            1,
            4096,
            [
                -0f64,
                9.,
                0.,
                0.,
                0.,
                f64::NAN,
                0.,
                0.,
                0.,
                0.,
                f64::INFINITY,
                0.,
                0.,
                0.,
                0.,
                f64::NEG_INFINITY,
            ],
        )
        .await;
        let diag = tensor.view().diag().await.unwrap();
        let expected = [
            if sparse { 0. } else { -0. },
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        consumers(&diag, &expected, |a, b| {
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
        })
        .await;
        cleanup(&root).await;
    }
    let (root, tensor) = fixture::source(
        "off_diagonal",
        smallvec![2, 2],
        Layout::Sparse { axis: None },
        1,
        4096,
        [0f32, 2., 3., 0.],
    )
    .await;
    let diag = tensor.view().diag().await.unwrap().exp().await.unwrap();
    consumers(&diag, &[0., 0.], |a, b| a == b).await;
    cleanup(&root).await;
}

#[tokio::test]
async fn bounded_batches_reuse_and_live_reads() {
    // A broadcasted single stored value exercises stream boundaries without a large matrix allocation.
    let n = 4097;
    let (root, tensor) = fixture::source(
        "diag_boundary",
        smallvec![1, 1],
        Layout::Dense,
        1,
        4096,
        [3u8],
    )
    .await;
    let nested = tensor
        .view()
        .broadcast(smallvec![2, 2, 2])
        .unwrap()
        .diag()
        .await
        .unwrap()
        .diag()
        .await
        .unwrap();
    fixture::blocks(&nested, &[3, 3], |a, b| a == b).await;
    let scalar = nested.slice(smallvec![AxisRange::At(0)]).unwrap();
    assert_eq!(scalar.read_value(&[]).await.unwrap(), 3);
    let expanded = scalar.clone().broadcast(smallvec![2, 2]).unwrap();
    fixture::blocks(&expanded, &[3; 4], |a, b| a == b).await;
    assert!(scalar.read_blocks().is_err());
    assert!(scalar.read_coordinate_blocks().is_err());
    let diag = tensor
        .view()
        .broadcast(smallvec![n, n])
        .unwrap()
        .diag()
        .await
        .unwrap();
    let mut stream = diag.read_coordinate_blocks().unwrap();
    assert!(!stream.try_next().await.unwrap().unwrap().0.is_empty());
    drop(stream);
    let expected = vec![3; n as usize];
    let ((), ()) = futures::join!(
        fixture::blocks(&diag, &expected, |a, b| a == b),
        fixture::blocks(&diag, &expected, |a, b| a == b)
    );
    tensor.write_value(&[0, 0], 127).await.unwrap();
    assert_eq!(diag.read_value(&[n - 1]).await.unwrap(), 127);
    let tiny = diag.slice(smallvec![AxisRange::In(n - 2, n, 1)]).unwrap();
    consumers(&tiny, &[127, 127], |a, b| a == b).await;
    cleanup(&root).await;

    let (root, dir) = new_dir("sparse_diag_boundary").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(f32::dtype(), smallvec![n, n]).unwrap(),
        Layout::Sparse { axis: None },
        1,
    )
    .await
    .unwrap();
    for i in [0, n - 1] {
        tensor.write_value(&[i, i], 0.2).await.unwrap();
    }
    let diag = tensor
        .view()
        .round()
        .await
        .unwrap()
        .diag()
        .await
        .unwrap()
        .exp()
        .await
        .unwrap();
    let mut expected = vec![0.; n as usize];
    expected[0] = 1.;
    expected[n as usize - 1] = 1.;
    fixture::blocks(&diag, &expected, |a, b| a == b).await;
    let entries: Vec<_> = diag
        .read_sparse_elements_in_order(smallvec![AxisRange::In(0, n, 1)], smallvec![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(entries, vec![(vec![0], 1.), (vec![n - 1], 1.)]);
    cleanup(&root).await;
}

#[tokio::test]
async fn huge_selected_diagonal_and_corrupt_blocks() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let n = 1_000_000_000u64;
        let (root, dir) = new_dir("huge_diag").await;
        let tensor = Tensor::<FsEntry, u8>::create(
            dir.clone(),
            TensorSchema::new(u8::dtype(), smallvec![n, n]).unwrap(),
            Layout::Sparse { axis: None },
            1,
        )
        .await
        .unwrap();
        tensor.write_value(&[n - 1, n - 1], 255).await.unwrap();
        tensor.write_value(&[0, 1], 127).await.unwrap();
        let off = tensor.lookup_block_id(&[0, 1]).await.unwrap().unwrap();
        dir.read()
            .await
            .get_dir("blocks")
            .unwrap()
            .write()
            .await
            .delete(&off.to_string())
            .await;
        let diag = tensor.view().diag().await.unwrap();
        assert_eq!(diag.read_value(&[0]).await.unwrap(), 0);
        let entries: Vec<_> = diag
            .read_sparse_elements_in_order(smallvec![AxisRange::In(n - 2, n, 1)], smallvec![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(entries, vec![(vec![n - 1], 255)]);
        assert!(diag.read_value(&[n]).await.is_err());
        assert!(
            diag.read_sparse_elements_in_order(smallvec![AxisRange::At(n)], smallvec![0])
                .await
                .is_err()
        );
        assert!(
            diag.read_sparse_elements_in_order(smallvec![AxisRange::At(0)], smallvec![1])
                .await
                .is_err()
        );
        let needed = tensor
            .lookup_block_id(&[n - 1, n * n - 1])
            .await
            .unwrap()
            .unwrap();
        let file = dir
            .read()
            .await
            .get_dir("blocks")
            .unwrap()
            .read()
            .await
            .get_file(&needed.to_string())
            .cloned()
            .unwrap();
        file.write::<Vec<u8>>().await.unwrap().clear();
        assert!(matches!(
            diag.read_value(&[n - 1]).await,
            Err(Error::InvalidLayout(_))
        ));
        dir.read()
            .await
            .get_dir("blocks")
            .unwrap()
            .write()
            .await
            .delete(&needed.to_string())
            .await;
        assert!(diag.read_value(&[n - 1]).await.is_err());
        cleanup(&root).await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn constrained_cache_and_source_reload() {
    // The stored payload exceeds this cache; many tiny blocks exercise spill/reload.
    let (root, tensor) = fixture::source(
        "diag_spill",
        smallvec![33, 33],
        Layout::Dense,
        1,
        4096,
        (0..33 * 33).map(|i| (i % 256) as u8),
    )
    .await;
    let expected: Vec<_> = (0..33).map(|i| (i * 34 % 256) as u8).collect();
    fixture::blocks(&tensor.view().diag().await.unwrap(), &expected, |a, b| {
        a == b
    })
    .await;
    tensor.sync().await.unwrap();
    drop(tensor);
    let dir = freqfs::Cache::<FsEntry>::new(4096, None, 0, std::time::Duration::from_secs(1))
        .load(root.clone())
        .unwrap();
    let tensor = Tensor::<FsEntry, u8>::load(dir).await.unwrap();
    consumers(&tensor.view().diag().await.unwrap(), &expected, |a, b| {
        a == b
    })
    .await;
    drop(tensor);
    cleanup(&root).await;
}
