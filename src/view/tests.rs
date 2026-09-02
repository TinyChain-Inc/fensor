use std::io;
use std::path::{Path, PathBuf};

use b_table::Node;
use destream::{de, en};
use freqfs::Cache;
use ha_ndarray::{AxisRange, axes, range, shape};
use safecast::as_type;

use crate::schema::row_major_coords;
use crate::{DType, Error, Layout, Tensor, TensorSchema};

use super::*;

#[derive(Clone, Debug)]
enum TestFE {
    Node(Node<u64>),
    F32(Vec<f32>),
    Text(String),
}

impl<'en> en::ToStream<'en> for TestFE {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        match self {
            Self::Node(node) => node.to_stream(encoder),
            Self::F32(values) => values.to_stream(encoder),
            Self::Text(text) => text.to_stream(encoder),
        }
    }
}

// Only ever read via a concrete `AsType` target (mirrors `src/lib.rs::sparse_lifecycle_tests`).
impl de::FromStream for TestFE {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _: (),
        _decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        Err(de::Error::custom(
            "TestFE does not support generic decoding; read via a concrete AsType target",
        ))
    }
}

as_type!(TestFE, Node, Node<u64>);
as_type!(TestFE, F32, Vec<f32>);
as_type!(TestFE, Text, String);

fn unique_tmp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    path.push(format!("fensor_view_tests_{name}_{unique}"));
    path
}

fn open_dir(root: &Path) -> io::Result<freqfs::DirLock<TestFE>> {
    let cache = Cache::<TestFE>::new(1_000_000, None);
    cache.load(root.to_path_buf())
}

async fn new_dir(name: &str) -> (PathBuf, freqfs::DirLock<TestFE>) {
    let root = unique_tmp_dir(name);
    tokio::fs::create_dir(&root).await.expect("create tmp dir");
    let dir = open_dir(&root).expect("load tmp dir");
    (root, dir)
}

async fn cleanup(root: &Path) {
    let _ = tokio::fs::remove_dir_all(root).await;
}

async fn create_dense(
    name: &str,
    shape: Shape,
    max_capacity: usize,
) -> (PathBuf, Tensor<TestFE, f32>) {
    let (root, dir) = new_dir(name).await;
    let schema = TensorSchema::new(DType::F32, shape).expect("schema");
    let tensor = Tensor::<TestFE, f32>::create(dir, schema, Layout::Dense, max_capacity)
        .await
        .expect("create dense");
    (root, tensor)
}

// -- flat_offset correctness ---------------------------------------------------

#[tokio::test]
async fn flat_offset_identity() {
    // [4,5,6], strides [30,6,1], coord [2,1,3] → k = 2*30 + 1*6 + 3*1 = 69
    let (root, tensor) = create_dense("flat_offset_identity", shape![4, 5, 6], 1000).await;
    let view = tensor.view();
    assert_eq!(view.flat_offset(&[2, 1, 3]).expect("offset"), 69);
    cleanup(&root).await;
}

#[tokio::test]
async fn flat_offset_reshape_rank_reducing() {
    // [2,3,4]→[6,4]: new strides [4,1], coord [3,2] → k = 3*4 + 2*1 = 14
    let (root, tensor) =
        create_dense("flat_offset_reshape_rank_reducing", shape![2, 3, 4], 1000).await;
    let view = tensor.view();
    let reshaped = view.reshape(shape![6, 4]).expect("reshape");
    assert_eq!(reshaped.flat_offset(&[3, 2]).expect("offset"), 14);
    cleanup(&root).await;
}

#[tokio::test]
async fn flat_offset_reshape_rank_increasing() {
    // [6,4]→[2,3,4]: new strides [12,4,1], coord [1,0,2] → k = 1*12 + 0*4 + 2*1 = 14
    let (root, tensor) =
        create_dense("flat_offset_reshape_rank_increasing", shape![6, 4], 1000).await;
    let view = tensor.view();
    let reshaped = view.reshape(shape![2, 3, 4]).expect("reshape");
    assert_eq!(reshaped.flat_offset(&[1, 0, 2]).expect("offset"), 14);
    cleanup(&root).await;
}

#[tokio::test]
async fn flat_offset_reshape_same_rank() {
    // [6,4]→[4,6]: new strides [6,1], coord [2,3] → k = 2*6 + 3*1 = 15
    let (root, tensor) = create_dense("flat_offset_reshape_same_rank", shape![6, 4], 1000).await;
    let view = tensor.view();
    let reshaped = view.reshape(shape![4, 6]).expect("reshape");
    assert_eq!(reshaped.flat_offset(&[2, 3]).expect("offset"), 15);
    cleanup(&root).await;
}

// -- slice / transpose flat_offset correctness --------------------------------

#[tokio::test]
async fn slice_then_resolve() {
    // [4,5,6], strides [30,6,1]
    // At(2): base_offset += 2*30 = 60, axis dropped
    // In(1,5,2): base_offset += 1*6 = 6, axis → Stride(12), extent 2
    // Of([0,3,4]): Gather([0,3,4]), extent 3
    // flat_offset([1,2]) = 66 + 1*12 + Gather[2]=4 = 82
    let (root, tensor) = create_dense("slice_then_resolve", shape![4, 5, 6], 1000).await;
    let view = tensor.view();
    let r = range![
        AxisRange::At(2),
        AxisRange::In(1, 5, 2),
        AxisRange::Of(shape![0, 3, 4])
    ];
    let sliced = view.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 3]);
    assert_eq!(sliced.flat_offset(&[1, 2]).expect("offset"), 82);
    cleanup(&root).await;
}

#[tokio::test]
async fn transpose_arbitrary_permutation() {
    // [3,4,5], strides [20,5,1], perm [2,0,1] → axes [Stride(1),Stride(20),Stride(5)]
    // flat_offset([1,2,3]) = 1*1 + 2*20 + 3*5 = 56
    let (root, tensor) =
        create_dense("transpose_arbitrary_permutation", shape![3, 4, 5], 1000).await;
    let view = tensor.view();
    let transposed = view.transpose(Some(axes![2, 0, 1])).expect("transpose");
    assert_eq!(transposed.flat_offset(&[1, 2, 3]).expect("offset"), 56);
    cleanup(&root).await;
}

#[tokio::test]
async fn compose_transpose_and_slice() {
    // [3,4,5], transpose [2,0,1] → axes [Stride(1),Stride(20),Stride(5)], shape [5,3,4]
    // In(1,5,2) on axis 0 (Stride(1)): base_offset+=1, Stride(2), extent 2
    // At(2) on axis 1 (Stride(20)): base_offset+=40, dropped
    // In(0,4,2) on axis 2 (Stride(5)): base_offset+=0, Stride(10), extent 2
    // base_offset=41, axes=[Stride(2),Stride(10)], shape=[2,2]
    // flat_offset([1,1]) = 41 + 2 + 10 = 53
    let (root, tensor) = create_dense("compose_transpose_and_slice", shape![3, 4, 5], 1000).await;
    let view = tensor.view();
    let transposed = view.transpose(Some(axes![2, 0, 1])).expect("transpose");
    let r = range![
        AxisRange::In(1, 5, 2),
        AxisRange::At(2),
        AxisRange::In(0, 4, 2)
    ];
    let sliced = transposed.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.flat_offset(&[1, 1]).expect("offset"), 53);
    cleanup(&root).await;
}

// -- invalid reshape chain guards ---------------------------------------------

#[tokio::test]
async fn transpose_then_reshape_rejected() {
    // transposed [3,4]: axes=[Stride(1),Stride(4)]; contiguous_strides([4,3])=[3,1]; 1≠3
    let (root, tensor) = create_dense("transpose_then_reshape_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let transposed = view.transpose(Some(axes![1, 0])).expect("transpose");
    assert!(matches!(
        transposed.reshape(shape![12]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn strided_slice_then_reshape_rejected() {
    // In(0,6,2) on axis 0 of [6,4]: axes=[Stride(8),Stride(1)]; contiguous_strides([3,4])=[4,1]; 8≠4
    let (root, tensor) =
        create_dense("strided_slice_then_reshape_rejected", shape![6, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(0, 6, 2), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    assert!(matches!(
        sliced.reshape(shape![12]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn gather_slice_then_reshape_rejected() {
    // Of([0,2,4]) on axis 0: Gather axis present → rejected
    let (root, tensor) =
        create_dense("gather_slice_then_reshape_rejected", shape![6, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::Of(shape![0, 2, 4]), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    assert!(matches!(
        sliced.reshape(shape![12]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn at_slice_non_first_axis_then_reshape_rejected() {
    // At(0) on axis 1 of [3,4]: axes=[Stride(4)]; contiguous_strides([3])=[1]; 4≠1
    let (root, tensor) = create_dense(
        "at_slice_non_first_axis_then_reshape_rejected",
        shape![3, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let r = range![AxisRange::In(0, 3, 1), AxisRange::At(0)];
    let sliced = view.slice(r).expect("slice");
    assert!(matches!(
        sliced.reshape(shape![3]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

// -- valid reshape chains -----------------------------------------------------

#[tokio::test]
async fn step1_slice_then_reshape_valid() {
    // In(1,4,1) on axis 0 of [6,4]: strides unchanged [4,1], contiguous for [3,4] ✓
    // base_offset=4; reshape [3,4]→[12]: axes=[Stride(1)]
    // flat_offset([5]) = 4 + 5*1 = 9
    let (root, tensor) = create_dense("step1_slice_then_reshape_valid", shape![6, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(1, 4, 1), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    let reshaped = sliced.reshape(shape![12]).expect("reshape must succeed");
    assert_eq!(reshaped.flat_offset(&[5]).expect("offset"), 9);
    cleanup(&root).await;
}

#[tokio::test]
async fn at_first_axis_then_reshape_valid() {
    // At(1) on axis 0 of [3,4]: base_offset=4, axes=[Stride(1)], contiguous for [4] ✓
    // reshape [4]→[2,2]: axes=[Stride(2),Stride(1)]
    // flat_offset([1,1]) = 4 + 1*2 + 1*1 = 7
    let (root, tensor) = create_dense("at_first_axis_then_reshape_valid", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::At(1), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    let reshaped = sliced.reshape(shape![2, 2]).expect("reshape must succeed");
    assert_eq!(reshaped.flat_offset(&[1, 1]).expect("offset"), 7);
    cleanup(&root).await;
}

#[tokio::test]
async fn reshape_then_slice_flat_offset() {
    // reshape [6,4]→[2,3,4]: axes=[Stride(12),Stride(4),Stride(1)]
    // slice In(0,2,1) / In(0,2,1) / In(0,4,1); shape=[2,2,4]
    // flat_offset([1,1,2]) = 0 + 1*12 + 1*4 + 2*1 = 18
    let (root, tensor) = create_dense("reshape_then_slice_flat_offset", shape![6, 4], 1000).await;
    let view = tensor.view();
    let reshaped = view.reshape(shape![2, 3, 4]).expect("reshape");
    let r = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(0, 2, 1),
        AxisRange::In(0, 4, 1)
    ];
    let sliced = reshaped.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2, 4]);
    assert_eq!(sliced.flat_offset(&[1, 1, 2]).expect("offset"), 18);
    cleanup(&root).await;
}

// -- broadcast flat_offset correctness -----------------------------------------

#[tokio::test]
async fn flat_offset_broadcast_rank_preserving() {
    // [1,3,4], strides [12,4,1] -> identity axes [Stride(12),Stride(4),Stride(1)]
    // broadcast to [2,3,4]: axis 0 (dim 1->2) becomes Broadcast(0), axes 1,2 unchanged
    // flat_offset([a,2,3]) = 0*a + 4*2 + 1*3 = 11, for any a
    let (root, tensor) = create_dense(
        "flat_offset_broadcast_rank_preserving",
        shape![1, 3, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![2, 3, 4])
        .expect("broadcast must be supported");
    assert_eq!(broadcasted.flat_offset(&[0, 2, 3]).expect("offset"), 11);
    assert_eq!(broadcasted.flat_offset(&[1, 2, 3]).expect("offset"), 11);
    cleanup(&root).await;
}

#[tokio::test]
async fn flat_offset_broadcast_rank_expanding() {
    // [3,4], strides [4,1] -> axes [Stride(4),Stride(1)]
    // broadcast to [2,3,4]: rank_diff=1, prepend Broadcast(0); remaining axes unchanged (3==3,4==4)
    // flat_offset([a,2,3]) = 4*2 + 1*3 = 11, for any a
    let (root, tensor) =
        create_dense("flat_offset_broadcast_rank_expanding", shape![3, 4], 1000).await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![2, 3, 4])
        .expect("broadcast must be supported");
    assert_eq!(broadcasted.flat_offset(&[0, 2, 3]).expect("offset"), 11);
    assert_eq!(broadcasted.flat_offset(&[1, 2, 3]).expect("offset"), 11);
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_preserves_offset_of_gathered_size_one_axis() {
    // [4,5,6], strides [30,6,1]. Of([3]) on axis 0 -> Gather([90]) (3*30=90), shape [1,5,6]
    // broadcast axis 0 to 7 -> Broadcast(90) (NOT Broadcast(0))
    // flat_offset([k,2,4]) = 90 + 6*2 + 4 = 106, for any k in 0..7
    let (root, tensor) = create_dense(
        "broadcast_preserves_offset_of_gathered_size_one_axis",
        shape![4, 5, 6],
        1000,
    )
    .await;
    tensor.write_value(&[3, 2, 4], 77.0).await.expect("seed");
    let view = tensor.view();
    let r = range![
        AxisRange::Of(shape![3]),
        AxisRange::In(0, 5, 1),
        AxisRange::In(0, 6, 1)
    ];
    let sliced = view.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[1, 5, 6]);
    let broadcasted = sliced
        .broadcast(shape![7, 5, 6])
        .expect("broadcast must be supported");
    assert_eq!(broadcasted.flat_offset(&[0, 2, 4]).expect("offset"), 106);
    assert_eq!(broadcasted.flat_offset(&[6, 2, 4]).expect("offset"), 106);
    for k in 0..7u64 {
        let v = broadcasted
            .read_value(&[k, 2, 4])
            .await
            .expect("broadcast read");
        assert_eq!(v, 77.0, "k={k}");
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_rejects_lower_rank_target() {
    let (root, tensor) =
        create_dense("broadcast_rejects_lower_rank_target", shape![2, 3], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.broadcast(shape![3]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_rejects_incompatible_dim() {
    let (root, tensor) =
        create_dense("broadcast_rejects_incompatible_dim", shape![2, 3], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.broadcast(shape![2, 5]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_noop_same_shape_preserves_write_through() {
    // broadcasting to the current shape is a no-op: axes unchanged, still fully write-through
    let (root, tensor) = create_dense(
        "broadcast_noop_same_shape_preserves_write_through",
        shape![2, 3],
        1000,
    )
    .await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![2, 3])
        .expect("no-op broadcast must be supported");
    assert!(broadcasted.supports_write_through());
    broadcasted
        .write_value(&[1, 2], 5.0)
        .await
        .expect("write through no-op broadcast");
    assert_eq!(tensor.read_value(&[1, 2]).await.expect("base read"), 5.0);
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_then_reshape_rejected() {
    // [1,4] broadcast to [3,4]: axis 0 becomes Broadcast(0) -> view is no longer c-contiguous
    let (root, tensor) = create_dense("broadcast_then_reshape_rejected", shape![1, 4], 1000).await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![3, 4])
        .expect("broadcast must be supported");
    assert!(matches!(
        broadcasted.reshape(shape![12]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_expands_middle_or_trailing_axis_with_rank_increase() {
    // [3,1,4], strides [4,4,1] -> axes [Stride(4),Stride(4),Stride(1)]
    // broadcast to [2,3,5,4]: rank_diff=1 prepends Broadcast(0); axis (dim 3->3) clones Stride(4);
    // middle axis (dim 1->5) becomes Broadcast(0); trailing axis (dim 4->4) clones Stride(1)
    // flat_offset([i,j,k,l]) = 4*j + l, independent of i and k
    let (root, tensor) = create_dense(
        "broadcast_expands_middle_or_trailing_axis_with_rank_increase",
        shape![3, 1, 4],
        1000,
    )
    .await;
    tensor.write_value(&[2, 0, 2], 55.0).await.expect("seed");
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![2, 3, 5, 4])
        .expect("broadcast must be supported");
    assert_eq!(broadcasted.shape(), &[2, 3, 5, 4]);
    assert_eq!(broadcasted.flat_offset(&[1, 2, 3, 2]).expect("offset"), 10);
    assert_eq!(broadcasted.flat_offset(&[0, 2, 0, 2]).expect("offset"), 10);
    for i in 0..2u64 {
        for k in 0..5u64 {
            let v = broadcasted.read_value(&[i, 2, k, 2]).await.expect("read");
            assert_eq!(v, 55.0, "i={i} k={k}");
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_twice_passes_through_existing_broadcast_axis() {
    // [4,4], strides [4,1]. Of([3]) on axis 0 -> Gather([12]) (3*4=12), shape [1,4]
    // first broadcast to [5,4]: axis 0 (dim 1->5) becomes Broadcast(12) (not 0)
    // second broadcast to [2,5,4]: prepends Broadcast(0); the existing Broadcast(12) axis
    // (dim 5->5, unchanged) must clone through as Broadcast(12), not reset to 0
    // flat_offset([i,j,k]) = 12 + k, independent of i and j
    let (root, tensor) = create_dense(
        "broadcast_twice_passes_through_existing_broadcast_axis",
        shape![4, 4],
        1000,
    )
    .await;
    for k in 0..4u64 {
        tensor
            .write_value(&[3, k], k as f32 * 1.5)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let r = range![AxisRange::Of(shape![3]), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    let once = sliced
        .broadcast(shape![5, 4])
        .expect("first broadcast must be supported");
    let twice = once
        .broadcast(shape![2, 5, 4])
        .expect("second broadcast must be supported");
    assert_eq!(twice.shape(), &[2, 5, 4]);
    for i in 0..2u64 {
        for j in 0..5u64 {
            for k in 0..4u64 {
                assert_eq!(
                    twice.flat_offset(&[i, j, k]).expect("offset"),
                    12 + k as i64,
                    "i={i} j={j} k={k}"
                );
                let v = twice.read_value(&[i, j, k]).await.expect("read");
                assert_eq!(v, k as f32 * 1.5, "i={i} j={j} k={k}");
            }
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn broadcast_then_transpose_preserves_constant() {
    // [4,4], Of([3]) -> Gather([12]); broadcast to [5,4] -> axes [Broadcast(12), Stride(1)]
    // transpose([1,0]) -> axes [Stride(1), Broadcast(12)], shape [4,5]
    // flat_offset([k,j]) = k + 12, independent of j
    let (root, tensor) = create_dense(
        "broadcast_then_transpose_preserves_constant",
        shape![4, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let r = range![AxisRange::Of(shape![3]), AxisRange::In(0, 4, 1)];
    let sliced = view.slice(r).expect("slice");
    let broadcasted = sliced
        .broadcast(shape![5, 4])
        .expect("broadcast must be supported");
    let transposed = broadcasted.transpose(Some(axes![1, 0])).expect("transpose");
    assert_eq!(transposed.shape(), &[4, 5]);
    for k in 0..4u64 {
        for j in 0..5u64 {
            assert_eq!(
                transposed.flat_offset(&[k, j]).expect("offset"),
                12 + k as i64,
                "k={k} j={j}"
            );
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn slice_of_after_broadcast_preserves_constant() {
    // [1,4] -> broadcast to [3,4]: axis 0 becomes Broadcast(0)
    // Of([0,2]) slice on that Broadcast axis must stay Broadcast(0), not become a Gather
    // flat_offset([x,y]) = y, independent of x
    let (root, tensor) = create_dense(
        "slice_of_after_broadcast_preserves_constant",
        shape![1, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![3, 4])
        .expect("broadcast must be supported");
    let r = range![AxisRange::Of(shape![0, 2]), AxisRange::In(0, 4, 1)];
    let sliced = broadcasted.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 4]);
    for x in 0..2u64 {
        for y in 0..4u64 {
            assert_eq!(
                sliced.flat_offset(&[x, y]).expect("offset"),
                y as i64,
                "x={x} y={y}"
            );
        }
    }
    cleanup(&root).await;
}

// -- flip --------------------------------------------------------------------

#[tokio::test]
async fn flip_stride_axis_reverses_offset() {
    // [4,5,6], strides [30,6,1]; flip axis 2 (dim 6, stride 1) -> offset_delta=5, Stride(-1)
    // flat_offset([0,0,0]) = 5 + 0*(-1) = 5; flat_offset([0,0,5]) = 5 + 5*(-1) = 0
    let (root, tensor) =
        create_dense("flip_stride_axis_reverses_offset", shape![4, 5, 6], 1000).await;
    let view = tensor.view();
    let flipped = view.flip(2).expect("flip");
    assert_eq!(flipped.flat_offset(&[0, 0, 0]).expect("offset"), 5);
    assert_eq!(flipped.flat_offset(&[0, 0, 5]).expect("offset"), 0);
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_then_flip_is_identity() {
    // flipping the same axis twice restores the original Stride and base_offset
    let shape = shape![4, 5, 6];
    let (root, tensor) = create_dense("flip_then_flip_is_identity", shape.clone(), 1000).await;
    let view = tensor.view();
    let flipped_twice = view
        .clone()
        .flip(2)
        .expect("flip")
        .flip(2)
        .expect("flip again");
    let coords = row_major_coords(&shape).expect("coords for shape");

    for c in coords {
        assert_eq!(
            flipped_twice.flat_offset(&c).expect("offset"),
            view.flat_offset(&c).expect("offset"),
            "{c:?}"
        );
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_on_broadcast_axis_is_noop() {
    // [1,4] broadcast to [3,4]: axis 0 becomes Broadcast(0); flip axis 0 leaves it unchanged
    let (root, tensor) = create_dense("flip_on_broadcast_axis_is_noop", shape![1, 4], 1000).await;
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![3, 4])
        .expect("broadcast must be supported");
    let before: Vec<i64> = (0..3u64)
        .map(|x| broadcasted.flat_offset(&[x, 2]).expect("offset"))
        .collect();
    let flipped = broadcasted.flip(0).expect("flip");
    let after: Vec<i64> = (0..3u64)
        .map(|x| flipped.flat_offset(&[x, 2]).expect("offset"))
        .collect();
    assert_eq!(before, after);
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_on_gather_axis_reverses_table() {
    // [4,4], strides [4,1]. Of([1,3]) on axis 0 -> Gather([4,12]), shape [2,4]
    // flip axis 0 reverses the table -> Gather([12,4])
    let (root, tensor) =
        create_dense("flip_on_gather_axis_reverses_table", shape![4, 4], 1000).await;
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![1, 3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    let before: Vec<i64> = (0..2u64)
        .map(|a| gathered.flat_offset(&[a, 0]).expect("offset"))
        .collect();
    let flipped = gathered.flip(0).expect("flip");
    let after: Vec<i64> = (0..2u64)
        .map(|a| flipped.flat_offset(&[a, 0]).expect("offset"))
        .collect();
    let expected: Vec<i64> = before.into_iter().rev().collect();
    assert_eq!(after, expected);
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_invalid_axis_out_of_bounds() {
    let (root, tensor) =
        create_dense("flip_invalid_axis_out_of_bounds", shape![3, 4, 5], 1000).await;
    let view = tensor.view();
    assert!(matches!(view.flip(3), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_then_reshape_rejected() {
    // flip axis 1 of [3,4]: axes=[Stride(4),Stride(-1)]; negative stride is never c-contiguous
    let (root, tensor) = create_dense("flip_then_reshape_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let flipped = view.flip(1).expect("flip");
    assert!(matches!(
        flipped.reshape(shape![12]),
        Err(Error::Unsupported(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn flip_write_through_roundtrip() {
    // flip axis 1 (dim 4) of [3,4]; write through flipped[1,1] must land on tensor[1,2]
    let (root, tensor) = create_dense("flip_write_through_roundtrip", shape![3, 4], 1000).await;
    let view = tensor.view();
    let flipped = view.flip(1).expect("flip");
    flipped
        .write_value(&[1, 1], 9.0)
        .await
        .expect("write through flipped");
    assert_eq!(tensor.read_value(&[1, 2]).await.expect("base read"), 9.0);
    cleanup(&root).await;
}

// -- slice() baseline coverage: AxisContrib chaining gaps ----------------------

#[tokio::test]
async fn at_slice_on_broadcast_axis() {
    // [1,4] -> broadcast to [3,4]: axis 0 becomes Broadcast(0)
    // At(1) on that Broadcast axis contributes 0 regardless of index, and the axis is dropped
    // flat_offset([y]) = y
    let (root, tensor) = create_dense("at_slice_on_broadcast_axis", shape![1, 4], 1000).await;
    for c in 0..4u64 {
        tensor
            .write_value(&[0, c], c as f32 * 2.0)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![3, 4])
        .expect("broadcast must be supported");
    let r = range![AxisRange::At(1), AxisRange::In(0, 4, 1)];
    let sliced = broadcasted.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[4]);
    for y in 0..4u64 {
        assert_eq!(sliced.flat_offset(&[y]).expect("offset"), y as i64);
        let v = sliced.read_value(&[y]).await.expect("read");
        assert_eq!(v, y as f32 * 2.0, "y={y}");
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn in_slice_on_broadcast_axis() {
    // [1,4] -> broadcast to [5,4]: axis 0 becomes Broadcast(0)
    // In(1,4,1) on that Broadcast axis stays Broadcast(0) with extent 3, still independent of index
    let (root, tensor) = create_dense("in_slice_on_broadcast_axis", shape![1, 4], 1000).await;
    for c in 0..4u64 {
        tensor
            .write_value(&[0, c], c as f32 * 2.0)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let broadcasted = view
        .broadcast(shape![5, 4])
        .expect("broadcast must be supported");
    let r = range![AxisRange::In(1, 4, 1), AxisRange::In(0, 4, 1)];
    let sliced = broadcasted.slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[3, 4]);
    for a in 0..3u64 {
        for y in 0..4u64 {
            assert_eq!(
                sliced.flat_offset(&[a, y]).expect("offset"),
                y as i64,
                "a={a} y={y}"
            );
            let v = sliced.read_value(&[a, y]).await.expect("read");
            assert_eq!(v, y as f32 * 2.0, "a={a} y={y}");
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn at_slice_on_gather_axis() {
    // [4,4], strides [4,1]. Of([1,3]) on axis 0 -> Gather([4,12]) (1*4=4, 3*4=12), shape [2,4]
    // At(1) on that Gather axis selects g[1]=12 (base row 3) and drops the axis
    // flat_offset([y]) = 12 + y
    let (root, tensor) = create_dense("at_slice_on_gather_axis", shape![4, 4], 1000).await;
    for y in 0..4u64 {
        tensor
            .write_value(&[3, y], y as f32 * 3.0)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![1, 3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    assert_eq!(gathered.shape(), &[2, 4]);
    let r = range![AxisRange::At(1), AxisRange::In(0, 4, 1)];
    let sliced = gathered.slice(r).expect("slice At on gather axis");
    assert_eq!(sliced.shape(), &[4]);
    for y in 0..4u64 {
        assert_eq!(
            sliced.flat_offset(&[y]).expect("offset"),
            12 + y as i64,
            "y={y}"
        );
        let v = sliced.read_value(&[y]).await.expect("read");
        assert_eq!(v, y as f32 * 3.0, "y={y}");
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn in_slice_on_gather_axis() {
    // [4,4], strides [4,1]. Of([1,3]) on axis 0 -> Gather([4,12]), shape [2,4]
    // In(1,2,1) on that Gather axis selects g[1]=12 (base row 3) via a sub-range, extent 1
    // flat_offset([0,y]) = 12 + y
    let (root, tensor) = create_dense("in_slice_on_gather_axis", shape![4, 4], 1000).await;
    for y in 0..4u64 {
        tensor
            .write_value(&[3, y], y as f32 * 3.0)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![1, 3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    let r = range![AxisRange::In(1, 2, 1), AxisRange::In(0, 4, 1)];
    let sliced = gathered.slice(r).expect("slice In on gather axis");
    assert_eq!(sliced.shape(), &[1, 4]);
    for y in 0..4u64 {
        assert_eq!(
            sliced.flat_offset(&[0, y]).expect("offset"),
            12 + y as i64,
            "y={y}"
        );
        let v = sliced.read_value(&[0, y]).await.expect("read");
        assert_eq!(v, y as f32 * 3.0, "y={y}");
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn of_slice_on_gather_axis() {
    // [4,4], strides [4,1]. Of([1,3]) on axis 0 -> Gather([4,12]), shape [2,4]
    // Of([1,0]) on that Gather axis reorders: new Gather([12,4]) (picks g[1] then g[0])
    // flat_offset([0,y]) = 12+y (base row 3), flat_offset([1,y]) = 4+y (base row 1)
    let (root, tensor) = create_dense("of_slice_on_gather_axis", shape![4, 4], 1000).await;
    for y in 0..4u64 {
        tensor
            .write_value(&[1, y], y as f32 * 5.0)
            .await
            .expect("seed row1");
        tensor
            .write_value(&[3, y], y as f32 * 3.0)
            .await
            .expect("seed row3");
    }
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![1, 3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    let r = range![AxisRange::Of(shape![1, 0]), AxisRange::In(0, 4, 1)];
    let sliced = gathered.slice(r).expect("slice Of on gather axis");
    assert_eq!(sliced.shape(), &[2, 4]);
    for y in 0..4u64 {
        assert_eq!(
            sliced.flat_offset(&[0, y]).expect("offset"),
            12 + y as i64,
            "y={y}"
        );
        assert_eq!(
            sliced.flat_offset(&[1, y]).expect("offset"),
            4 + y as i64,
            "y={y}"
        );
        assert_eq!(
            sliced.read_value(&[0, y]).await.expect("read"),
            y as f32 * 3.0,
            "y={y}"
        );
        assert_eq!(
            sliced.read_value(&[1, y]).await.expect("read"),
            y as f32 * 5.0,
            "y={y}"
        );
    }
    cleanup(&root).await;
}

// -- slice() baseline coverage: validation error paths --------------------------

#[tokio::test]
async fn slice_rank_mismatch_rejected() {
    let (root, tensor) = create_dense("slice_rank_mismatch_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(0, 3, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn at_slice_out_of_bounds_rejected() {
    let (root, tensor) = create_dense("at_slice_out_of_bounds_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::At(3), AxisRange::In(0, 4, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn in_slice_step_zero_rejected() {
    let (root, tensor) = create_dense("in_slice_step_zero_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(0, 3, 0), AxisRange::In(0, 4, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn in_slice_start_after_stop_rejected() {
    let (root, tensor) =
        create_dense("in_slice_start_after_stop_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(2, 1, 1), AxisRange::In(0, 4, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn in_slice_stop_exceeds_dim_rejected() {
    let (root, tensor) =
        create_dense("in_slice_stop_exceeds_dim_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::In(0, 5, 1), AxisRange::In(0, 4, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

#[tokio::test]
async fn of_slice_out_of_bounds_rejected() {
    let (root, tensor) = create_dense("of_slice_out_of_bounds_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    let r = range![AxisRange::Of(shape![0, 5]), AxisRange::In(0, 4, 1)];
    assert!(matches!(view.slice(r), Err(Error::InvalidLayout(_))));
    cleanup(&root).await;
}

// -- squeeze/unsqueeze --

#[tokio::test]
async fn squeeze_stride_axis_folds_zero_offset() {
    // shape [3,1,4], strides [4,4,1] (from schema::contiguous_strides)
    // squeeze axis 1 (dim=1) -> shape [3,4]
    // flat_offset([x,z]) should equal flat_offset([x,0,z]) on original
    let (root, tensor) = create_dense(
        "squeeze_stride_axis_folds_zero_offset",
        shape![3, 1, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let squeezed = view
        .clone()
        .squeeze(axes![1])
        .expect("squeeze must be supported");
    assert_eq!(squeezed.shape(), &[3, 4]);

    // For a stride axis with dim=1, index 0 contributes 0 to the offset
    // so flat_offset on squeezed [x,z] should match original [x,0,z]
    for x in 0..3u64 {
        for z in 0..4u64 {
            let orig_offset = view.flat_offset(&[x, 0, z]).expect("orig offset");
            let squeeze_offset = squeezed.flat_offset(&[x, z]).expect("squeeze offset");
            assert_eq!(orig_offset, squeeze_offset, "x={}, z={}", x, z);
        }
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_gather_singleton_folds_offset() {
    // [4,4], strides [4,1]. Of([3]) on axis 0 -> Gather([12]), shape [1,4]
    // squeeze axis 0 -> shape [4]
    // flat_offset([y]) = 12 + y
    let (root, tensor) =
        create_dense("squeeze_gather_singleton_folds_offset", shape![4, 4], 1000).await;
    for y in 0..4u64 {
        tensor
            .write_value(&[3, y], y as f32 * 2.0)
            .await
            .expect("seed");
    }
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    assert_eq!(gathered.shape(), &[1, 4]);
    let squeezed = gathered
        .squeeze(axes![0])
        .expect("squeeze must be supported");
    assert_eq!(squeezed.shape(), &[4]);

    for y in 0..4u64 {
        assert_eq!(
            squeezed.flat_offset(&[y]).expect("offset"),
            12 + y as i64,
            "y={}",
            y
        );
        let v = squeezed.read_value(&[y]).await.expect("read");
        assert_eq!(v, y as f32 * 2.0, "y={}", y);
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_broadcast_axis_folds_constant() {
    // [4,5,6], strides [30,6,1]. Of([3]) on axis 0 -> Gather([90]), shape [1,5,6].
    // Broadcasting axis 0 to its OWN size (1 -> 1) still converts it to
    // AxisContrib::Broadcast(90) (broadcast() always wraps an old_dim==1
    // axis in Broadcast, regardless of the target size), so this is how a
    // dim==1 axis ends up typed as Broadcast rather than Gather/Stride.
    // squeeze(axes![0]) must fold that constant 90 into base_offset.
    let (root, tensor) = create_dense(
        "squeeze_broadcast_axis_folds_constant",
        shape![4, 5, 6],
        1000,
    )
    .await;
    tensor.write_value(&[3, 2, 4], 77.0).await.expect("seed");
    let view = tensor.view();
    let gathered = view
        .slice(range![
            AxisRange::Of(shape![3]),
            AxisRange::In(0, 5, 1),
            AxisRange::In(0, 6, 1)
        ])
        .expect("slice Of");
    assert_eq!(gathered.shape(), &[1, 5, 6]);

    let broadcasted = gathered
        .broadcast(shape![1, 5, 6])
        .expect("same-size broadcast must be supported");

    let squeezed = broadcasted
        .squeeze(axes![0])
        .expect("squeeze must be supported");
    assert_eq!(squeezed.shape(), &[5, 6]);

    for y in 0..5u64 {
        for z in 0..6u64 {
            assert_eq!(
                squeezed.flat_offset(&[y, z]).expect("offset"),
                90 + 6 * y as i64 + z as i64,
                "y={y} z={z}"
            );
        }
    }
    let v = squeezed.read_value(&[2, 4]).await.expect("read");
    assert_eq!(v, 77.0);
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_empty_axes_rejected() {
    let (root, tensor) = create_dense("squeeze_empty_axes_rejected", shape![3, 1, 4], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.squeeze(axes![]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_all_axes_rejected() {
    let (root, tensor) = create_dense("squeeze_all_axes_rejected", shape![1, 1], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.squeeze(axes![0, 1]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_duplicate_axis_rejected() {
    let (root, tensor) =
        create_dense("squeeze_duplicate_axis_rejected", shape![1, 3, 1, 4], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.squeeze(axes![0, 0]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_axis_out_of_bounds_rejected() {
    let (root, tensor) =
        create_dense("squeeze_axis_out_of_bounds_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.squeeze(axes![5]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_removes_gather_axis_restores_write_through() {
    // [4,4], strides [4,1]. Of([3]) on axis 0 -> Gather([12]), shape [1,4]
    // Before squeeze: supports_write_through() is false (has Gather)
    // After squeeze: supports_write_through() should be true, and write_value should work
    let (root, tensor) = create_dense(
        "squeeze_removes_gather_axis_restores_write_through",
        shape![4, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let gathered = view
        .slice(range![AxisRange::Of(shape![3]), AxisRange::In(0, 4, 1)])
        .expect("slice Of");
    assert!(
        !gathered.supports_write_through(),
        "gathered should have Gather axis"
    );

    let squeezed = gathered
        .squeeze(axes![0])
        .expect("squeeze must be supported");
    assert!(
        squeezed.supports_write_through(),
        "squeezed should support write_through"
    );

    squeezed.write_value(&[2], 9.0).await.expect("write");
    let v = tensor.read_value(&[3, 2]).await.expect("read from base");
    assert_eq!(v, 9.0);
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_uses_contiguous_stride_at_insertion_point() {
    // shape [3,4], .unsqueeze(axes![0,1]) -> shape [1,3,1,4]
    // strides should be [12,4,4,1] (from contiguous_strides([1,3,1,4]))
    // flat_offset([0,1,0,2]) = 0*12 + 1*4 + 0*4 + 2*1 = 6
    let (root, tensor) = create_dense(
        "unsqueeze_uses_contiguous_stride_at_insertion_point",
        shape![3, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let unsqueezed = view
        .unsqueeze(axes![0, 1])
        .expect("unsqueeze must be supported");
    assert_eq!(unsqueezed.shape(), &[1, 3, 1, 4]);

    // Verify expected strides via flat_offset calculations
    // flat_offset([0,1,0,2]) should be 1*4 + 2*1 = 6
    assert_eq!(
        unsqueezed.flat_offset(&[0, 1, 0, 2]).expect("offset"),
        6,
        "flat_offset([0,1,0,2])"
    );
    assert_eq!(
        unsqueezed.flat_offset(&[0, 2, 0, 3]).expect("offset"),
        2 * 4 + 3,
        "flat_offset([0,2,0,3])"
    );
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_then_reshape_preserves_contiguity() {
    // shape [3,4], .unsqueeze(axes![0]) -> shape [1,3,4]
    // then .reshape(shape![12]) should succeed (contiguity is preserved)
    let (root, tensor) = create_dense(
        "unsqueeze_then_reshape_preserves_contiguity",
        shape![3, 4],
        1000,
    )
    .await;
    let view = tensor.view();
    let unsqueezed = view
        .unsqueeze(axes![0])
        .expect("unsqueeze must be supported");
    assert_eq!(unsqueezed.shape(), &[1, 3, 4]);

    // This should succeed, proving contiguity is preserved
    let reshaped = unsqueezed
        .reshape(shape![12])
        .expect("reshape after unsqueeze must succeed");
    assert_eq!(reshaped.shape(), &[12]);
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_before_first_middle_and_last_original_axis() {
    // shape [2,3,4], test unsqueeze at positions 0, 1, 2
    let (root, tensor) = create_dense(
        "unsqueeze_before_first_middle_and_last_original_axis",
        shape![2, 3, 4],
        1000,
    )
    .await;
    let view = tensor.view();

    let unsqueeze_first = view.clone().unsqueeze(axes![0]).expect("unsqueeze at 0");
    assert_eq!(unsqueeze_first.shape(), &[1, 2, 3, 4]);

    let unsqueeze_middle = view.clone().unsqueeze(axes![1]).expect("unsqueeze at 1");
    assert_eq!(unsqueeze_middle.shape(), &[2, 1, 3, 4]);

    let unsqueeze_last = view.clone().unsqueeze(axes![2]).expect("unsqueeze at 2");
    assert_eq!(unsqueeze_last.shape(), &[2, 3, 1, 4]);

    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_empty_axes_rejected() {
    let (root, tensor) = create_dense("unsqueeze_empty_axes_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.unsqueeze(axes![]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_duplicate_axis_rejected() {
    let (root, tensor) =
        create_dense("unsqueeze_duplicate_axis_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    assert!(matches!(
        view.unsqueeze(axes![0, 0]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_axis_out_of_bounds_rejected() {
    let (root, tensor) =
        create_dense("unsqueeze_axis_out_of_bounds_rejected", shape![3, 4], 1000).await;
    let view = tensor.view();
    // shape is rank 2, so valid axes are 0,1 only
    assert!(matches!(
        view.clone().unsqueeze(axes![2]),
        Err(Error::InvalidLayout(_))
    ));
    assert!(matches!(
        view.unsqueeze(axes![3]),
        Err(Error::InvalidLayout(_))
    ));
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_write_through_roundtrip() {
    // shape [3,4], .unsqueeze(axes![0]) -> shape [1,3,4]
    // write_value at [0,1,2] should write to [1,2] on base tensor
    let (root, tensor) =
        create_dense("unsqueeze_write_through_roundtrip", shape![3, 4], 1000).await;
    let view = tensor.view();
    let unsqueezed = view
        .unsqueeze(axes![0])
        .expect("unsqueeze must be supported");
    assert_eq!(unsqueezed.shape(), &[1, 3, 4]);

    unsqueezed
        .write_value(&[0, 1, 2], 9.0)
        .await
        .expect("write");
    let v = tensor.read_value(&[1, 2]).await.expect("read from base");
    assert_eq!(v, 9.0);
    cleanup(&root).await;
}

#[tokio::test]
async fn squeeze_then_unsqueeze_round_trip() {
    // shape [3,1,4], squeeze axis 1 -> [3,4]
    // then unsqueeze axes![1] -> [3,1,4]
    // assert flat_offset for all coordinates matches original
    let shape = shape![3, 1, 4];
    let (root, tensor) =
        create_dense("squeeze_then_unsqueeze_round_trip", shape.clone(), 1000).await;
    let view = tensor.view();
    let squeezed = view
        .clone()
        .squeeze(axes![1])
        .expect("squeeze must be supported");
    assert_eq!(squeezed.shape(), &[3, 4]);

    let restored = squeezed
        .unsqueeze(axes![1])
        .expect("unsqueeze must be supported");
    assert_eq!(restored.shape(), &[3, 1, 4]);

    // Verify flat_offset matches for all coordinates
    let coords = row_major_coords(&shape).expect("coords");
    for c in coords {
        assert_eq!(
            restored.flat_offset(&c).expect("restored offset"),
            view.flat_offset(&c).expect("original offset"),
            "{:?}",
            c
        );
    }
    cleanup(&root).await;
}

#[tokio::test]
async fn unsqueeze_then_squeeze_round_trip() {
    // shape [3,4], unsqueeze axes![0] -> [1,3,4]
    // then squeeze axes![0] -> [3,4]
    // assert flat_offset for all coordinates matches original
    let shape = shape![3, 4];
    let (root, tensor) =
        create_dense("unsqueeze_then_squeeze_round_trip", shape.clone(), 1000).await;
    let view = tensor.view();
    let unsqueezed = view
        .clone()
        .unsqueeze(axes![0])
        .expect("unsqueeze must be supported");
    assert_eq!(unsqueezed.shape(), &[1, 3, 4]);

    let restored = unsqueezed
        .squeeze(axes![0])
        .expect("squeeze must be supported");
    assert_eq!(restored.shape(), &[3, 4]);

    // Verify flat_offset matches for all coordinates
    let coords = row_major_coords(&shape).expect("coords");
    for c in coords {
        assert_eq!(
            restored.flat_offset(&c).expect("restored offset"),
            view.flat_offset(&c).expect("original offset"),
            "{:?}",
            c
        );
    }
    cleanup(&root).await;
}
