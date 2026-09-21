//! Native u8 storage and lazy boolean expressions over real filesystem tensors.

use fensor::unary::UnaryOp;
use fensor::{
    DType, Error, Layout, Tensor, TensorArray, TensorCast, TensorElement, TensorFileEntry,
    TensorGeometry, TensorNumeric, TensorRead, TensorSchema, TensorTransform, TensorU8,
    TensorUnaryBoolean, TensorView, TensorViewDecoder, TensorWrite, UnaryView,
};
use futures::TryStreamExt;
use ha_ndarray::{
    Array, ArrayAccess, AxisRange, Buffer, NDArrayNumeric, NDArrayRead, NDArrayUnaryBoolean, axes,
    range, shape,
};

use common::{FsEntry, new_dir};
mod common;

async fn check_consumers<T, O>(view: UnaryView<TensorView<'_, FsEntry, T>, O>, expected: &[u8])
where
    T: TensorElement,
    FsEntry: TensorFileEntry<T>,
    O: UnaryOp<T, Output = u8>,
{
    let blocks: Vec<Vec<u8>> = view.read_blocks().unwrap().try_collect().await.unwrap();
    assert_eq!(blocks.into_iter().flatten().collect::<Vec<_>>(), expected);
    let (out_root, out_dir) = new_dir("boolean_out").await;
    let output = Tensor::<FsEntry, u8>::copy_from(out_dir, &view, 2)
        .await
        .unwrap();
    let (wire_root, wire_dir) = new_dir("boolean_wire").await;
    let decoded: TensorViewDecoder<FsEntry, u8> =
        tbon::de::try_decode(wire_dir, tbon::en::encode(view.view_encoder()).unwrap())
            .await
            .unwrap();
    let decoded = decoded.into_inner();
    assert_eq!(decoded.schema().dtype(), DType::U8);
    for (i, &value) in expected.iter().enumerate() {
        let coord = [i as u64];
        assert_eq!(view.read_value(&coord).await.unwrap(), value);
        assert_eq!(output.read_value(&coord).await.unwrap(), value);
        assert_eq!(decoded.read_value(&coord).await.unwrap(), value);
    }
    if matches!(view.layout(), Layout::Sparse { .. }) {
        let rows: Vec<_> = view
            .read_sparse_elements_in_order(range![AxisRange::In(0, expected.len(), 1)], axes![0])
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(
            rows,
            expected
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0)
                .map(|(i, &v)| (vec![i as u64], v))
                .collect::<Vec<_>>()
        );
    }
    common::cleanup(&out_root).await;
    common::cleanup(&wire_root).await;
}

macro_rules! float_predicates {
    ($name:ident, $t:ty) => {
        #[tokio::test]
        async fn $name() {
            for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
                let (root, dir) = new_dir(stringify!($name)).await;
                let input: Vec<$t> = vec![
                    0.0,
                    -0.0,
                    -2.0,
                    0.2,
                    <$t>::NAN,
                    <$t>::INFINITY,
                    <$t>::NEG_INFINITY,
                ];
                let tensor = Tensor::<FsEntry, $t>::create(
                    dir,
                    TensorSchema::new(<$t>::DTYPE, shape![input.len()]).unwrap(),
                    layout,
                    2,
                )
                .await
                .unwrap();
                for (i, &value) in input.iter().enumerate() {
                    tensor.write_value(&[i as u64], value).await.unwrap();
                }
                macro_rules! check {
                    ($method:ident) => {{
                        let array = ArrayAccess::from(
                            Array::new(Buffer::from(input.clone()), shape![input.len()]).unwrap(),
                        );
                        let mut expected = array
                            .$method()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec();
                        if matches!(layout, Layout::Sparse { .. }) {
                            for (i, &value) in input.iter().enumerate() {
                                if value == 0.0 {
                                    expected[i] = 0;
                                }
                            }
                        }
                        check_consumers(tensor.view().$method().await.unwrap(), &expected).await;
                    }};
                }
                check!(not);
                check!(is_nan);
                check!(is_inf);
                common::cleanup(&root).await;
            }
        }
    };
}
float_predicates!(f32_predicate_parity, f32);
float_predicates!(f64_predicate_parity, f64);

#[tokio::test]
async fn u8_storage_views_not_and_wire_roundtrips() {
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (root, dir) = new_dir("u8_storage").await;
        let tensor: TensorU8<FsEntry> = Tensor::create(
            dir.clone(),
            TensorSchema::new(DType::U8, shape![4]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();
        for (i, value) in [0, 1, 127, 255].into_iter().enumerate() {
            tensor.write_value(&[i as u64], value).await.unwrap();
        }
        let flipped = tensor.view().flip(0).unwrap();
        flipped.write_value(&[0], 254).await.unwrap();
        assert_eq!(tensor.read_value(&[3]).await.unwrap(), 254);
        flipped.write_value(&[0], 255).await.unwrap();
        // Deleting a sparse value, then recreating it, retains zero-write semantics.
        tensor.write_value(&[1], 0).await.unwrap();
        assert_eq!(tensor.read_value(&[1]).await.unwrap(), 0);
        tensor.write_value(&[1], 1).await.unwrap();
        let input = vec![0u8, 1, 127, 255];
        let array = ArrayAccess::from(Array::new(Buffer::from(input), shape![4]).unwrap());
        let mut expected = array
            .not()
            .unwrap()
            .buffer()
            .unwrap()
            .to_slice()
            .unwrap()
            .into_vec();
        if matches!(layout, Layout::Sparse { .. }) {
            expected[0] = 0;
        }
        check_consumers(tensor.view().not().await.unwrap(), &expected).await;
        let (wire_root, wire_dir) = new_dir("u8_native_wire").await;
        let decoded: TensorViewDecoder<FsEntry, u8> =
            tbon::de::try_decode(wire_dir, tbon::en::encode(flipped.view_encoder()).unwrap())
                .await
                .unwrap();
        let decoded = decoded.into_inner();
        for (i, value) in [255, 127, 1, 0].into_iter().enumerate() {
            assert_eq!(decoded.read_value(&[i as u64]).await.unwrap(), value);
        }
        let schema_wire = (tensor.schema().dtype(), vec![4u64], layout);
        let decoded_schema: (DType, Vec<u64>, Layout) =
            tbon::de::try_decode((), tbon::en::encode(&schema_wire).unwrap())
                .await
                .unwrap();
        assert_eq!(decoded_schema, schema_wire);
        dir.sync().await.unwrap();
        drop(flipped);
        drop(tensor);
        drop(dir);
        let reloaded = Tensor::<FsEntry, u8>::load(common::open_dir(&root).unwrap())
            .await
            .unwrap();
        for (i, value) in [0, 1, 127, 255].into_iter().enumerate() {
            assert_eq!(reloaded.read_value(&[i as u64]).await.unwrap(), value);
        }
        assert!(
            Tensor::<FsEntry, f32>::load(common::open_dir(&root).unwrap())
                .await
                .is_err()
        );
        common::cleanup(&root).await;
        common::cleanup(&wire_root).await;
    }
}

#[tokio::test]
async fn sparse_predicate_chain_preserves_support_across_cast_and_false_results() {
    let (root, dir) = new_dir("boolean_chain").await;
    let tensor = Tensor::<FsEntry, f32>::create(
        dir,
        TensorSchema::new(DType::F32, shape![1, 4100]).unwrap(),
        Layout::Sparse { axis: Some(0) },
        4096,
    )
    .await
    .unwrap();
    let expression = tensor
        .view()
        .transpose(None)
        .unwrap()
        .cast()
        .await
        .unwrap()
        .is_nan()
        .await
        .unwrap()
        .flip(0)
        .unwrap()
        .not()
        .await
        .unwrap()
        .squeeze(axes![1])
        .unwrap();
    // The previously constructed expression reads the live source.
    for i in 0..4099 {
        tensor.write_value(&[0, i], 0.2).await.unwrap();
    }
    tensor.write_value(&[0, 4098], f32::NAN).await.unwrap();
    let mut dropped = expression.read_blocks().unwrap();
    assert_eq!(dropped.try_next().await.unwrap().unwrap().len(), 4096);
    drop(dropped);
    let (first, second) = futures::try_join!(
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
        expression.read_blocks().unwrap().try_collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4096, 4]
    );
    let values: Vec<_> = first.into_iter().flatten().collect();
    assert_eq!(&values[..2], &[0, 0]);
    assert!(values[2..].iter().all(|&v| v == 1));
    let rows: Vec<_> = expression
        .read_sparse_elements_in_order(range![AxisRange::In(0, 4100, 1)], axes![0])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows.len(), 4098);
    assert!(rows.iter().all(|(c, v)| c[0] >= 2 && *v == 1));
    let selected = expression.slice(range![AxisRange::In(0, 4, 1)]).unwrap();
    let (out_root, out_dir) = new_dir("boolean_chain_out").await;
    let output = Tensor::copy_from(out_dir, &selected, 2).await.unwrap();
    let (wire_root, wire_dir) = new_dir("boolean_chain_wire").await;
    let decoded: TensorViewDecoder<FsEntry, u8> =
        tbon::de::try_decode(wire_dir, tbon::en::encode(selected.view_encoder()).unwrap())
            .await
            .unwrap();
    let decoded = decoded.into_inner();
    for (i, value) in [0, 0, 1, 1].into_iter().enumerate() {
        assert_eq!(selected.read_value(&[i as u64]).await.unwrap(), value);
        assert_eq!(output.read_value(&[i as u64]).await.unwrap(), value);
        assert_eq!(decoded.read_value(&[i as u64]).await.unwrap(), value);
    }
    let source = tensor
        .view()
        .slice(range![AxisRange::At(0), AxisRange::In(0, 4, 1)])
        .unwrap();
    let (mid_root, mid_dir) = new_dir("boolean_intermediate").await;
    let intermediate = Tensor::copy_from(mid_dir, &source.is_nan().await.unwrap(), 2)
        .await
        .unwrap();
    assert_eq!(
        source
            .is_nan()
            .await
            .unwrap()
            .not()
            .await
            .unwrap()
            .read_value(&[0])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        intermediate
            .view()
            .not()
            .await
            .unwrap()
            .read_value(&[0])
            .await
            .unwrap(),
        0
    );
    for root in [&root, &out_root, &wire_root, &mid_root] {
        common::cleanup(root).await;
    }
}

#[tokio::test]
async fn u8_cache_spill_and_reload() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (root, _) = new_dir("u8_spill").await;
        let cache = freqfs::Cache::<FsEntry>::new(512, None, 0, std::time::Duration::from_secs(1));
        let dir = cache.load(root.clone()).unwrap();
        let tensor = Tensor::<FsEntry, u8>::create(
            dir.clone(),
            TensorSchema::new(DType::U8, shape![2048]).unwrap(),
            Layout::Dense,
            128,
        )
        .await
        .unwrap();
        let mut files = tokio::fs::read_dir(root.join("blocks")).await.unwrap();
        let mut count = 0;
        while files.next_entry().await.unwrap().is_some() {
            count += 1;
        }
        assert!(count > 1, "u8 payloads must count towards cache admission");
        tensor.write_value(&[2047], 255).await.unwrap();
        dir.sync().await.unwrap();
        drop(tensor);
        drop(dir);
        let reloaded = Tensor::<FsEntry, u8>::load(common::open_dir(&root).unwrap())
            .await
            .unwrap();
        assert_eq!(reloaded.read_value(&[2047]).await.unwrap(), 255);
        assert_eq!(reloaded.read_value(&[0]).await.unwrap(), 0);
        common::cleanup(&root).await;
    })
    .await
    .expect("u8 cache pressure must make progress");
}

#[tokio::test]
async fn u8_corruption_and_invalid_wire_values_fail_closed() {
    let (root, dir) = new_dir("u8_corruption").await;
    let tensor = Tensor::<FsEntry, u8>::create(
        dir.clone(),
        TensorSchema::new(DType::U8, shape![4]).unwrap(),
        Layout::Dense,
        4,
    )
    .await
    .unwrap();
    let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
    let file = blocks.read().await.get_file("0").unwrap().clone();
    file.write::<Vec<u8>>().await.unwrap().truncate(1);
    assert!(matches!(
        tensor.read_value(&[0]).await,
        Err(Error::InvalidLayout(_))
    ));
    assert!(matches!(
        tensor.write_value(&[0], 255).await,
        Err(Error::InvalidLayout(_))
    ));
    assert!(
        tensor
            .view()
            .not()
            .await
            .unwrap()
            .read_blocks()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .is_err()
    );
    for value in [-1i16, 256] {
        let (wire_root, wire_dir) = new_dir("u8_invalid_payload").await;
        let wire = (
            (DType::U8, vec![1u64], Layout::Dense, vec![1usize]),
            vec![(vec![0u64], value)],
        );
        let decoded: Result<TensorViewDecoder<FsEntry, u8>, _> =
            tbon::de::try_decode(wire_dir, tbon::en::encode(&wire).unwrap()).await;
        assert!(
            decoded.is_err(),
            "invalid u8 value {value} must be rejected"
        );
        common::cleanup(&wire_root).await;
    }
    let (wire_root, wire_dir) = new_dir("u8_wrong_dtype").await;
    let wire = (
        (DType::F32, vec![1u64], Layout::Dense, vec![1usize]),
        vec![(vec![0u64], 1f32)],
    );
    let decoded: Result<TensorViewDecoder<FsEntry, u8>, _> =
        tbon::de::try_decode(wire_dir, tbon::en::encode(&wire).unwrap()).await;
    assert!(decoded.is_err());
    common::cleanup(&root).await;
    common::cleanup(&wire_root).await;
}
