use std::io;
use std::path::PathBuf;

use b_table::Node;
use destream::{de, en};
use fensor::{
    DType, Layout, Tensor, TensorRead, TensorSchema, TensorWrite, contiguous_strides,
};
use freqfs::Cache;
use ha_ndarray::{shape, Shape};
use safecast::as_type;

#[derive(Clone)]
enum FsEntry {
    Node(Node<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Text(String),
}

impl<'en> en::ToStream<'en> for FsEntry {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Node(node) => en::IntoStream::into_stream(
                (
                    0u8,
                    Some(node.clone()),
                    None::<Vec<f32>>,
                    None::<Vec<f64>>,
                    None::<String>,
                ),
                encoder,
            ),
            Self::F32(values) => en::IntoStream::into_stream(
                (
                    1u8,
                    None::<Node<u64>>,
                    Some(values.clone()),
                    None::<Vec<f64>>,
                    None::<String>,
                ),
                encoder,
            ),
            Self::F64(values) => en::IntoStream::into_stream(
                (
                    2u8,
                    None::<Node<u64>>,
                    None::<Vec<f32>>,
                    Some(values.clone()),
                    None::<String>,
                ),
                encoder,
            ),
            Self::Text(text) => en::IntoStream::into_stream(
                (
                    3u8,
                    None::<Node<u64>>,
                    None::<Vec<f32>>,
                    None::<Vec<f64>>,
                    Some(text.clone()),
                ),
                encoder,
            ),
        }
    }
}

impl de::FromStream for FsEntry {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, node, f32_values, f64_values, text): (
            u8,
            Option<Node<u64>>,
            Option<Vec<f32>>,
            Option<Vec<f64>>,
            Option<String>,
        ) = <(
            u8,
            Option<Node<u64>>,
            Option<Vec<f32>>,
            Option<Vec<f64>>,
            Option<String>,
        )>::from_stream((), decoder)
        .await?;

        match tag {
            0 => node
                .map(Self::Node)
                .ok_or_else(|| de::Error::custom("missing node payload")),
            1 => f32_values
                .map(Self::F32)
                .ok_or_else(|| de::Error::custom("missing f32 payload")),
            2 => f64_values
                .map(Self::F64)
                .ok_or_else(|| de::Error::custom("missing f64 payload")),
            3 => text
                .map(Self::Text)
                .ok_or_else(|| de::Error::custom("missing string payload")),
            _ => Err(de::Error::custom(format!("unknown fs entry tag {tag}"))),
        }
    }
}

as_type!(FsEntry, Node, Node<u64>);
as_type!(FsEntry, F32, Vec<f32>);
as_type!(FsEntry, F64, Vec<f64>);
as_type!(FsEntry, Text, String);

fn unique_tmp_dir() -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    path.push(format!("fensor_f64_roundtrip_{unique}"));
    path
}

#[tokio::test]
async fn filesystem_tensor_f64_write_read_roundtrip() -> io::Result<()> {
    let root = unique_tmp_dir();
    tokio::fs::create_dir(&root).await?;

    let cache = Cache::<FsEntry>::new(1_000_000, None);
    let dir = cache.load(root.clone())?;

    let shape: Shape = shape![2, 2];
    let schema = TensorSchema::new(
        DType::F64,
        shape.clone(),
        Layout::Dense,
        shape![1, 2],
        contiguous_strides(&shape),
    )
    .expect("valid schema");

    let tensor = Tensor::<FsEntry, f64>::create(dir.clone(), schema.clone())
        .await
        .expect("create tensor");

    tensor
        .write_value(&[0, 0], std::f64::consts::PI)
        .await
        .expect("write pi");
    tensor
        .write_value(&[1, 1], std::f64::consts::E)
        .await
        .expect("write e");

    let pi = tensor.read_value(&[0, 0]).await.expect("read pi");
    let e = tensor.read_value(&[1, 1]).await.expect("read e");

    assert_eq!(pi, std::f64::consts::PI);
    assert_eq!(e, std::f64::consts::E);

    let loaded = Tensor::<FsEntry, f64>::load_with_schema(dir.clone(), &schema)
        .await
        .expect("load tensor");
    let loaded_e = loaded.read_value(&[1, 1]).await.expect("read loaded e");
    assert_eq!(loaded_e, std::f64::consts::E);

    dir.sync().await?;
    let _ = tokio::fs::remove_dir_all(&root).await;

    Ok(())
}