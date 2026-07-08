use destream::{de, en};

use crate::wire_tags::{LAYOUT_TAG_DENSE, LAYOUT_TAG_SPARSE};
use crate::{
    DType, Layout, Tensor, TensorArray, TensorElement, TensorFileEntry, TensorSchema,
    TensorViewSemantics, contiguous_strides,
};

fn encode_sparse_axis<E: en::Error>(axis: Option<usize>) -> Result<Option<u64>, E> {
    axis.map(|axis| u64::try_from(axis).map_err(|_| E::custom("sparse axis overflow")))
        .transpose()
}

fn encode_layout<E: en::Error>(layout: Layout) -> Result<(u8, Option<u64>), E> {
    match layout {
        Layout::Dense => Ok((LAYOUT_TAG_DENSE, None)),
        Layout::Sparse { axis } => Ok((LAYOUT_TAG_SPARSE, encode_sparse_axis(axis)?)),
    }
}

fn encode_tensor_schema_ref(schema: &TensorSchema) -> Result<(DType, Vec<u64>, Layout), String> {
    let shape = schema
        .shape_u64()
        .map_err(|err| err.to_string())?
        .into_iter()
        .collect::<Vec<u64>>();
    Ok((schema.dtype(), shape, schema.layout().clone()))
}

fn encode_tensor_schema(schema: TensorSchema) -> Result<(DType, Vec<u64>, Layout), String> {
    encode_tensor_schema_ref(&schema)
}

fn encode_tensor_ref<FE, T>(tensor: &Tensor<FE, T>) -> Result<TensorSchema, String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    if !tensor.is_base_tensor() {
        return Err(
            "cannot serialize a tensor with a non-identity view; views are metadata-only and \
            are never persisted"
                .to_string(),
        );
    }

    Ok(tensor.schema().clone())
}

fn encode_tensor<FE, T>(tensor: Tensor<FE, T>) -> Result<TensorSchema, String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    encode_tensor_ref(&tensor)
}

impl de::FromStream for DType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let dtype = String::from_stream((), decoder).await?;
        DType::try_parse(&dtype)
            .ok_or_else(|| de::Error::custom(format!("unsupported dtype {dtype}")))
    }
}

impl<'en> en::ToStream<'en> for DType {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_str(self.as_str())
    }
}

impl<'en> en::IntoStream<'en> for DType {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_str(self.as_str())
    }
}

impl de::FromStream for Layout {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, axis): (u8, Option<u64>) = <(u8, Option<u64>)>::from_stream((), decoder).await?;

        match tag {
            LAYOUT_TAG_DENSE => {
                if axis.is_some() {
                    return Err(de::Error::custom(
                        "dense layout must not include a sparse axis",
                    ));
                }

                Ok(Self::Dense)
            }
            LAYOUT_TAG_SPARSE => {
                let axis = axis
                    .map(|axis| {
                        usize::try_from(axis).map_err(|_| de::Error::custom("sparse axis overflow"))
                    })
                    .transpose()?;

                Ok(Self::Sparse { axis })
            }
            _ => Err(de::Error::custom(format!(
                "unknown tensor layout tag {tag}"
            ))),
        }
    }
}

impl<'en> en::ToStream<'en> for Layout {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_layout(self.clone())?, encoder)
    }
}

impl<'en> en::IntoStream<'en> for Layout {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_layout(self)?, encoder)
    }
}

impl de::FromStream for TensorSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (dtype, shape, layout): (DType, Vec<u64>, Layout) =
            <(DType, Vec<u64>, Layout)>::from_stream((), decoder).await?;
        let shape = TensorSchema::shape_usize_from_u64(&shape).map_err(de::Error::custom)?;
        let block_shape = super::default_block_shape(&shape);
        let strides = contiguous_strides(&shape);

        TensorSchema::new(dtype, shape, layout, block_shape, strides).map_err(de::Error::custom)
    }
}

impl<'en> en::ToStream<'en> for TensorSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            encode_tensor_schema_ref(self).map_err(en::Error::custom)?,
            encoder,
        )
    }
}

impl<'en> en::IntoStream<'en> for TensorSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            encode_tensor_schema(self).map_err(en::Error::custom)?,
            encoder,
        )
    }
}

impl<FE, T> de::FromStream for Tensor<FE, T>
where
    FE: TensorFileEntry<T> + safecast::AsType<String> + From<String>,
    T: TensorElement,
{
    type Context = freqfs::DirLock<FE>;

    async fn from_stream<D: de::Decoder>(
        dir: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let schema = TensorSchema::from_stream((), decoder).await?;

        Tensor::create(dir, schema).await.map_err(de::Error::custom)
    }
}

impl<'en, FE, T> en::ToStream<'en> for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_tensor_ref(self).map_err(en::Error::custom)?, encoder)
    }
}

impl<'en, FE, T> en::IntoStream<'en> for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_tensor(self).map_err(en::Error::custom)?, encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_stream<T>()
    where
        T: de::FromStream<Context = ()>,
        for<'en> T: en::ToStream<'en> + en::IntoStream<'en>,
    {
    }

    #[test]
    fn schema_types_have_destream_impls() {
        assert_stream::<DType>();
        assert_stream::<Layout>();
        assert_stream::<TensorSchema>();
    }
}
