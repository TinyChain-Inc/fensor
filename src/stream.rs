use destream::{de, en};

use crate::wire_tags::{
    AXIS_MAP_TAG_AFFINE, AXIS_MAP_TAG_GATHER, AXIS_MAP_TAG_IDENTITY, LAYOUT_TAG_DENSE,
    LAYOUT_TAG_SPARSE,
};
use crate::{
    DType, Layout, Tensor, TensorElement, TensorFileEntry, TensorSchema, ViewAxisMapSchema,
    ViewAxisSchema, ViewSchema, contiguous_strides,
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

fn encode_axis_map_ref(axis_map: &ViewAxisMapSchema) -> (u8, Vec<u64>) {
    match axis_map {
        ViewAxisMapSchema::Identity => (AXIS_MAP_TAG_IDENTITY, Vec::new()),
        ViewAxisMapSchema::Affine { start, step } => (AXIS_MAP_TAG_AFFINE, vec![*start, *step]),
        ViewAxisMapSchema::Gather(indices) => (
            AXIS_MAP_TAG_GATHER,
            indices.iter().copied().collect::<Vec<u64>>(),
        ),
    }
}

fn encode_axis_map(axis_map: ViewAxisMapSchema) -> (u8, Vec<u64>) {
    match axis_map {
        ViewAxisMapSchema::Identity => (AXIS_MAP_TAG_IDENTITY, Vec::new()),
        ViewAxisMapSchema::Affine { start, step } => (AXIS_MAP_TAG_AFFINE, vec![start, step]),
        ViewAxisMapSchema::Gather(indices) => (
            AXIS_MAP_TAG_GATHER,
            indices.into_iter().collect::<Vec<u64>>(),
        ),
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

fn encode_view_axis_ref(schema: &ViewAxisSchema) -> Result<(u64, ViewAxisMapSchema), String> {
    let base_axis =
        u64::try_from(schema.base_axis).map_err(|_| "view axis overflow".to_string())?;
    Ok((base_axis, schema.map.clone()))
}

fn encode_view_axis(schema: ViewAxisSchema) -> Result<(u64, ViewAxisMapSchema), String> {
    let base_axis =
        u64::try_from(schema.base_axis).map_err(|_| "view axis overflow".to_string())?;
    Ok((base_axis, schema.map))
}

fn encode_view_schema_ref(
    schema: &ViewSchema,
) -> Result<(u64, Vec<ViewAxisSchema>, Vec<Option<u64>>), String> {
    let axes = schema.axes.iter().cloned().collect::<Vec<ViewAxisSchema>>();
    let base_fixed = schema
        .base_fixed
        .iter()
        .cloned()
        .collect::<Vec<Option<u64>>>();
    let base_rank =
        u64::try_from(schema.base_rank).map_err(|_| "base rank overflow".to_string())?;

    Ok((base_rank, axes, base_fixed))
}

fn encode_view_schema(
    schema: ViewSchema,
) -> Result<(u64, Vec<ViewAxisSchema>, Vec<Option<u64>>), String> {
    let axes = schema.axes.into_iter().collect::<Vec<ViewAxisSchema>>();
    let base_fixed = schema.base_fixed.into_iter().collect::<Vec<Option<u64>>>();
    let base_rank =
        u64::try_from(schema.base_rank).map_err(|_| "base rank overflow".to_string())?;

    Ok((base_rank, axes, base_fixed))
}

fn encode_tensor_ref<FE, T>(tensor: &Tensor<FE, T>) -> Result<(TensorSchema, ViewSchema), String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    let schema = tensor.schema.clone();
    let view = tensor.view_schema().map_err(|err| err.to_string())?;

    Ok((schema, view))
}

fn encode_tensor<FE, T>(tensor: Tensor<FE, T>) -> Result<(TensorSchema, ViewSchema), String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    let view = tensor.view_schema().map_err(|err| err.to_string())?;
    let schema = tensor.schema;

    Ok((schema, view))
}

impl de::FromStream for DType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let dtype = String::from_stream((), decoder).await?;
        DType::from_str(dtype.as_str())
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

impl de::FromStream for ViewAxisMapSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, data): (u8, Vec<u64>) = <(u8, Vec<u64>)>::from_stream((), decoder).await?;

        match tag {
            AXIS_MAP_TAG_IDENTITY => {
                if !data.is_empty() {
                    return Err(de::Error::custom("identity axis map expects no payload"));
                }
                Ok(Self::Identity)
            }
            AXIS_MAP_TAG_AFFINE => {
                if data.len() != 2 {
                    return Err(de::Error::custom("affine axis map expects [start, step]"));
                }
                Ok(Self::Affine {
                    start: data[0],
                    step: data[1],
                })
            }
            AXIS_MAP_TAG_GATHER => Ok(Self::Gather(data.into())),
            _ => Err(de::Error::custom(format!("unknown axis map tag {tag}"))),
        }
    }
}

impl<'en> en::ToStream<'en> for ViewAxisMapSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_axis_map_ref(self), encoder)
    }
}

impl<'en> en::IntoStream<'en> for ViewAxisMapSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_axis_map(self), encoder)
    }
}

impl de::FromStream for ViewAxisSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (base_axis, map): (u64, ViewAxisMapSchema) =
            <(u64, ViewAxisMapSchema)>::from_stream((), decoder).await?;

        let base_axis =
            usize::try_from(base_axis).map_err(|_| de::Error::custom("view axis overflow"))?;

        Ok(Self { base_axis, map })
    }
}

impl<'en> en::ToStream<'en> for ViewAxisSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            encode_view_axis_ref(self).map_err(en::Error::custom)?,
            encoder,
        )
    }
}

impl<'en> en::IntoStream<'en> for ViewAxisSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_view_axis(self).map_err(en::Error::custom)?, encoder)
    }
}

impl de::FromStream for ViewSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (base_rank, axes, base_fixed): (u64, Vec<ViewAxisSchema>, Vec<Option<u64>>) =
            <(u64, Vec<ViewAxisSchema>, Vec<Option<u64>>)>::from_stream((), decoder).await?;

        let base_rank =
            usize::try_from(base_rank).map_err(|_| de::Error::custom("base rank overflow"))?;

        Ok(Self {
            base_rank,
            axes: axes.into(),
            base_fixed: base_fixed.into(),
        })
    }
}

impl<'en> en::ToStream<'en> for ViewSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            encode_view_schema_ref(self).map_err(en::Error::custom)?,
            encoder,
        )
    }
}

impl<'en> en::IntoStream<'en> for ViewSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            encode_view_schema(self).map_err(en::Error::custom)?,
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
        let (schema, view): (TensorSchema, ViewSchema) =
            <(TensorSchema, ViewSchema)>::from_stream((), decoder).await?;

        let tensor = Tensor::create(dir, schema)
            .await
            .map_err(de::Error::custom)?;

        tensor.with_view_schema(&view).map_err(de::Error::custom)
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
        assert_stream::<ViewAxisMapSchema>();
        assert_stream::<ViewAxisSchema>();
        assert_stream::<ViewSchema>();
    }
}
