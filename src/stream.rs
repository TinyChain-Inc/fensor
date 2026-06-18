use destream::{de, en};

use crate::{
    DType, Layout, Tensor, TensorElement, TensorFileEntry, TensorSchema, ViewAxisMapSchema,
    ViewAxisSchema, ViewSchema, contiguous_strides,
};

impl de::FromStream for DType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let dtype = String::from_stream((), decoder).await?;
        match dtype.as_str() {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(de::Error::custom(format!("unsupported dtype {dtype}"))),
        }
    }
}

impl<'en> en::ToStream<'en> for DType {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let tag = match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        };

        encoder.encode_str(tag)
    }
}

impl<'en> en::IntoStream<'en> for DType {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let tag = match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        };

        encoder.encode_str(tag)
    }
}

impl de::FromStream for Layout {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, axis): (u8, Option<u64>) = <(u8, Option<u64>)>::from_stream((), decoder).await?;

        match tag {
            0 => {
                if axis.is_some() {
                    return Err(de::Error::custom("dense layout must not include a sparse axis"));
                }

                Ok(Self::Dense)
            }
            1 => {
                let axis = axis
                    .map(|axis| {
                        usize::try_from(axis)
                            .map_err(|_| de::Error::custom("sparse axis overflow"))
                    })
                    .transpose()?;

                Ok(Self::Sparse { axis })
            }
            _ => Err(de::Error::custom(format!("unknown tensor layout tag {tag}"))),
        }
    }
}

impl<'en> en::ToStream<'en> for Layout {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let encoded = match self {
            Self::Dense => (0u8, None),
            Self::Sparse { axis } => {
                let axis = axis
                    .map(|axis| {
                        u64::try_from(axis)
                            .map_err(|_| en::Error::custom("sparse axis overflow"))
                    })
                    .transpose()?;

                (1u8, axis)
            }
        };

        en::IntoStream::into_stream(encoded, encoder)
    }
}

impl<'en> en::IntoStream<'en> for Layout {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let encoded = match self {
            Self::Dense => (0u8, None),
            Self::Sparse { axis } => {
                let axis = axis
                    .map(|axis| {
                        u64::try_from(axis)
                            .map_err(|_| en::Error::custom("sparse axis overflow"))
                    })
                    .transpose()?;

                (1u8, axis)
            }
        };

        en::IntoStream::into_stream(encoded, encoder)
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

        TensorSchema::new(dtype, shape, layout, block_shape, strides)
            .map_err(de::Error::custom)
    }
}

impl<'en> en::ToStream<'en> for TensorSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let shape = self
            .shape_u64()
            .map_err(en::Error::custom)?
            .into_iter()
            .collect::<Vec<u64>>();
        en::IntoStream::into_stream((self.dtype(), shape, self.layout().clone()), encoder)
    }
}

impl<'en> en::IntoStream<'en> for TensorSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let dtype = self.dtype();
        let shape = self
            .shape_u64()
            .map_err(en::Error::custom)?
            .into_iter()
            .collect::<Vec<u64>>();
        en::IntoStream::into_stream((dtype, shape, self.layout().clone()), encoder)
    }
}

impl de::FromStream for ViewAxisMapSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, data): (u8, Vec<u64>) = <(u8, Vec<u64>)>::from_stream((), decoder).await?;

        match tag {
            0 => {
                if !data.is_empty() {
                    return Err(de::Error::custom("identity axis map expects no payload"));
                }
                Ok(Self::Identity)
            }
            1 => {
                if data.len() != 2 {
                    return Err(de::Error::custom("affine axis map expects [start, step]"));
                }
                Ok(Self::Affine {
                    start: data[0],
                    step: data[1],
                })
            }
            2 => Ok(Self::Gather(data.into())),
            _ => Err(de::Error::custom(format!("unknown axis map tag {tag}"))),
        }
    }
}

impl<'en> en::ToStream<'en> for ViewAxisMapSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Identity => en::IntoStream::into_stream((0u8, Vec::<u64>::new()), encoder),
            Self::Affine { start, step } => {
                en::IntoStream::into_stream((1u8, vec![*start, *step]), encoder)
            }
            Self::Gather(indices) => en::IntoStream::into_stream(
                (2u8, indices.iter().copied().collect::<Vec<u64>>()),
                encoder,
            ),
        }
    }
}

impl<'en> en::IntoStream<'en> for ViewAxisMapSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Identity => en::IntoStream::into_stream((0u8, Vec::<u64>::new()), encoder),
            Self::Affine { start, step } => {
                en::IntoStream::into_stream((1u8, vec![start, step]), encoder)
            }
            Self::Gather(indices) => {
                en::IntoStream::into_stream((2u8, indices.into_iter().collect::<Vec<u64>>()), encoder)
            }
        }
    }
}

impl de::FromStream for ViewAxisSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (base_axis, map): (u64, ViewAxisMapSchema) =
            <(u64, ViewAxisMapSchema)>::from_stream((), decoder).await?;

        Ok(Self { base_axis, map })
    }
}

impl<'en> en::ToStream<'en> for ViewAxisSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.base_axis, self.map.clone()), encoder)
    }
}

impl<'en> en::IntoStream<'en> for ViewAxisSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.base_axis, self.map), encoder)
    }
}

impl de::FromStream for ViewSchema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (base_rank, axes, base_fixed): (u64, Vec<ViewAxisSchema>, Vec<Option<u64>>) =
            <(u64, Vec<ViewAxisSchema>, Vec<Option<u64>>)>::from_stream((), decoder).await?;

        Ok(Self {
            base_rank,
            axes: axes.into(),
            base_fixed: base_fixed.into(),
        })
    }
}

impl<'en> en::ToStream<'en> for ViewSchema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let axes = self.axes.iter().cloned().collect::<Vec<ViewAxisSchema>>();
        let base_fixed = self.base_fixed.iter().cloned().collect::<Vec<Option<u64>>>();

        en::IntoStream::into_stream((self.base_rank, axes, base_fixed), encoder)
    }
}

impl<'en> en::IntoStream<'en> for ViewSchema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let axes = self.axes.into_iter().collect::<Vec<ViewAxisSchema>>();
        let base_fixed = self.base_fixed.into_iter().collect::<Vec<Option<u64>>>();

        en::IntoStream::into_stream((self.base_rank, axes, base_fixed), encoder)
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
        let schema = self.schema.clone();
        let view = self.view_schema().map_err(en::Error::custom)?;

        en::IntoStream::into_stream((schema, view), encoder)
    }
}

impl<'en, FE, T> en::IntoStream<'en> for Tensor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let view = self.view_schema().map_err(en::Error::custom)?;
        let schema = self.schema;

        en::IntoStream::into_stream((schema, view), encoder)
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
