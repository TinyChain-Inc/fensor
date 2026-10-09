use std::marker::PhantomData;

use destream::{de, en};
use get_size::GetSize;

use crate::schema::{StorageSchema, TensorSchema};
use crate::{Error, Layout, Result, Shape, TensorElement};

/// Typed filesystem metadata, independent of the byte codec used by the file adapter.
///
/// The adapter must preserve the element type when saving and loading entries, for
/// example with distinct variants for `TensorMetadata<f32>` and `TensorMetadata<u8>`.
/// The destream representation contains geometry; it does not encode `T`.
/// Decoding requires the adapter's maximum metadata rank as its `usize` context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMetadata<T> {
    shape: Shape,
    layout: Layout,
    block_shape: Shape,
    dtype: PhantomData<T>,
}

impl<T: TensorElement> TensorMetadata<T> {
    pub fn new(shape: Shape, layout: Layout, block_shape: Shape) -> Result<Self> {
        TensorSchema::new(T::dtype(), shape.clone())?;
        if shape.len() != block_shape.len() {
            return Err(Error::InvalidSchema(
                "block shape rank differs from tensor rank".into(),
            ));
        }

        let capacity = crate::schema::checked_product(&block_shape)?;
        if capacity == 0 || capacity > crate::schema::MAX_BLOCK_CAPACITY as u64 {
            return Err(Error::InvalidSchema("invalid block capacity".into()));
        }
        StorageSchema::from_block_shape(&shape, layout, block_shape.clone())?;
        Ok(Self {
            shape,
            layout,
            block_shape,
            dtype: PhantomData,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn block_shape(&self) -> &Shape {
        &self.block_shape
    }

    pub(crate) fn schemas(&self) -> Result<(TensorSchema, StorageSchema)> {
        Ok((
            TensorSchema::new(T::dtype(), self.shape.clone())?,
            StorageSchema::from_block_shape(&self.shape, self.layout, self.block_shape.clone())?,
        ))
    }
}

impl<T> GetSize for TensorMetadata<T> {
    fn get_heap_size(&self) -> usize {
        [&self.shape, &self.block_shape]
            .into_iter()
            .filter(|shape| shape.spilled())
            .map(|shape| shape.capacity() * std::mem::size_of::<u64>())
            .sum()
    }
}

// Dimension decoding is bounded by the adapter, independently of storage capacity.
async fn decode_shape<D: de::Decoder>(
    limit: usize,
    decoder: &mut D,
) -> std::result::Result<Shape, D::Error> {
    let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
    let mut shape = Shape::new();
    while sequence.slot() != de::Slot::End {
        let dimension = <u64 as de::FromStream>::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;
        if shape.len() == limit {
            return Err(de::Error::custom(
                "tensor metadata exceeds the adapter rank limit",
            ));
        }
        shape.push(dimension);
    }
    Ok(shape)
}

fn require_field<E: de::Error>(sequence: &de::Container) -> std::result::Result<(), E> {
    if sequence.slot() == de::Slot::End {
        Err(E::custom("missing tensor metadata field"))
    } else {
        Ok(())
    }
}

impl<T: TensorElement> de::FromStream for TensorMetadata<T> {
    type Context = usize;

    async fn from_stream<D: de::Decoder>(
        limit: usize,
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        let mut sequence = decoder.open_container(de::Kind::Seq, None).await?;
        require_field::<D::Error>(&sequence)?;
        let shape = decode_shape(limit, decoder).await?;
        decoder.finish_child(&mut sequence).await?;

        require_field::<D::Error>(&sequence)?;
        let sparse = bool::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;

        require_field::<D::Error>(&sequence)?;
        let axis = Option::<u64>::from_stream((), decoder).await?;
        decoder.finish_child(&mut sequence).await?;

        require_field::<D::Error>(&sequence)?;
        let block_shape = decode_shape(limit, decoder).await?;
        decoder.finish_child(&mut sequence).await?;

        let layout = if sparse {
            Layout::Sparse {
                axis: axis
                    .map(usize::try_from)
                    .transpose()
                    .map_err(de::Error::custom)?,
            }
        } else if axis.is_none() {
            Layout::Dense
        } else {
            return Err(de::Error::custom("dense metadata contains a sparse axis"));
        };

        if sequence.slot() != de::Slot::End {
            return Err(de::Error::custom("unexpected metadata field"));
        }
        TensorMetadata::new(shape, layout, block_shape).map_err(de::Error::custom)
    }
}

impl<'en, T: TensorElement> en::ToStream<'en> for TensorMetadata<T> {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        let (sparse, axis) = match self.layout {
            Layout::Dense => (false, None),
            Layout::Sparse { axis } => (
                true,
                axis.map(u64::try_from)
                    .transpose()
                    .map_err(en::Error::custom)?,
            ),
        };
        en::IntoStream::into_stream(
            (
                self.shape.as_slice(),
                sparse,
                axis,
                self.block_shape.as_slice(),
            ),
            encoder,
        )
    }
}
