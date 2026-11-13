use std::marker::PhantomData;

use destream::{de, en};
use get_size::GetSize;

use crate::schema::{StorageSchema, TensorSchema};
use crate::{Layout, Result, Shape, TensorElement};

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
    block_capacity: usize,
    dtype: PhantomData<T>,
}

impl<T: TensorElement> TensorMetadata<T> {
    pub fn new(shape: Shape, layout: Layout, block_capacity: usize) -> Result<Self> {
        TensorSchema::new(T::dtype(), shape.clone())?;

        let storage = StorageSchema::new(&shape, layout, block_capacity)?;
        let block_capacity = storage.blocks.capacity;
        Ok(Self {
            shape,
            layout,
            block_capacity,
            dtype: PhantomData,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn block_capacity(&self) -> usize {
        self.block_capacity
    }

    pub(crate) fn schemas(&self) -> Result<(TensorSchema, StorageSchema)> {
        Ok((
            TensorSchema::new(T::dtype(), self.shape.clone())?,
            StorageSchema::new(&self.shape, self.layout, self.block_capacity)?,
        ))
    }
}

impl<T> GetSize for TensorMetadata<T> {
    fn get_heap_size(&self) -> usize {
        if self.shape.spilled() {
            self.shape.capacity() * std::mem::size_of::<u64>()
        } else {
            0
        }
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
        let block_capacity =
            usize::try_from(u64::from_stream((), decoder).await?).map_err(de::Error::custom)?;

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

        TensorMetadata::new(shape, layout, block_capacity).map_err(de::Error::custom)
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
                self.block_capacity as u64,
            ),
            encoder,
        )
    }
}
