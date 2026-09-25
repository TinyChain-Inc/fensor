use std::marker::PhantomData;

use destream::{de, en};

use crate::schema::{StorageSchema, TensorSchema};
use crate::{Error, Layout, Result, Shape, TensorElement};

/// Typed filesystem metadata, independent of the byte codec used by the file adapter.
///
/// The adapter must preserve the element type when saving and loading entries, for
/// example with distinct variants for `TensorMetadata<f32>` and `TensorMetadata<u8>`.
/// The destream representation contains geometry only; it does not encode `T`.
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

    pub(crate) fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + [&self.shape, &self.block_shape]
                .into_iter()
                .filter(|s| s.spilled())
                .map(|s| s.capacity() * std::mem::size_of::<u64>())
                .sum::<usize>()
    }
}

impl<T: TensorElement> de::FromStream for TensorMetadata<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _: (),
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        let (shape, sparse, axis, block_shape): (Vec<u64>, bool, Option<u64>, Vec<u64>) =
            <(Vec<u64>, bool, Option<u64>, Vec<u64>) as de::FromStream>::from_stream((), decoder)
                .await?;
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
        Self::new(shape.into(), layout, block_shape.into()).map_err(de::Error::custom)
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
