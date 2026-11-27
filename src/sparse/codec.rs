use destream::en::EncodeSeq;
use destream::{de, en};
use futures::stream;

use super::{SparseCell, SparseNode};
use crate::{MAX_BLOCK_CAPACITY, TensorElement};

// Allocation fences for decoded nodes, independent of native node packing.
const MAX_NODE_ENTRIES: usize = 1024;
const MAX_NODE_VALUES: usize = 4 * MAX_BLOCK_CAPACITY;

trait StorageValue: TensorElement {
    type Wire: de::FromStream<Context = ()> + for<'en> en::IntoStream<'en> + 'static;

    fn into_wire(self) -> Self::Wire;
    fn from_wire(wire: Self::Wire) -> Self;
}

macro_rules! real_values {
    ($($ty:ty),+) => {
        $(
            impl StorageValue for $ty {
                type Wire = Self;

                fn into_wire(self) -> Self {
                    self
                }

                fn from_wire(wire: Self) -> Self {
                    wire
                }
            }
        )+
    };
}

real_values!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cfg(feature = "complex")]
impl<F: StorageValue> StorageValue for crate::complex::Complex<F>
where
    Self: TensorElement,
{
    type Wire = (F::Wire, F::Wire);

    fn into_wire(self) -> Self::Wire {
        (self.re.into_wire(), self.im.into_wire())
    }

    fn from_wire((re, im): Self::Wire) -> Self {
        Self::new(F::from_wire(re), F::from_wire(im))
    }
}

struct Payload<'a, T>(&'a [T]);

impl<'en, T: StorageValue> en::IntoStream<'en> for Payload<'en, T> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_seq_stream(stream::iter(self.0.iter().copied().map(T::into_wire)))
    }
}

struct Rows<T>(Vec<Vec<SparseCell<T>>>);

impl<T> From<Rows<T>> for Vec<Vec<SparseCell<T>>> {
    fn from(rows: Rows<T>) -> Self {
        rows.0
    }
}

impl<T: StorageValue> de::FromStream for Rows<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let mut seq = decoder.open_container(de::Kind::Seq, None).await?;
        let mut rows = Vec::new();
        let mut remaining = MAX_NODE_VALUES;
        while seq.slot() != de::Slot::End {
            if rows.len() == MAX_NODE_ENTRIES {
                return Err(de::Error::custom("too many sparse rows"));
            }

            let mut fields = decoder.open_container(de::Kind::Seq, None).await?;
            if fields.slot() == de::Slot::End {
                return Err(de::Error::custom("missing sparse block ID"));
            }

            let key = SparseCell::Key(u64::from_stream((), decoder).await?);
            decoder.finish_child(&mut fields).await?;
            let row = if fields.slot() == de::Slot::End {
                vec![key]
            } else {
                let mut payload = decoder.open_container(de::Kind::Seq, None).await?;
                let mut values = Vec::new();
                let limit = remaining.min(MAX_BLOCK_CAPACITY);
                while payload.slot() != de::Slot::End {
                    if values.len() == limit {
                        return Err(de::Error::custom("too many sparse payload values"));
                    }

                    let value = T::Wire::from_stream((), decoder).await?;
                    decoder.finish_child(&mut payload).await?;
                    values.try_reserve(1).map_err(de::Error::custom)?;
                    values.push(T::from_wire(value));
                }

                if values.is_empty() {
                    return Err(de::Error::custom("empty sparse block"));
                }

                decoder.finish_child(&mut fields).await?;
                if fields.slot() != de::Slot::End {
                    return Err(de::Error::custom("unexpected sparse row field"));
                }

                remaining -= values.len();
                vec![key, SparseCell::Payload(values)]
            };

            decoder.finish_child(&mut seq).await?;
            rows.try_reserve(1).map_err(de::Error::custom)?;
            rows.push(row);
        }

        Ok(Self(rows))
    }
}

impl<T: StorageValue> de::FromStream for SparseNode<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let node =
            b_tree::Node::decode_bounded::<Rows<T>, _>((), decoder, MAX_NODE_ENTRIES).await?;

        if matches!(&node, b_tree::Node::Leaf(rows) if rows.iter().any(|row| row.len() != 2)) {
            return Err(de::Error::custom("incomplete sparse leaf row"));
        }

        Ok(Self(node))
    }
}

impl<'en, T: StorageValue> en::ToStream<'en> for SparseNode<T> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        struct BorrowedRows<'a, T> {
            leaf: bool,
            rows: &'a [Vec<SparseCell<T>>],
        }

        impl<'en, T: StorageValue> en::IntoStream<'en> for BorrowedRows<'en, T> {
            fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
                if self.rows.len() > MAX_NODE_ENTRIES {
                    return Err(en::Error::custom("too many sparse rows"));
                }

                let mut seq = encoder.encode_seq(Some(self.rows.len()))?;
                let mut remaining = MAX_NODE_VALUES;
                for row in self.rows {
                    match row.as_slice() {
                        [SparseCell::Key(block)] if !self.leaf => seq.encode_element((*block,))?,
                        [SparseCell::Key(block), SparseCell::Payload(values)] => {
                            if values.is_empty() || values.len() > remaining.min(MAX_BLOCK_CAPACITY)
                            {
                                return Err(en::Error::custom("invalid sparse payload length"));
                            }

                            remaining -= values.len();
                            seq.encode_element((*block, Payload(values)))?;
                        }
                        _ => return Err(en::Error::custom("invalid sparse row")),
                    }
                }

                seq.end()
            }
        }

        self.0
            .encode_bounded(encoder, MAX_NODE_ENTRIES, |rows| BorrowedRows {
                leaf: self.0.is_leaf(),
                rows,
            })
    }
}
