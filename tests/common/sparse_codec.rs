//! Adapter-owned, bounded typed sparse payload serialization.
use destream::{de, en};
use fensor::{SparseCell, SparseNode, TensorElement};

const MAX_NODE_ENTRIES: usize = 1024;

pub trait Bits: TensorElement {
    fn bits(self) -> (u64, u64);

    fn from_bits(a: u64, b: u64) -> Option<Self>;
}
macro_rules! integer {
    ($($t:ty => $u:ty),+) => {
        $(
            impl Bits for $t {
                fn bits(self) -> (u64, u64) {
                    (self as $u as u64, 0)
                }

                fn from_bits(a: u64, b: u64) -> Option<Self> {
                    (b == 0 && a <= <$u>::MAX as u64).then_some(a as $u as Self)
                }
            }
        )+
    };
}

integer!(
    u8 => u8, u16 => u16, u32 => u32, u64 => u64,
    i8 => u8, i16 => u16, i32 => u32, i64 => u64
);

impl Bits for f32 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits() as u64, 0)
    }

    fn from_bits(a: u64, b: u64) -> Option<Self> {
        (b == 0 && a <= u32::MAX as u64).then_some(f32::from_bits(a as u32))
    }
}

impl Bits for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }

    fn from_bits(a: u64, b: u64) -> Option<Self> {
        (b == 0).then_some(f64::from_bits(a))
    }
}

#[cfg(feature = "complex")]
impl Bits for fensor::complex::Complex32 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits() as u64, self.im.to_bits() as u64)
    }

    fn from_bits(a: u64, b: u64) -> Option<Self> {
        (a <= u32::MAX as u64 && b <= u32::MAX as u64).then_some(Self::new(
            f32::from_bits(a as u32),
            f32::from_bits(b as u32),
        ))
    }
}

#[cfg(feature = "complex")]
impl Bits for fensor::complex::Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }

    fn from_bits(a: u64, b: u64) -> Option<Self> {
        Some(Self::new(f64::from_bits(a), f64::from_bits(b)))
    }
}

pub struct Encoded<'a, T>(pub &'a SparseNode<T>);

impl<'en, T: Bits> en::IntoStream<'en> for Encoded<'en, T> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let (leaf, rows, children) = match self.0 {
            b_table::Node::Leaf(rows) => (true, rows, &[][..]),
            b_table::Node::Index(rows, children) => (false, rows, children.as_slice()),
        };
        let mut encoded = Vec::with_capacity(rows.len());
        for row in rows {
            let [
                SparseCell::Key(block),
                SparseCell::Key(offset),
                SparseCell::Value(value),
            ] = row.as_slice()
            else {
                return Err(en::Error::custom("invalid sparse payload row"));
            };
            encoded.push((*block, *offset, value.bits()));
        }
        (leaf, encoded, children).into_stream(encoder)
    }
}

struct Limited<T>(Vec<T>);

impl<T: de::FromStream<Context = ()> + Send> de::FromStream for Limited<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);

        impl<T: de::FromStream<Context = ()> + Send> de::Visitor for Visitor<T> {
            type Value = Limited<T>;

            fn expecting() -> &'static str {
                "bounded sparse node sequence"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(v) = seq.next_element(()).await? {
                    if values.len() == MAX_NODE_ENTRIES {
                        return Err(de::Error::custom("oversized sparse node"));
                    }
                    values.push(v);
                }
                Ok(Limited(values))
            }
        }
        decoder.decode_seq(Visitor(std::marker::PhantomData)).await
    }
}
type EncodedRow = (u64, u64, (u64, u64));

struct Rows<T>(Vec<Vec<SparseCell<T>>>);

impl<T: Bits> de::FromStream for Rows<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);

        impl<T: Bits> de::Visitor for Visitor<T> {
            type Value = Rows<T>;

            fn expecting() -> &'static str {
                "bounded sparse node rows"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut rows = Vec::new();
                while let Some((block, offset, (a, b))) = seq.next_element::<EncodedRow>(()).await?
                {
                    if rows.len() == MAX_NODE_ENTRIES {
                        return Err(de::Error::custom("oversized sparse node"));
                    }
                    let value = T::from_bits(a, b)
                        .ok_or_else(|| de::Error::custom("invalid sparse scalar bits"))?;
                    rows.push(vec![
                        SparseCell::Key(block),
                        SparseCell::Key(offset),
                        SparseCell::Value(value),
                    ]);
                }
                Ok(Rows(rows))
            }
        }
        decoder.decode_seq(Visitor(std::marker::PhantomData)).await
    }
}

pub struct Decoded<T>(pub SparseNode<T>);

impl<T: Bits> de::FromStream for Decoded<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (leaf, rows, children) =
            <(bool, Rows<T>, Limited<_>) as de::FromStream>::from_stream((), decoder).await?;
        if leaf {
            if !children.0.is_empty() {
                return Err(de::Error::custom("leaf has children"));
            }
            Ok(Self(b_table::Node::Leaf(rows.0)))
        } else {
            Ok(Self(b_table::Node::Index(rows.0, children.0)))
        }
    }
}
