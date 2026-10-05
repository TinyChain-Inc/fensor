//! Adapter-owned, bounded typed sparse payload serialization.
use destream::{de, en};
use fensor::{SparseCell, SparseNode, TensorElement};

pub(super) const MAX_NODE_ENTRIES: usize = 1024;
pub(super) const MAX_NODE_VALUES: usize = 4 * fensor::MAX_BLOCK_CAPACITY;

pub trait Bits: TensorElement {
    const COMPONENT_BYTES: usize = std::mem::size_of::<Self>();

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
    const COMPONENT_BYTES: usize = 4;

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
    const COMPONENT_BYTES: usize = 8;

    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }

    fn from_bits(a: u64, b: u64) -> Option<Self> {
        Some(Self::new(f64::from_bits(a), f64::from_bits(b)))
    }
}

struct EncodedPayload<'a, T>(&'a [T]);

impl<'en, T: Bits> en::IntoStream<'en> for EncodedPayload<'en, T> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if self.0.is_empty() || self.0.len() > fensor::MAX_BLOCK_CAPACITY {
            return Err(en::Error::custom("invalid sparse payload length"));
        }
        let width = std::mem::size_of::<T>();
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(self.0));
        for value in self.0 {
            let (a, b) = value.bits();
            bytes.extend_from_slice(&a.to_le_bytes()[..T::COMPONENT_BYTES]);
            if T::COMPONENT_BYTES != width {
                bytes.extend_from_slice(&b.to_le_bytes()[..T::COMPONENT_BYTES]);
            }
        }
        encoder.encode_array_u8(futures::stream::iter([bytes]))
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
            let [SparseCell::Key(block), SparseCell::Payload(values)] = row.as_slice() else {
                return Err(en::Error::custom("invalid sparse payload row"));
            };
            encoded.push((*block, EncodedPayload(values)));
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

struct Payload<T>(Vec<T>);

impl<T: Bits> de::FromStream for Payload<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);

        impl<T: Bits> de::Visitor for Visitor<T> {
            type Value = Payload<T>;

            fn expecting() -> &'static str {
                "bounded dense sparse-row payload"
            }

            async fn visit_array_u8<A: de::ArrayAccess<u8>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                let width = std::mem::size_of::<T>();
                let mut values = Vec::new();
                let mut scratch = [0u8; 16];
                let mut filled = 0;
                loop {
                    let read = array.buffer(&mut scratch[filled..width]).await?;
                    if read == 0 {
                        if filled != 0 {
                            return Err(de::Error::custom("incomplete sparse scalar bytes"));
                        }
                        break;
                    }
                    filled += read;
                    if filled != width {
                        continue;
                    }
                    if values.len() == fensor::MAX_BLOCK_CAPACITY {
                        return Err(de::Error::custom("oversized sparse payload"));
                    }
                    let mut a = [0; 8];
                    let mut b = [0; 8];
                    a[..T::COMPONENT_BYTES].copy_from_slice(&scratch[..T::COMPONENT_BYTES]);
                    if T::COMPONENT_BYTES != width {
                        b[..T::COMPONENT_BYTES]
                            .copy_from_slice(&scratch[T::COMPONENT_BYTES..width]);
                    }
                    let value = T::from_bits(u64::from_le_bytes(a), u64::from_le_bytes(b))
                        .ok_or_else(|| de::Error::custom("invalid sparse scalar bits"))?;
                    values.try_reserve(1).map_err(de::Error::custom)?;
                    values.push(value);
                    filled = 0;
                }
                if values.is_empty() {
                    return Err(de::Error::custom("empty sparse payload"));
                }
                Ok(Payload(values))
            }
        }

        decoder
            .decode_array_u8(Visitor(std::marker::PhantomData))
            .await
    }
}

type EncodedRow<T> = (u64, Payload<T>);

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
                let mut count = 0;
                while let Some((block, Payload(values))) =
                    seq.next_element::<EncodedRow<T>>(()).await?
                {
                    if rows.len() == MAX_NODE_ENTRIES {
                        return Err(de::Error::custom("oversized sparse node"));
                    }
                    count += values.len();
                    if count > MAX_NODE_VALUES {
                        return Err(de::Error::custom("oversized sparse node payloads"));
                    }

                    rows.try_reserve(1).map_err(de::Error::custom)?;
                    rows.push(vec![SparseCell::Key(block), SparseCell::Payload(values)]);
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
