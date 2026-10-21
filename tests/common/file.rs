//! Test/benchmark filesystem adapter, not fensor's production storage format.
//!
//! Each entry encodes as `(tag, payload)`. Tags distinguish dense blocks,
//! metadata, and sparse nodes by dtype; the encoder and decoder share the same
//! constants. Their existing numeric values are preserved. Complex blocks encode
//! component pairs in a temporary vector bounded by the stored block capacity.

use destream::{de, en};
use futures::TryStreamExt;
use get_size::GetSize;
use safecast::as_type;
use tokio::io::AsyncWriteExt;

use super::{counters, sparse_codec};

// Private adapter tags, grouped by dtype. Gaps have no decoding meaning.
mod tag {
    pub(super) const BLOCK_U8: u8 = 3;
    pub(super) const METADATA_U8: u8 = 5;
    pub(super) const SPARSE_U8: u8 = 30;

    pub(super) const BLOCK_U16: u8 = 7;
    pub(super) const METADATA_U16: u8 = 8;
    pub(super) const SPARSE_U16: u8 = 31;

    pub(super) const BLOCK_U32: u8 = 9;
    pub(super) const METADATA_U32: u8 = 10;
    pub(super) const SPARSE_U32: u8 = 32;

    pub(super) const BLOCK_U64: u8 = 11;
    pub(super) const METADATA_U64: u8 = 12;
    pub(super) const SPARSE_U64: u8 = 33;

    pub(super) const BLOCK_I8: u8 = 13;
    pub(super) const METADATA_I8: u8 = 14;
    pub(super) const SPARSE_I8: u8 = 34;

    pub(super) const BLOCK_I16: u8 = 15;
    pub(super) const METADATA_I16: u8 = 16;
    pub(super) const SPARSE_I16: u8 = 35;

    pub(super) const BLOCK_I32: u8 = 17;
    pub(super) const METADATA_I32: u8 = 18;
    pub(super) const SPARSE_I32: u8 = 36;

    pub(super) const BLOCK_I64: u8 = 19;
    pub(super) const METADATA_I64: u8 = 20;
    pub(super) const SPARSE_I64: u8 = 37;

    pub(super) const BLOCK_F32: u8 = 1;
    pub(super) const METADATA_F32: u8 = 2;
    pub(super) const SPARSE_F32: u8 = 38;

    pub(super) const BLOCK_F64: u8 = 4;
    pub(super) const METADATA_F64: u8 = 6;
    pub(super) const SPARSE_F64: u8 = 39;

    #[cfg(feature = "complex")]
    pub(super) const BLOCK_C32: u8 = 21;
    #[cfg(feature = "complex")]
    pub(super) const METADATA_C32: u8 = 22;
    #[cfg(feature = "complex")]
    pub(super) const SPARSE_C32: u8 = 40;

    #[cfg(feature = "complex")]
    pub(super) const BLOCK_C64: u8 = 23;
    #[cfg(feature = "complex")]
    pub(super) const METADATA_C64: u8 = 24;
    #[cfg(feature = "complex")]
    pub(super) const SPARSE_C64: u8 = 41;
}

const MAX_METADATA_RANK: usize = 4096;

#[derive(Clone, Debug)]
pub enum FsEntry {
    SparseU8(fensor::SparseNode<u8>),
    SparseU16(fensor::SparseNode<u16>),
    SparseU32(fensor::SparseNode<u32>),
    SparseU64(fensor::SparseNode<u64>),
    SparseI8(fensor::SparseNode<i8>),
    SparseI16(fensor::SparseNode<i16>),
    SparseI32(fensor::SparseNode<i32>),
    SparseI64(fensor::SparseNode<i64>),
    SparseF32(fensor::SparseNode<f32>),
    SparseF64(fensor::SparseNode<f64>),
    #[cfg(feature = "complex")]
    SparseC32(fensor::SparseNode<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    SparseC64(fensor::SparseNode<fensor::complex::Complex64>),
    F32(Vec<f32>),
    MetadataF32(fensor::TensorMetadata<f32>),
    U8(Vec<u8>),
    F64(Vec<f64>),
    MetadataU8(fensor::TensorMetadata<u8>),
    MetadataF64(fensor::TensorMetadata<f64>),
    U16(Vec<u16>),
    MetadataU16(fensor::TensorMetadata<u16>),
    U32(Vec<u32>),
    MetadataU32(fensor::TensorMetadata<u32>),
    U64(Vec<u64>),
    MetadataU64(fensor::TensorMetadata<u64>),
    I8(Vec<i8>),
    MetadataI8(fensor::TensorMetadata<i8>),
    I16(Vec<i16>),
    MetadataI16(fensor::TensorMetadata<i16>),
    I32(Vec<i32>),
    MetadataI32(fensor::TensorMetadata<i32>),
    I64(Vec<i64>),
    MetadataI64(fensor::TensorMetadata<i64>),
    #[cfg(feature = "complex")]
    C32(Vec<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    MetadataC32(fensor::TensorMetadata<fensor::complex::Complex32>),
    #[cfg(feature = "complex")]
    C64(Vec<fensor::complex::Complex64>),
    #[cfg(feature = "complex")]
    MetadataC64(fensor::TensorMetadata<fensor::complex::Complex64>),
}

impl GetSize for FsEntry {
    fn get_heap_size(&self) -> usize {
        match self {
            Self::U8(values) => values.capacity() * std::mem::size_of::<u8>(),
            Self::MetadataU8(metadata) => metadata.get_heap_size(),
            Self::SparseU8(node) => node.get_heap_size(),
            Self::U16(values) => values.capacity() * std::mem::size_of::<u16>(),
            Self::MetadataU16(metadata) => metadata.get_heap_size(),
            Self::SparseU16(node) => node.get_heap_size(),
            Self::U32(values) => values.capacity() * std::mem::size_of::<u32>(),
            Self::MetadataU32(metadata) => metadata.get_heap_size(),
            Self::SparseU32(node) => node.get_heap_size(),
            Self::U64(values) => values.capacity() * std::mem::size_of::<u64>(),
            Self::MetadataU64(metadata) => metadata.get_heap_size(),
            Self::SparseU64(node) => node.get_heap_size(),
            Self::I8(values) => values.capacity() * std::mem::size_of::<i8>(),
            Self::MetadataI8(metadata) => metadata.get_heap_size(),
            Self::SparseI8(node) => node.get_heap_size(),
            Self::I16(values) => values.capacity() * std::mem::size_of::<i16>(),
            Self::MetadataI16(metadata) => metadata.get_heap_size(),
            Self::SparseI16(node) => node.get_heap_size(),
            Self::I32(values) => values.capacity() * std::mem::size_of::<i32>(),
            Self::MetadataI32(metadata) => metadata.get_heap_size(),
            Self::SparseI32(node) => node.get_heap_size(),
            Self::I64(values) => values.capacity() * std::mem::size_of::<i64>(),
            Self::MetadataI64(metadata) => metadata.get_heap_size(),
            Self::SparseI64(node) => node.get_heap_size(),
            Self::F32(values) => values.capacity() * std::mem::size_of::<f32>(),
            Self::MetadataF32(metadata) => metadata.get_heap_size(),
            Self::SparseF32(node) => node.get_heap_size(),
            Self::F64(values) => values.capacity() * std::mem::size_of::<f64>(),
            Self::MetadataF64(metadata) => metadata.get_heap_size(),
            Self::SparseF64(node) => node.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C32(values) => {
                values.capacity() * std::mem::size_of::<fensor::complex::Complex32>()
            }
            #[cfg(feature = "complex")]
            Self::MetadataC32(metadata) => metadata.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::SparseC32(node) => node.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::C64(values) => {
                values.capacity() * std::mem::size_of::<fensor::complex::Complex64>()
            }
            #[cfg(feature = "complex")]
            Self::MetadataC64(metadata) => metadata.get_heap_size(),
            #[cfg(feature = "complex")]
            Self::SparseC64(node) => node.get_heap_size(),
        }
    }
}

impl<'en> en::ToStream<'en> for FsEntry {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        match self {
            Self::SparseU8(node) => {
                en::IntoStream::into_stream((tag::SPARSE_U8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU16(node) => {
                en::IntoStream::into_stream((tag::SPARSE_U16, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU32(node) => {
                en::IntoStream::into_stream((tag::SPARSE_U32, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseU64(node) => {
                en::IntoStream::into_stream((tag::SPARSE_U64, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI8(node) => {
                en::IntoStream::into_stream((tag::SPARSE_I8, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI16(node) => {
                en::IntoStream::into_stream((tag::SPARSE_I16, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI32(node) => {
                en::IntoStream::into_stream((tag::SPARSE_I32, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseI64(node) => {
                en::IntoStream::into_stream((tag::SPARSE_I64, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseF32(node) => {
                en::IntoStream::into_stream((tag::SPARSE_F32, sparse_codec::Encoded(node)), encoder)
            }
            Self::SparseF64(node) => {
                en::IntoStream::into_stream((tag::SPARSE_F64, sparse_codec::Encoded(node)), encoder)
            }
            #[cfg(feature = "complex")]
            Self::SparseC32(node) => {
                en::IntoStream::into_stream((tag::SPARSE_C32, sparse_codec::Encoded(node)), encoder)
            }
            #[cfg(feature = "complex")]
            Self::SparseC64(node) => {
                en::IntoStream::into_stream((tag::SPARSE_C64, sparse_codec::Encoded(node)), encoder)
            }
            Self::U16(value) => en::IntoStream::into_stream((tag::BLOCK_U16, value), encoder),
            Self::MetadataU16(value) => {
                en::IntoStream::into_stream((tag::METADATA_U16, value), encoder)
            }
            Self::U32(value) => en::IntoStream::into_stream((tag::BLOCK_U32, value), encoder),
            Self::MetadataU32(value) => {
                en::IntoStream::into_stream((tag::METADATA_U32, value), encoder)
            }
            Self::U64(value) => en::IntoStream::into_stream((tag::BLOCK_U64, value), encoder),
            Self::MetadataU64(value) => {
                en::IntoStream::into_stream((tag::METADATA_U64, value), encoder)
            }
            Self::I8(value) => en::IntoStream::into_stream((tag::BLOCK_I8, value), encoder),
            Self::MetadataI8(value) => {
                en::IntoStream::into_stream((tag::METADATA_I8, value), encoder)
            }
            Self::I16(value) => en::IntoStream::into_stream((tag::BLOCK_I16, value), encoder),
            Self::MetadataI16(value) => {
                en::IntoStream::into_stream((tag::METADATA_I16, value), encoder)
            }
            Self::I32(value) => en::IntoStream::into_stream((tag::BLOCK_I32, value), encoder),
            Self::MetadataI32(value) => {
                en::IntoStream::into_stream((tag::METADATA_I32, value), encoder)
            }
            Self::I64(value) => en::IntoStream::into_stream((tag::BLOCK_I64, value), encoder),
            Self::MetadataI64(value) => {
                en::IntoStream::into_stream((tag::METADATA_I64, value), encoder)
            }
            #[cfg(feature = "complex")]
            Self::C32(value) => en::IntoStream::into_stream(
                (
                    tag::BLOCK_C32,
                    value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>(),
                ),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC32(value) => {
                en::IntoStream::into_stream((tag::METADATA_C32, value), encoder)
            }
            #[cfg(feature = "complex")]
            Self::C64(value) => en::IntoStream::into_stream(
                (
                    tag::BLOCK_C64,
                    value.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>(),
                ),
                encoder,
            ),
            #[cfg(feature = "complex")]
            Self::MetadataC64(value) => {
                en::IntoStream::into_stream((tag::METADATA_C64, value), encoder)
            }
            Self::F32(value) => en::IntoStream::into_stream((tag::BLOCK_F32, value), encoder),
            Self::MetadataF32(value) => {
                en::IntoStream::into_stream((tag::METADATA_F32, value), encoder)
            }
            Self::U8(value) => en::IntoStream::into_stream((tag::BLOCK_U8, value), encoder),
            Self::F64(value) => en::IntoStream::into_stream((tag::BLOCK_F64, value), encoder),
            Self::MetadataU8(value) => {
                en::IntoStream::into_stream((tag::METADATA_U8, value), encoder)
            }
            Self::MetadataF64(value) => {
                en::IntoStream::into_stream((tag::METADATA_F64, value), encoder)
            }
        }
    }
}

/// Reject oversized payloads while decoding, before an untrusted length allocates.
struct BlockValues<T>(Vec<T>);

impl<T: de::FromStream<Context = ()> + Send> de::FromStream for BlockValues<T> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);

        impl<T: de::FromStream<Context = ()> + Send> de::Visitor for Visitor<T> {
            type Value = BlockValues<T>;

            fn expecting() -> &'static str {
                "one bounded tensor block"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element(()).await? {
                    if values.len() == fensor::MAX_BLOCK_CAPACITY {
                        return Err(de::Error::custom("tensor block exceeds storage capacity"));
                    }
                    values.push(value);
                }
                Ok(BlockValues(values))
            }
        }
        decoder.decode_seq(Visitor(std::marker::PhantomData)).await
    }
}

struct FsEntryVisitor;

impl de::Visitor for FsEntryVisitor {
    type Value = FsEntry;

    fn expecting() -> &'static str {
        "a typed filesystem entry"
    }

    async fn visit_seq<A: de::SeqAccess>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let entry = match seq.expect_next::<u8>(()).await? {
            tag::SPARSE_U8 => {
                FsEntry::SparseU8(seq.expect_next::<sparse_codec::Decoded<u8>>(()).await?.0)
            }
            tag::SPARSE_U16 => {
                FsEntry::SparseU16(seq.expect_next::<sparse_codec::Decoded<u16>>(()).await?.0)
            }
            tag::SPARSE_U32 => {
                FsEntry::SparseU32(seq.expect_next::<sparse_codec::Decoded<u32>>(()).await?.0)
            }
            tag::SPARSE_U64 => {
                FsEntry::SparseU64(seq.expect_next::<sparse_codec::Decoded<u64>>(()).await?.0)
            }
            tag::SPARSE_I8 => {
                FsEntry::SparseI8(seq.expect_next::<sparse_codec::Decoded<i8>>(()).await?.0)
            }
            tag::SPARSE_I16 => {
                FsEntry::SparseI16(seq.expect_next::<sparse_codec::Decoded<i16>>(()).await?.0)
            }
            tag::SPARSE_I32 => {
                FsEntry::SparseI32(seq.expect_next::<sparse_codec::Decoded<i32>>(()).await?.0)
            }
            tag::SPARSE_I64 => {
                FsEntry::SparseI64(seq.expect_next::<sparse_codec::Decoded<i64>>(()).await?.0)
            }
            tag::SPARSE_F32 => {
                FsEntry::SparseF32(seq.expect_next::<sparse_codec::Decoded<f32>>(()).await?.0)
            }
            tag::SPARSE_F64 => {
                FsEntry::SparseF64(seq.expect_next::<sparse_codec::Decoded<f64>>(()).await?.0)
            }
            #[cfg(feature = "complex")]
            tag::SPARSE_C32 => FsEntry::SparseC32(
                seq.expect_next::<sparse_codec::Decoded<fensor::complex::Complex32>>(())
                    .await?
                    .0,
            ),
            #[cfg(feature = "complex")]
            tag::SPARSE_C64 => FsEntry::SparseC64(
                seq.expect_next::<sparse_codec::Decoded<fensor::complex::Complex64>>(())
                    .await?
                    .0,
            ),
            tag::BLOCK_F32 => FsEntry::F32(seq.expect_next::<BlockValues<f32>>(()).await?.0),
            tag::METADATA_F32 => FsEntry::MetadataF32(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_U8 => FsEntry::U8(seq.expect_next::<BlockValues<u8>>(()).await?.0),
            tag::BLOCK_F64 => FsEntry::F64(seq.expect_next::<BlockValues<f64>>(()).await?.0),
            tag::METADATA_U8 => FsEntry::MetadataU8(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::METADATA_F64 => FsEntry::MetadataF64(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_U16 => FsEntry::U16(seq.expect_next::<BlockValues<u16>>(()).await?.0),
            tag::METADATA_U16 => FsEntry::MetadataU16(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_U32 => FsEntry::U32(seq.expect_next::<BlockValues<u32>>(()).await?.0),
            tag::METADATA_U32 => FsEntry::MetadataU32(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_U64 => FsEntry::U64(seq.expect_next::<BlockValues<u64>>(()).await?.0),
            tag::METADATA_U64 => FsEntry::MetadataU64(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_I8 => FsEntry::I8(seq.expect_next::<BlockValues<i8>>(()).await?.0),
            tag::METADATA_I8 => FsEntry::MetadataI8(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_I16 => FsEntry::I16(seq.expect_next::<BlockValues<i16>>(()).await?.0),
            tag::METADATA_I16 => FsEntry::MetadataI16(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_I32 => FsEntry::I32(seq.expect_next::<BlockValues<i32>>(()).await?.0),
            tag::METADATA_I32 => FsEntry::MetadataI32(seq.expect_next(MAX_METADATA_RANK).await?),
            tag::BLOCK_I64 => FsEntry::I64(seq.expect_next::<BlockValues<i64>>(()).await?.0),
            tag::METADATA_I64 => FsEntry::MetadataI64(seq.expect_next(MAX_METADATA_RANK).await?),
            #[cfg(feature = "complex")]
            tag::BLOCK_C32 => FsEntry::C32(
                seq.expect_next::<BlockValues<(f32, f32)>>(())
                    .await?
                    .0
                    .into_iter()
                    .map(|(re, im)| fensor::complex::Complex32::new(re, im))
                    .collect(),
            ),
            #[cfg(feature = "complex")]
            tag::METADATA_C32 => FsEntry::MetadataC32(seq.expect_next(MAX_METADATA_RANK).await?),
            #[cfg(feature = "complex")]
            tag::BLOCK_C64 => FsEntry::C64(
                seq.expect_next::<BlockValues<(f64, f64)>>(())
                    .await?
                    .0
                    .into_iter()
                    .map(|(re, im)| fensor::complex::Complex64::new(re, im))
                    .collect(),
            ),
            #[cfg(feature = "complex")]
            tag::METADATA_C64 => FsEntry::MetadataC64(seq.expect_next(MAX_METADATA_RANK).await?),
            tag => return Err(de::Error::custom(format!("unknown entry tag {tag}"))),
        };

        if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
            return Err(de::Error::custom("unexpected entry field"));
        }

        Ok(entry)
    }
}

impl de::FromStream for FsEntry {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _: (),
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        decoder.decode_seq(FsEntryVisitor).await
    }
}

as_type!(FsEntry, F32, Vec<f32>);
as_type!(FsEntry, MetadataF32, fensor::TensorMetadata<f32>);
as_type!(FsEntry, U8, Vec<u8>);
as_type!(FsEntry, F64, Vec<f64>);
as_type!(FsEntry, MetadataU8, fensor::TensorMetadata<u8>);
as_type!(FsEntry, MetadataF64, fensor::TensorMetadata<f64>);

as_type!(FsEntry, U16, Vec<u16>);
as_type!(FsEntry, MetadataU16, fensor::TensorMetadata<u16>);

as_type!(FsEntry, U32, Vec<u32>);
as_type!(FsEntry, MetadataU32, fensor::TensorMetadata<u32>);

as_type!(FsEntry, U64, Vec<u64>);
as_type!(FsEntry, MetadataU64, fensor::TensorMetadata<u64>);

as_type!(FsEntry, I8, Vec<i8>);
as_type!(FsEntry, MetadataI8, fensor::TensorMetadata<i8>);

as_type!(FsEntry, I16, Vec<i16>);
as_type!(FsEntry, MetadataI16, fensor::TensorMetadata<i16>);

as_type!(FsEntry, I32, Vec<i32>);
as_type!(FsEntry, MetadataI32, fensor::TensorMetadata<i32>);

as_type!(FsEntry, I64, Vec<i64>);
as_type!(FsEntry, MetadataI64, fensor::TensorMetadata<i64>);

#[cfg(feature = "complex")]
as_type!(FsEntry, C32, Vec<fensor::complex::Complex32>);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    MetadataC32,
    fensor::TensorMetadata<fensor::complex::Complex32>
);

#[cfg(feature = "complex")]
as_type!(FsEntry, C64, Vec<fensor::complex::Complex64>);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    MetadataC64,
    fensor::TensorMetadata<fensor::complex::Complex64>
);

impl freqfs::FileLoad for FsEntry {
    async fn load(
        _: &std::path::Path,
        file: tokio::fs::File,
        _: std::fs::Metadata,
    ) -> std::io::Result<Self> {
        counters::record_load();
        tbon::de::read_from((), file)
            .await
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}

impl freqfs::FileSave for FsEntry {
    async fn save(&self, file: &mut tokio::fs::File) -> std::io::Result<u64> {
        counters::record_save();

        let mut file = tokio::io::BufWriter::new(file);
        let mut stream = tbon::en::encode(self).map_err(std::io::Error::other)?;
        let mut size = 0;

        while let Some(chunk) = stream.try_next().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
            size += chunk.len() as u64;
        }

        file.flush().await?;
        Ok(size)
    }
}

as_type!(FsEntry, SparseU8, fensor::SparseNode<u8>);
as_type!(FsEntry, SparseU16, fensor::SparseNode<u16>);
as_type!(FsEntry, SparseU32, fensor::SparseNode<u32>);
as_type!(FsEntry, SparseU64, fensor::SparseNode<u64>);
as_type!(FsEntry, SparseI8, fensor::SparseNode<i8>);
as_type!(FsEntry, SparseI16, fensor::SparseNode<i16>);
as_type!(FsEntry, SparseI32, fensor::SparseNode<i32>);
as_type!(FsEntry, SparseI64, fensor::SparseNode<i64>);
as_type!(FsEntry, SparseF32, fensor::SparseNode<f32>);
as_type!(FsEntry, SparseF64, fensor::SparseNode<f64>);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    SparseC32,
    fensor::SparseNode<fensor::complex::Complex32>
);
#[cfg(feature = "complex")]
as_type!(
    FsEntry,
    SparseC64,
    fensor::SparseNode<fensor::complex::Complex64>
);
