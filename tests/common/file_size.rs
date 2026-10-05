//! Preflight the adapter's tagged containers without allocating their payloads.
use std::mem::size_of;

use destream::de;
use fensor::{SparseCell, TensorMetadata};

use super::sparse_codec::{MAX_NODE_ENTRIES, MAX_NODE_VALUES};

const MAX_ENTRIES: usize = 4096;

pub struct Size(pub usize);

fn add<E: de::Error>(left: usize, right: usize) -> Result<usize, E> {
    left.checked_add(right)
        .ok_or_else(|| E::custom("file allocation bound overflow"))
}

fn bytes<E: de::Error>(count: usize, width: usize) -> Result<usize, E> {
    count
        .checked_mul(width)
        .ok_or_else(|| E::custom("file allocation bound overflow"))
}

// Match the decoder's cautious initial capacity and Vec's geometric growth. A
// nonempty allocation has a small minimum capacity, including byte vectors.
fn capacity(len: usize, hint: Option<usize>, width: usize) -> usize {
    let initial = hint.unwrap_or(0).min(MAX_ENTRIES);
    if len <= initial {
        initial
    } else {
        len.next_power_of_two()
            .max(if width == 1 { 8 } else { 4 })
            .max(initial.next_power_of_two())
    }
}

struct Count {
    len: usize,
    capacity: usize,
}

impl de::FromStream for Count {
    type Context = (usize, usize);

    async fn from_stream<D: de::Decoder>(
        (width, limit): Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor(usize, usize);

        impl de::Visitor for Visitor {
            type Value = Count;

            fn expecting() -> &'static str {
                "bounded file sequence"
            }

            async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Count, A::Error> {
                let hint = seq.size_hint();
                let mut len = 0;
                while seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    if len == self.1 {
                        return Err(de::Error::custom("oversized file sequence"));
                    }
                    len += 1;
                }
                Ok(Count {
                    len,
                    capacity: capacity(len, hint, self.0),
                })
            }
        }
        decoder.decode_seq(Visitor(width, limit)).await
    }
}

// Inspect packed payload bytes without retaining encoded or decoded values.
struct PayloadCount(usize);

impl de::FromStream for PayloadCount {
    type Context = usize;

    async fn from_stream<D: de::Decoder>(width: usize, decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor(usize);

        impl de::Visitor for Visitor {
            type Value = PayloadCount;

            fn expecting() -> &'static str {
                "bounded packed sparse payload"
            }

            async fn visit_array_u8<A: de::ArrayAccess<u8>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                let limit = bytes(fensor::MAX_BLOCK_CAPACITY, self.0)?;
                let mut scratch = [0u8; 64];
                let mut len = 0;
                loop {
                    let read = array.buffer(&mut scratch).await?;
                    if read == 0 {
                        break;
                    }
                    len = add(len, read)?;
                    if len > limit {
                        return Err(de::Error::custom("oversized sparse payload"));
                    }
                }
                if len == 0 || len % self.0 != 0 {
                    return Err(de::Error::custom("invalid sparse payload byte length"));
                }
                Ok(PayloadCount(len / self.0))
            }
        }
        decoder.decode_array_u8(Visitor(width)).await
    }
}

struct SparseRowSize {
    values: usize,
    heap: usize,
}

impl de::FromStream for SparseRowSize {
    type Context = usize;

    async fn from_stream<D: de::Decoder>(
        width: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor(usize);

        impl de::Visitor for Visitor {
            type Value = SparseRowSize;

            fn expecting() -> &'static str {
                "dense sparse-row allocation bound"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<SparseRowSize, A::Error> {
                seq.expect_next::<u64>(()).await?;
                let PayloadCount(values) = seq.expect_next::<PayloadCount>(self.0).await?;
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected sparse row field"));
                }

                Ok(SparseRowSize {
                    values,
                    heap: bytes(capacity(values, None, self.0), self.0)?,
                })
            }
        }

        decoder.decode_seq(Visitor(width)).await
    }
}

struct SparseRowsSize(usize);

impl de::FromStream for SparseRowsSize {
    type Context = (usize, usize);

    async fn from_stream<D: de::Decoder>(
        (cell_width, value_width): Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor(usize, usize);

        impl de::Visitor for Visitor {
            type Value = SparseRowsSize;

            fn expecting() -> &'static str {
                "dense sparse-node allocation bound"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<SparseRowsSize, A::Error> {
                let mut len = 0;
                let mut values = 0;
                let mut heap = 0;
                while let Some(row) = seq.next_element::<SparseRowSize>(self.1).await? {
                    if len == MAX_NODE_ENTRIES {
                        return Err(de::Error::custom("oversized sparse node"));
                    }
                    len += 1;
                    values = add(values, row.values)?;
                    if values > MAX_NODE_VALUES {
                        return Err(de::Error::custom("oversized sparse node payloads"));
                    }
                    heap = add(heap, add(bytes(2, self.0)?, row.heap)?)?;
                }

                Ok(SparseRowsSize(add(
                    heap,
                    bytes(
                        capacity(len, None, size_of::<Vec<u64>>()),
                        size_of::<Vec<u64>>(),
                    )?,
                )?))
            }
        }

        decoder.decode_seq(Visitor(cell_width, value_width)).await
    }
}

struct NodeSize(usize);

impl de::FromStream for NodeSize {
    type Context = (usize, usize);

    async fn from_stream<D: de::Decoder>(
        width: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        struct Visitor((usize, usize));

        impl de::Visitor for Visitor {
            type Value = NodeSize;

            fn expecting() -> &'static str {
                "native node allocation bound"
            }

            async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<NodeSize, A::Error> {
                seq.expect_next::<bool>(()).await?;
                let mut bound = size_of::<fensor::SparseNode<u64>>();
                bound = add(bound, seq.expect_next::<SparseRowsSize>(self.0).await?.0)?;
                let children = seq.expect_next::<Count>((16, MAX_NODE_ENTRIES)).await?;
                bound = add(bound, bytes(capacity(children.len, None, 16), 16)?)?;
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected native node field"));
                }
                Ok(NodeSize(bound))
            }
        }
        decoder.decode_seq(Visitor(width)).await
    }
}

struct MetadataSize(usize);

impl de::FromStream for MetadataSize {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = MetadataSize;

            fn expecting() -> &'static str {
                "native metadata allocation bound"
            }

            async fn visit_seq<A: de::SeqAccess>(
                self,
                mut seq: A,
            ) -> Result<MetadataSize, A::Error> {
                let shape = seq
                    .expect_next::<Count>((8, super::MAX_METADATA_RANK))
                    .await?;
                seq.expect_next::<bool>(()).await?;
                seq.expect_next::<Option<u64>>(()).await?;
                let block = seq
                    .expect_next::<Count>((8, super::MAX_METADATA_RANK))
                    .await?;
                let mut bound = size_of::<TensorMetadata<u8>>();
                for count in [shape.len, block.len] {
                    if count > fensor::Shape::new().inline_size() {
                        bound = add(
                            bound,
                            bytes(
                                count
                                    .next_power_of_two()
                                    .max(2 * fensor::Shape::new().inline_size()),
                                8 * 8,
                            )?,
                        )?;
                    }
                }
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected metadata field"));
                }
                Ok(MetadataSize(bound))
            }
        }
        decoder.decode_seq(Visitor).await
    }
}

impl de::FromStream for Size {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        struct Visitor;

        impl de::Visitor for Visitor {
            type Value = Size;

            fn expecting() -> &'static str {
                "tagged file allocation bound"
            }

            async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Size, A::Error> {
                let tag = seq.expect_next::<u8>(()).await?;
                let payload = match tag {
                    2 | 5 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 => {
                        seq.expect_next::<MetadataSize>(()).await?.0
                    }
                    30..=41 => {
                        let width = match tag {
                            30 | 34 => 1,
                            31 | 35 => 2,
                            32 | 36 | 38 => 4,
                            41 => 16,
                            _ => 8,
                        };
                        seq.expect_next::<NodeSize>((size_of::<SparseCell<u64>>(), width))
                            .await?
                            .0
                    }
                    1 | 3 | 4 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 => {
                        let width = match tag {
                            3 | 13 => 1,
                            7 | 15 => 2,
                            1 | 9 | 17 => 4,
                            23 => 16,
                            _ => 8,
                        };
                        let values = seq
                            .expect_next::<Count>((width, fensor::MAX_BLOCK_CAPACITY))
                            .await?;
                        let heap = bytes(values.capacity, width)?;
                        // Complex decoding may retain its component-pair input
                        // until the new output allocation has been assembled.
                        add(
                            size_of::<Vec<u64>>(),
                            if tag == 21 || tag == 23 {
                                add(heap, heap)?
                            } else {
                                heap
                            },
                        )?
                    }
                    _ => return Err(de::Error::custom("unknown file payload tag")),
                };
                if seq.next_element::<de::IgnoredAny>(()).await?.is_some() {
                    return Err(de::Error::custom("unexpected file field"));
                }
                Ok(Size(add(size_of::<super::FsEntry>(), payload)?))
            }
        }
        decoder.decode_seq(Visitor).await
    }
}
