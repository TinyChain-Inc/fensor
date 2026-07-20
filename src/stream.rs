use std::fmt;

use destream::{de, en};

use crate::wire_tags::{ELEM_TAG_ERROR, ELEM_TAG_PAIR, LAYOUT_TAG_DENSE, LAYOUT_TAG_SPARSE};
use crate::{
    DType, Error, Layout, Tensor, TensorArray, TensorElement, TensorFileEntry, TensorRead,
    TensorSchema, TensorViewSemantics, TensorWrite, contiguous_strides, schema,
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

fn encode_tensor_ref<FE, T>(tensor: &Tensor<FE, T>) -> Result<TensorSchema, String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    if !tensor.is_base_tensor() {
        return Err(
            "cannot serialize a tensor with a non-identity view; views are metadata-only and \
            are never persisted"
                .to_string(),
        );
    }

    Ok(tensor.schema().clone())
}

fn encode_tensor<FE, T>(tensor: Tensor<FE, T>) -> Result<TensorSchema, String>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    encode_tensor_ref(&tensor)
}

impl de::FromStream for DType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let dtype = String::from_stream((), decoder).await?;
        DType::try_parse(&dtype)
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
        let schema = TensorSchema::from_stream((), decoder).await?;

        Tensor::create(dir, schema).await.map_err(de::Error::custom)
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

// ---------------------------------------------------------------------------
// Tensor view streaming: lazy encode + streaming decode
// ---------------------------------------------------------------------------

/// A closed, wire-safe classification of a `crate::Error` that occurred while
/// the encoder was lazily reading values from storage. Carries no payload
/// data (no paths, no coordinates, no underlying `io::Error` text) since this
/// crosses a machine boundary; only which failure shape occurred survives.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum ElemErrorCode {
    Io = 0,
    Nd = 1,
    InvalidSchema = 2,
    InvalidCoord = 3,
    InvalidLayout = 4,
    SparseIndex = 5,
    UnsupportedSparseIterationOrder = 6,
    Unsupported = 7,
    DataMismatch = 8,
}

impl ElemErrorCode {
    fn from_error(err: &Error) -> Self {
        match err {
            Error::Io(_) => Self::Io,
            Error::Nd(_) => Self::Nd,
            Error::InvalidSchema(_) => Self::InvalidSchema,
            Error::InvalidCoord(_) => Self::InvalidCoord,
            Error::InvalidLayout(_) => Self::InvalidLayout,
            Error::SparseIndex(_) => Self::SparseIndex,
            Error::UnsupportedSparseIterationOrder { .. } => Self::UnsupportedSparseIterationOrder,
            Error::Unsupported(_) => Self::Unsupported,
            Error::DataMismatch(_) => Self::DataMismatch,
        }
    }

    fn try_from_u8(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Io),
            1 => Some(Self::Nd),
            2 => Some(Self::InvalidSchema),
            3 => Some(Self::InvalidCoord),
            4 => Some(Self::InvalidLayout),
            5 => Some(Self::SparseIndex),
            6 => Some(Self::UnsupportedSparseIterationOrder),
            7 => Some(Self::Unsupported),
            8 => Some(Self::DataMismatch),
            _ => None,
        }
    }
}

impl fmt::Display for ElemErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Io => "I/O error",
            Self::Nd => "array computation error",
            Self::InvalidSchema => "invalid schema",
            Self::InvalidCoord => "invalid coordinate",
            Self::InvalidLayout => "invalid layout",
            Self::SparseIndex => "sparse index error",
            Self::UnsupportedSparseIterationOrder => "unsupported sparse iteration order",
            Self::Unsupported => "unsupported operation",
            Self::DataMismatch => "data mismatch",
        })
    }
}

enum Elem<T> {
    Pair(Vec<u64>, T),
    Error(ElemErrorCode),
}

impl<T> de::FromStream for Elem<T>
where
    T: TensorElement,
{
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (tag, coord, value, error_code): (u8, Vec<u64>, Option<T>, Option<u8>) =
            <(u8, Vec<u64>, Option<T>, Option<u8>)>::from_stream((), decoder).await?;

        match tag {
            ELEM_TAG_PAIR => {
                let value =
                    value.ok_or_else(|| de::Error::custom("missing tensor view pair value"))?;
                Ok(Elem::Pair(coord, value))
            }
            ELEM_TAG_ERROR => {
                let code = error_code
                    .ok_or_else(|| de::Error::custom("missing tensor view error code"))?;
                let code = ElemErrorCode::try_from_u8(code).ok_or_else(|| {
                    de::Error::custom(format!("unknown tensor view error code {code}"))
                })?;
                Ok(Elem::Error(code))
            }
            _ => Err(de::Error::custom(format!(
                "unknown tensor view stream element tag {tag}"
            ))),
        }
    }
}

impl<'en, T> en::ToStream<'en> for Elem<T>
where
    T: TensorElement,
{
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Elem::Pair(coord, value) => en::IntoStream::into_stream(
                (ELEM_TAG_PAIR, coord.clone(), Some(*value), None::<u8>),
                encoder,
            ),
            Elem::Error(code) => en::IntoStream::into_stream(
                (
                    ELEM_TAG_ERROR,
                    Vec::<u64>::new(),
                    None::<T>,
                    Some(*code as u8),
                ),
                encoder,
            ),
        }
    }
}

impl<'en, T> en::IntoStream<'en> for Elem<T>
where
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Elem::Pair(coord, value) => en::IntoStream::into_stream(
                (ELEM_TAG_PAIR, coord, Some(value), None::<u8>),
                encoder,
            ),
            Elem::Error(code) => en::IntoStream::into_stream(
                (
                    ELEM_TAG_ERROR,
                    Vec::<u64>::new(),
                    None::<T>,
                    Some(code as u8),
                ),
                encoder,
            ),
        }
    }
}

pub struct TensorViewEncoder<'a, FE, T> {
    tensor: &'a Tensor<FE, T>,
}

impl<'a, FE, T> TensorViewEncoder<'a, FE, T> {
    pub(crate) fn new(tensor: &'a Tensor<FE, T>) -> Self {
        Self { tensor }
    }
}

fn encode_view<'en, E, FE, T>(tensor: &'en Tensor<FE, T>, encoder: E) -> Result<E::Ok, E::Error>
where
    E: en::Encoder<'en>,
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    let is_identity = tensor.is_base_tensor();
    let schema =
        schema::snapshot_schema(tensor.schema(), is_identity).map_err(en::Error::custom)?;
    let shape: Vec<usize> = tensor.schema().shape().iter().copied().collect();
    let pairs = TensorPairs { tensor, shape };
    en::IntoStream::into_stream((schema, pairs), encoder)
}

impl<'en, FE, T> en::ToStream<'en> for TensorViewEncoder<'_, FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encode_view(self.tensor, encoder)
    }
}

impl<'a, 'en, FE, T> en::IntoStream<'en> for TensorViewEncoder<'a, FE, T>
where
    'a: 'en,
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encode_view(self.tensor, encoder)
    }
}

struct TensorPairs<'a, FE, T> {
    tensor: &'a Tensor<FE, T>,
    shape: Vec<usize>,
}

enum PairState<'a, FE, T> {
    Walking {
        tensor: &'a Tensor<FE, T>,
        coords: crate::schema::RowMajorCoords,
    },
    Done,
}

impl<'a, 'en, FE, T> en::IntoStream<'en> for TensorPairs<'a, FE, T>
where
    'a: 'en,
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let coords = crate::schema::row_major_coords(&self.shape).map_err(en::Error::custom)?;
        let initial = PairState::Walking {
            tensor: self.tensor,
            coords,
        };
        let stream = Box::pin(futures::stream::unfold(initial, |state| async move {
            match state {
                PairState::Walking { tensor, mut coords } => loop {
                    match coords.next() {
                        Some(coord) => match tensor.read_value(&coord).await {
                            Ok(value) if value == T::default() => continue,
                            Ok(value) => {
                                break Some((
                                    Elem::Pair(coord, value),
                                    PairState::Walking { tensor, coords },
                                ));
                            }
                            Err(err) => {
                                break Some((
                                    Elem::Error(ElemErrorCode::from_error(&err)),
                                    PairState::Done,
                                ));
                            }
                        },
                        None => break None,
                    }
                },
                PairState::Done => None,
            }
        }));
        encoder.encode_seq_stream(stream)
    }
}

pub struct TensorViewDecoder<FE, T> {
    tensor: Tensor<FE, T>,
}

impl<FE, T> TensorViewDecoder<FE, T> {
    pub fn into_inner(self) -> Tensor<FE, T> {
        self.tensor
    }
}

impl<FE, T> de::FromStream for TensorViewDecoder<FE, T>
where
    FE: TensorFileEntry<T> + safecast::AsType<String> + From<String>,
    T: TensorElement,
{
    type Context = freqfs::DirLock<FE>;

    async fn from_stream<D: de::Decoder>(
        dir: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder
            .decode_seq(TensorViewVisitor {
                dir,
                _marker: std::marker::PhantomData,
            })
            .await
    }
}

struct TensorViewVisitor<FE, T> {
    dir: freqfs::DirLock<FE>,
    _marker: std::marker::PhantomData<T>,
}

impl<FE, T> de::Visitor for TensorViewVisitor<FE, T>
where
    FE: TensorFileEntry<T> + safecast::AsType<String> + From<String>,
    T: TensorElement,
{
    type Value = TensorViewDecoder<FE, T>;

    fn expecting() -> &'static str {
        "a tensor view (schema followed by pairs sequence)"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema: TensorSchema = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::custom("missing tensor view schema"))?;

        if schema.dtype() != T::DTYPE {
            return Err(de::Error::custom(format!(
                "tensor dtype mismatch: schema {:?} != tensor {:?}",
                schema.dtype(),
                T::DTYPE,
            )));
        }

        let tensor = Tensor::<FE, T>::create(self.dir.clone(), schema)
            .await
            .map_err(de::Error::custom)?;

        // Decode the nested pairs sequence, moving `tensor` in by value
        // (Context) and getting it back out as the decoded Value on success
        // -- each nonzero value is written to storage as soon as it arrives
        // off the wire, with no in-memory buffering.
        match seq.next_element::<DecodedPairs<FE, T>>(tensor).await {
            Ok(Some(DecodedPairs { tensor })) => Ok(TensorViewDecoder { tensor }),
            Ok(None) => {
                let mut dir_guard = self.dir.write().await;
                let _ = dir_guard.truncate_and_sync().await;
                Err(de::Error::custom("missing tensor view data"))
            }
            Err(err) => {
                let mut dir_guard = self.dir.write().await;
                let _ = dir_guard.truncate_and_sync().await;
                Err(err)
            }
        }
    }
}

struct DecodedPairs<FE, T> {
    tensor: Tensor<FE, T>,
}

impl<FE, T> de::FromStream for DecodedPairs<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Context = Tensor<FE, T>;

    async fn from_stream<D: de::Decoder>(
        tensor: Tensor<FE, T>,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let tensor = decoder.decode_seq(PairsVisitor { tensor }).await?;
        Ok(Self { tensor })
    }
}

struct PairsVisitor<FE, T> {
    tensor: Tensor<FE, T>,
}

impl<FE, T> de::Visitor for PairsVisitor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Value = Tensor<FE, T>;

    fn expecting() -> &'static str {
        "a sequence of tensor view element pairs (coord, value)"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        while let Some(elem) = seq.next_element::<Elem<T>>(()).await? {
            match elem {
                Elem::Pair(coord, value) => {
                    if value == T::default() {
                        // defensive: encode never emits defaults, but don't
                        // trust the wire.
                        continue;
                    }
                    self.tensor
                        .write_value(&coord, value)
                        .await
                        .map_err(de::Error::custom)?;
                }
                Elem::Error(code) => {
                    return Err(de::Error::custom(format!(
                        "tensor view encode-side failure: {code}"
                    )));
                }
            }
        }
        // Natural exhaustion of the sequence is itself the success signal --
        // `tbon`'s SeqAccess only returns `None` after consuming a real
        // `LIST_END` delimiter, so a genuinely truncated stream surfaces as
        // a decode error before this point is ever reached.
        Ok(self.tensor)
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
    }
}
