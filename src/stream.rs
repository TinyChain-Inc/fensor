use destream::{de, en};
use freqfs::DirLock;

use crate::wire_tags::{LAYOUT_TAG_DENSE, LAYOUT_TAG_SPARSE};
use crate::{
    DType, Layout, Tensor, TensorArray, TensorElement, TensorFileEntry, TensorRead, TensorSchema,
    TensorViewSemantics, TensorWrite, schema,
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
        en::IntoStream::into_stream(encode_layout(*self)?, encoder)
    }
}

impl<'en> en::IntoStream<'en> for Layout {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(encode_layout(self)?, encoder)
    }
}

// ---------------------------------------------------------------------------
// Tensor view streaming: lazy encode + streaming decode
// ---------------------------------------------------------------------------

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
    // Encode schema info needed to reconstruct the tensor: (dtype, shape, layout, block_shape)
    let dtype = tensor.schema().dtype();
    let shape: Vec<u64> = tensor.schema().shape().iter().map(|&d| d as u64).collect();
    // A `Sparse { axis }` hint is preserved only when the tensor is currently an
    // identity/base view -- `transpose`/`slice`/`reshape` never update layout when
    // they mutate the view's shape/strides, so a stale axis hint surviving a
    // transform could silently denote the wrong logical axis (or, after a
    // rank-reducing slice, be out of bounds). Resetting it to `None` on any
    // non-identity view sidesteps that silent-corruption risk; `None` is always
    // valid and defaults to axis 0 downstream.
    let is_identity = tensor.is_base_tensor();
    let layout = match tensor.layout() {
        Layout::Dense => Layout::Dense,
        Layout::Sparse { axis } => Layout::Sparse {
            axis: if is_identity { axis } else { None },
        },
    };
    let block_shape: Vec<usize> = tensor.block_shape().to_vec();
    let view_shape: Vec<usize> = tensor.view.shape().to_vec();
    let schema_info = (dtype, shape, layout, block_shape);
    let view_values = ViewSnapshotValues {
        tensor,
        shape: view_shape,
    };
    en::IntoStream::into_stream((schema_info, view_values), encoder)
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

struct ViewSnapshotValues<'a, FE, T> {
    tensor: &'a Tensor<FE, T>,
    shape: Vec<usize>,
}

enum ValuesEncodingState<'a, FE, T> {
    Walking {
        tensor: &'a Tensor<FE, T>,
        coords: schema::RowMajorCoords,
    },
    Done,
}

impl<'a, 'en, FE, T> en::IntoStream<'en> for ViewSnapshotValues<'a, FE, T>
where
    'a: 'en,
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let coords = schema::row_major_coords(&self.shape).map_err(en::Error::custom)?;
        let initial = ValuesEncodingState::Walking {
            tensor: self.tensor,
            coords,
        };
        let stream = Box::pin(futures::stream::unfold(initial, |state| async move {
            let ValuesEncodingState::Walking { tensor, mut coords } = state else {
                return None;
            };

            loop {
                let Some(coord) = coords.next() else {
                    break None;
                };

                let value = tensor.read_value(&coord).await;

                match value {
                    Ok(value) => {
                        if value == T::default() {
                            continue;
                        }

                        break Some((
                            Ok((coord, value)),
                            ValuesEncodingState::Walking { tensor, coords },
                        ));
                    }
                    Err(error) => break Some((Err(format!("{error}")), ValuesEncodingState::Done)),
                }
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
    type Context = DirLock<FE>;

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
    dir: DirLock<FE>,
    _marker: std::marker::PhantomData<T>,
}

impl<FE, T> de::Visitor for TensorViewVisitor<FE, T>
where
    FE: TensorFileEntry<T> + safecast::AsType<String> + From<String>,
    T: TensorElement,
{
    type Value = TensorViewDecoder<FE, T>;

    fn expecting() -> &'static str {
        "a tensor view (schema followed by coord with value sequence)"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let (dtype, shape_u64, layout, block_shape): (DType, Vec<u64>, Layout, Vec<usize>) = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::custom("missing tensor view schema info"))?;

        if dtype != T::DTYPE {
            return Err(de::Error::custom(format!(
                "tensor dtype mismatch: schema {:?} != tensor {:?}",
                dtype,
                T::DTYPE,
            )));
        }

        // Convert shape from u64 to usize
        let shape: Vec<usize> = shape_u64
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| de::Error::custom("shape dimension overflow")))
            .collect::<Result<Vec<_>, _>>()?;

        let schema = TensorSchema::new(dtype, shape.clone().into()).map_err(de::Error::custom)?;
        // Use max_capacity inferred from block_shape
        let max_capacity = block_shape.iter().product::<usize>();
        let tensor = Tensor::<FE, T>::create(self.dir.clone(), schema, layout, max_capacity)
            .await
            .map_err(de::Error::custom)?;

        // Decode the nested pairs sequence, moving `tensor` in by value
        // (Context) and getting it back out as the decoded Value on success
        // -- each nonzero value is written to storage as soon as it arrives
        // off the wire, with no in-memory buffering.
        match seq.next_element::<TensorDecodedValues<FE, T>>(tensor).await {
            Ok(Some(TensorDecodedValues { tensor })) => Ok(TensorViewDecoder { tensor }),
            Ok(None) => Err(de::Error::custom("missing tensor view data")),
            Err(err) => Err(err),
        }
    }
}

struct TensorDecodedValues<FE, T> {
    tensor: Tensor<FE, T>,
}

impl<FE, T> de::FromStream for TensorDecodedValues<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Context = Tensor<FE, T>;

    async fn from_stream<D: de::Decoder>(
        tensor: Tensor<FE, T>,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let tensor = decoder.decode_seq(TensorValuesVisitor { tensor }).await?;
        Ok(Self { tensor })
    }
}

struct TensorValuesVisitor<FE, T> {
    tensor: Tensor<FE, T>,
}

impl<FE, T> de::Visitor for TensorValuesVisitor<FE, T>
where
    FE: TensorFileEntry<T>,
    T: TensorElement,
{
    type Value = Tensor<FE, T>;

    fn expecting() -> &'static str {
        "a sequence of tensor view element pairs (coord, value)"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        loop {
            let Some(value) = seq.next_element::<(Vec<u64>, T)>(()).await? else {
                let Some(err) = seq.next_element::<String>(()).await? else {
                    break;
                };

                return Err(de::Error::custom(err));
            };

            let (coord, value) = value;

            self.tensor
                .write_value(&coord, value)
                .await
                .map_err(de::Error::custom)?;
        }

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
    }
}
