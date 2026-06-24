use std::io;

use b_table::{IndexSchema, Schema};
use ha_ndarray::{Shape, Strides};
use smallvec::SmallVec;

use crate::{Error, Result as FResult};

pub const PORTABLE_INLINE_RANK: usize = 8;
pub type TensorShape = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct ViewAxisSchema {
    pub base_axis: usize,
    pub map: ViewAxisMapSchema,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum ViewAxisMapSchema {
    Identity,
    Affine { start: u64, step: u64 },
    Gather(TensorShape),
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct ViewSchema {
    pub base_rank: usize,
    pub axes: SmallVec<[ViewAxisSchema; PORTABLE_INLINE_RANK]>,
    pub base_fixed: SmallVec<[Option<u64>; PORTABLE_INLINE_RANK]>,
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct InternalLayoutMetadata {
    layout: Layout,
    block_shape: Shape,
    strides: Strides,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub fn try_parse(value: &str) -> Option<Self> {
        match value {
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Layout {
    Dense,
    Sparse { axis: Option<usize> },
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorSchema {
    dtype: DType,
    shape: Shape,
    internal: InternalLayoutMetadata,
}

impl TensorSchema {
    pub(crate) fn shape_u64(&self) -> FResult<TensorShape> {
        Self::shape_u64_from_shape(&self.shape)
    }

    pub(crate) fn shape_usize_from_u64(shape: &[u64]) -> FResult<Shape> {
        validate_shape_dims(shape)?;

        shape
            .iter()
            .map(|dim| {
                usize::try_from(*dim)
                    .map_err(|_| Error::InvalidSchema("shape dimension overflow".to_string()))
            })
            .collect::<FResult<Vec<usize>>>()
            .map(Into::into)
    }

    fn shape_u64_from_shape(shape: &Shape) -> FResult<TensorShape> {
        validate_shape_dims(shape.as_slice())?;

        shape
            .iter()
            .map(|dim| {
                u64::try_from(*dim)
                    .map_err(|_| Error::InvalidSchema("shape dimension overflow".to_string()))
            })
            .collect::<FResult<TensorShape>>()
    }

    pub fn new(
        dtype: DType,
        shape: Shape,
        layout: Layout,
        block_shape: Shape,
        strides: Strides,
    ) -> FResult<Self> {
        validate_shape_dims(shape.as_slice())?;

        if block_shape.len() != shape.len() || block_shape.contains(&0) {
            return Err(Error::InvalidSchema(
                "block_shape must be non-zero and match tensor rank".to_string(),
            ));
        }

        if strides.len() != shape.len() {
            return Err(Error::InvalidSchema(
                "strides rank must match tensor shape rank".to_string(),
            ));
        }

        if let Layout::Sparse { axis: Some(axis) } = layout
            && axis >= shape.len()
        {
            return Err(Error::InvalidSchema(
                "sparse axis hint out of bounds".to_string(),
            ));
        }

        // Validate that shape dimensions are representable in the portable u64 form.
        let _ = Self::shape_u64_from_shape(&shape)?;

        let internal = InternalLayoutMetadata {
            layout,
            block_shape,
            strides,
        };

        Ok(Self {
            dtype,
            shape,
            internal,
        })
    }

    pub fn dense_with_dtype(dtype: DType, shape: Shape, block_shape: Shape) -> FResult<Self> {
        let strides = contiguous_strides(&shape);
        Self::new(dtype, shape, Layout::Dense, block_shape, strides)
    }

    pub fn sparse_with_dtype(
        dtype: DType,
        shape: Shape,
        block_shape: Shape,
        axis: Option<usize>,
    ) -> FResult<Self> {
        let strides = contiguous_strides(&shape);
        Self::new(dtype, shape, Layout::Sparse { axis }, block_shape, strides)
    }

    pub fn dense(shape: Shape, block_shape: Shape) -> FResult<Self> {
        Self::dense_with_dtype(DType::F32, shape, block_shape)
    }

    pub fn sparse(shape: Shape, block_shape: Shape, axis: Option<usize>) -> FResult<Self> {
        Self::sparse_with_dtype(DType::F32, shape, block_shape, axis)
    }

    pub fn block_len(&self) -> usize {
        self.internal.block_shape.iter().product()
    }

    pub fn validate_coord(&self, coord: &[u64]) -> FResult<()> {
        let shape = self.shape();

        if coord.len() != shape.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        for (i, (c, dim)) in coord.iter().zip(shape.iter()).enumerate() {
            let coord = usize::try_from(*c).map_err(|_| {
                Error::InvalidCoord(format!("coordinate at axis {i} overflows usize"))
            })?;

            if coord >= *dim {
                return Err(Error::InvalidCoord(format!(
                    "coordinate at axis {i} is out of bounds"
                )));
            }
        }

        Ok(())
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn set_shape(&mut self, shape: Shape) -> FResult<()> {
        validate_shape_dims(shape.as_slice())?;

        // Preserve portability invariant by ensuring we can encode this shape as u64.
        let _ = Self::shape_u64_from_shape(&shape)?;
        self.shape = shape;
        Ok(())
    }

    pub fn layout(&self) -> &Layout {
        &self.internal.layout
    }

    pub fn block_shape(&self) -> &Shape {
        &self.internal.block_shape
    }

    pub fn strides(&self) -> &Strides {
        &self.internal.strides
    }

    pub fn set_strides(&mut self, strides: Strides) -> FResult<()> {
        if strides.len() != self.rank() {
            return Err(Error::InvalidSchema(
                "strides rank must match tensor shape rank".to_string(),
            ));
        }

        self.internal.strides = strides;
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

fn validate_shape_dims<T>(shape: &[T]) -> FResult<()>
where
    T: Copy + PartialEq + From<u8>,
{
    if shape.is_empty() {
        return Err(Error::InvalidSchema(
            "tensor shape cannot be empty".to_string(),
        ));
    }

    if shape.iter().any(|dim| *dim == T::from(0u8)) {
        return Err(Error::InvalidSchema(
            "tensor shape dimensions must be non-zero".to_string(),
        ));
    }

    Ok(())
}

pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];

    for i in (0..ndim).rev() {
        if i + 1 < ndim {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    strides.into()
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SparseIndexSchema {
    columns: Vec<String>,
}

impl SparseIndexSchema {
    pub fn new(columns: Vec<String>) -> Self {
        Self { columns }
    }
}

impl b_table::BTreeSchema for SparseIndexSchema {
    type Error = io::Error;
    type Value = u64;

    fn block_size(&self) -> usize {
        4096
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    fn order(&self) -> usize {
        16
    }

    fn validate_key(
        &self,
        key: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse index key length",
            ))
        }
    }
}

impl IndexSchema for SparseIndexSchema {
    type Id = String;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SparseTableSchema {
    primary: SparseIndexSchema,
    auxiliary: Vec<(String, SparseIndexSchema)>,
}

impl Default for SparseTableSchema {
    fn default() -> Self {
        Self {
            primary: SparseIndexSchema::new(vec![
                "coord".to_string(),
                "block_offset".to_string(),
                "block_id".to_string(),
            ]),
            auxiliary: vec![],
        }
    }
}

impl Schema for SparseTableSchema {
    type Id = String;
    type Error = io::Error;
    type Value = u64;
    type Index = SparseIndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns()[0..2]
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns()[2..]
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, Self::Index)] {
        &self.auxiliary
    }

    fn validate_key(
        &self,
        key: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if key.len() == 2 {
            Ok(key)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse table key length",
            ))
        }
    }

    fn validate_values(
        &self,
        values: Vec<Self::Value>,
    ) -> std::result::Result<Vec<Self::Value>, Self::Error> {
        if values.len() == 1 {
            Ok(values)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid sparse table value length",
            ))
        }
    }
}
