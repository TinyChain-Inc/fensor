use std::io;

use b_table::{IndexSchema, Schema};
use ha_ndarray::{Shape, Strides};
use smallvec::SmallVec;

use crate::{Error, Result as FResult};

pub const PORTABLE_INLINE_RANK: usize = 8;
pub type TensorShape = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

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

    /// Total number of elements described by this schema's shape, computed
    /// with overflow checking so a corrupted/adversarial shape fails closed
    /// instead of silently wrapping.
    pub fn element_count(&self) -> FResult<usize> {
        checked_product(self.shape.as_slice())
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

fn checked_product(shape: &[usize]) -> FResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::InvalidSchema("shape element count overflows usize".to_string()))
}

/// Row-major (C-order) coordinate walk over `shape`, used to materialize or
/// reconstruct a tensor view's data as a flat sequence for wire transfer.
pub(crate) struct RowMajorCoords {
    shape: Vec<usize>,
    coord: Vec<u64>,
    remaining: usize,
}

pub(crate) fn row_major_coords(shape: &[usize]) -> FResult<RowMajorCoords> {
    let remaining = if shape.is_empty() {
        0
    } else {
        checked_product(shape)?
    };

    Ok(RowMajorCoords {
        shape: shape.to_vec(),
        coord: vec![0u64; shape.len()],
        remaining,
    })
}

impl Iterator for RowMajorCoords {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let out = self.coord.clone();
        self.remaining -= 1;

        if self.remaining > 0 {
            let mut axis = self.shape.len() - 1;
            loop {
                self.coord[axis] += 1;
                if (self.coord[axis] as usize) < self.shape[axis] {
                    break;
                }
                self.coord[axis] = 0;
                if axis == 0 {
                    break;
                }
                axis -= 1;
            }
        }

        Some(out)
    }
}

/// Build the destination schema for a materialized view snapshot: same
/// dtype/shape as `source`, with layout normalized for a fresh, independent
/// destination tensor and freshly computed `block_shape`/`strides`.
///
/// A `Sparse { axis }` hint is preserved only when `source` is currently an
/// identity/base view (`is_identity`) -- `transpose`/`slice`/`reshape` never
/// update `layout` when they mutate `shape`/`strides`, so a stale axis hint
/// surviving a transform could silently denote the wrong logical axis (or,
/// after a rank-reducing slice, be out of bounds). Resetting it to `None` on
/// any non-identity view sidesteps that silent-corruption risk; `None` is
/// always valid and defaults to axis 0 downstream.
pub(crate) fn snapshot_schema(source: &TensorSchema, is_identity: bool) -> FResult<TensorSchema> {
    let shape = source.shape().clone();
    let layout = match source.layout() {
        Layout::Dense => Layout::Dense,
        Layout::Sparse { axis } => Layout::Sparse {
            axis: if is_identity { *axis } else { None },
        },
    };
    let block_shape = crate::default_block_shape(&shape);
    let strides = contiguous_strides(&shape);
    TensorSchema::new(source.dtype(), shape, layout, block_shape, strides)
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
