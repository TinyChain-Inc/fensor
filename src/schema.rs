use std::io;

use b_table::{IndexSchema, Schema};
use ha_ndarray::{Shape, Strides};

use crate::{Error, Result as FResult};

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum DType {
    F32,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Layout {
    Dense,
    Sparse { axis: Option<usize> },
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorSchema {
    pub dtype: DType,
    pub shape: Shape,
    pub layout: Layout,
    pub block_shape: Shape,
    pub strides: Strides,
}

impl TensorSchema {
    pub fn new(
        dtype: DType,
        shape: Shape,
        layout: Layout,
        block_shape: Shape,
        strides: Strides,
    ) -> FResult<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSchema(
                "tensor shape cannot be empty".to_string(),
            ));
        }

        if block_shape.len() != shape.len() || block_shape.iter().any(|dim| *dim == 0) {
            return Err(Error::InvalidSchema(
                "block_shape must be non-zero and match tensor rank".to_string(),
            ));
        }

        if strides.len() != shape.len() {
            return Err(Error::InvalidSchema(
                "strides rank must match tensor shape rank".to_string(),
            ));
        }

        if let Layout::Sparse { axis: Some(axis) } = layout {
            if axis >= shape.len() {
                return Err(Error::InvalidSchema(
                    "sparse axis hint out of bounds".to_string(),
                ));
            }
        }

        Ok(Self {
            dtype,
            shape,
            layout,
            block_shape,
            strides,
        })
    }

    pub fn dense(shape: Shape, block_shape: Shape) -> FResult<Self> {
        let strides = contiguous_strides(&shape);
        Self::new(DType::F32, shape, Layout::Dense, block_shape, strides)
    }

    pub fn sparse(shape: Shape, block_shape: Shape, axis: Option<usize>) -> FResult<Self> {
        let strides = contiguous_strides(&shape);
        Self::new(
            DType::F32,
            shape,
            Layout::Sparse { axis },
            block_shape,
            strides,
        )
    }

    pub fn block_len(&self) -> usize {
        self.block_shape.iter().product()
    }

    pub fn validate_coord(&self, coord: &[u64]) -> FResult<()> {
        if coord.len() != self.shape.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        for (i, (c, dim)) in coord.iter().zip(self.shape.iter()).enumerate() {
            if *c as usize >= *dim {
                return Err(Error::InvalidCoord(format!(
                    "coordinate at axis {i} is out of bounds"
                )));
            }
        }

        Ok(())
    }
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
