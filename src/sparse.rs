//! Typed, paged sparse payloads. Serialization belongs to file adapters.
use crate::TensorElement;
use b_table::{BTreeSchema, IndexSchema, Schema};
use number_general::{Complex, Float, Number};
use std::{cmp::Ordering, io, marker::PhantomData};

/// A native table cell, not a numerical cast or an erased storage value.
#[derive(Clone, Copy, Debug)]
pub enum SparseCell<T> {
    Key(u64),
    Value(T),
}
// Tensor elements and the discriminator are stored entirely inline.
impl<T: TensorElement> get_size::GetSize for SparseCell<T> {}

impl<T> Default for SparseCell<T> {
    fn default() -> Self {
        Self::Key(0)
    }
}
fn bits<T: TensorElement>(v: T) -> [u64; 2] {
    match v.into() {
        Number::Float(Float::F32(v)) => [v.to_bits() as u64, 0],
        Number::Float(Float::F64(v)) => [v.to_bits(), 0],
        Number::Complex(Complex::C32(v)) => [v.re.to_bits() as u64, v.im.to_bits() as u64],
        Number::Complex(Complex::C64(v)) => [v.re.to_bits(), v.im.to_bits()],
        Number::Int(v) => [i64::from(v) as u64, 0],
        Number::UInt(v) => [u64::from(v), 0],
        Number::Bool(v) => [u64::from(bool::from(v)), 0],
    }
}
impl<T: TensorElement> PartialEq for SparseCell<T> {
    fn eq(&self, b: &Self) -> bool {
        self.cmp(b) == Ordering::Equal
    }
}
impl<T: TensorElement> Eq for SparseCell<T> {}
impl<T: TensorElement> PartialOrd for SparseCell<T> {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl<T: TensorElement> Ord for SparseCell<T> {
    fn cmp(&self, b: &Self) -> Ordering {
        match (self, b) {
            (Self::Key(a), Self::Key(b)) => a.cmp(b),
            (Self::Value(a), Self::Value(b)) => bits(*a).cmp(&bits(*b)),
            (Self::Key(_), _) => Ordering::Less,
            _ => Ordering::Greater,
        }
    }
}
/// Caller-owned native payload node; each row has two keys and one typed value.
pub type SparseNode<T> = b_table::Node<SparseCell<T>>;

#[derive(Clone, Debug)]
pub(crate) struct PayloadSchema<T> {
    columns: Vec<String>,
    dtype: PhantomData<T>,
}
impl<T> Default for PayloadSchema<T> {
    fn default() -> Self {
        Self {
            columns: ["block", "offset", "value"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            dtype: PhantomData,
        }
    }
}
impl<T> PartialEq for PayloadSchema<T> {
    fn eq(&self, b: &Self) -> bool {
        self.columns == b.columns
    }
}
impl<T> Eq for PayloadSchema<T> {}
fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid sparse payload row")
}
impl<T: TensorElement> BTreeSchema for PayloadSchema<T> {
    type Error = io::Error;
    type Value = SparseCell<T>;
    fn block_size(&self) -> usize {
        crate::schema::SPARSE_INDEX_BLOCK_BYTES
    }
    fn len(&self) -> usize {
        3
    }
    fn order(&self) -> usize {
        crate::schema::sparse_node_order::<SparseCell<T>>(3)
    }
    fn validate_key(&self, row: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if row.len() != 3
            || !row[..2].iter().all(|v| matches!(v, SparseCell::Key(_)))
            || !matches!(row[2], SparseCell::Value(_))
        {
            return Err(invalid());
        }
        Ok(row)
    }
}
impl<T: TensorElement> IndexSchema for PayloadSchema<T> {
    type Id = String;
    fn columns(&self) -> &[String] {
        &self.columns
    }
}
impl<T: TensorElement> Schema for PayloadSchema<T> {
    type Id = String;
    type Error = io::Error;
    type Value = SparseCell<T>;
    type Index = Self;
    fn key(&self) -> &[String] {
        &self.columns[..2]
    }
    fn values(&self) -> &[String] {
        &self.columns[2..]
    }
    fn primary(&self) -> &Self {
        self
    }
    fn auxiliary(&self) -> &[(String, Self)] {
        &[]
    }
    fn validate_key(&self, key: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if key.len() != 2 || !key.iter().all(|v| matches!(v, SparseCell::Key(_))) {
            return Err(invalid());
        }
        Ok(key)
    }
    fn validate_values(&self, v: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if v.len() != 1 || !matches!(v[0], SparseCell::Value(_)) {
            return Err(invalid());
        }
        Ok(v)
    }
}
