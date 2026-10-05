//! Typed, paged sparse payloads. Serialization belongs to file adapters.

use std::{cmp::Ordering, io, marker::PhantomData};

use b_table::{BTreeSchema, IndexSchema, Schema};
use number_general::{Complex, Float, Number};

use crate::TensorElement;

/// A native table cell, not a numerical cast or an erased storage value.
#[derive(Clone, Debug)]
pub enum SparseCell<T> {
    Key(u64),
    Payload(Vec<T>),
}

impl<T: TensorElement> get_size::GetSize for SparseCell<T> {
    fn get_heap_size(&self) -> usize {
        match self {
            Self::Key(_) => 0,
            Self::Payload(values) => values.capacity() * std::mem::size_of::<T>(),
        }
    }
}

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
            (Self::Payload(a), Self::Payload(b)) => {
                a.iter().copied().map(bits).cmp(b.iter().copied().map(bits))
            }
            (Self::Key(_), _) => Ordering::Less,
            _ => Ordering::Greater,
        }
    }
}

/// Caller-owned native payload node; each row has one logical block key and one dense payload.
pub type SparseNode<T> = b_table::Node<SparseCell<T>>;

#[derive(Clone, Debug)]
pub(crate) struct PayloadSchema<T> {
    columns: Vec<String>,
    block_len: usize,
    dtype: PhantomData<T>,
}

impl<T> PayloadSchema<T> {
    pub(crate) fn new(block_len: usize) -> Self {
        Self {
            columns: ["block", "payload"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            block_len,
            dtype: PhantomData,
        }
    }
}

impl<T> PartialEq for PayloadSchema<T> {
    fn eq(&self, b: &Self) -> bool {
        self.columns == b.columns && self.block_len == b.block_len
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
        2
    }

    fn order(&self) -> usize {
        let header = std::mem::size_of::<b_table::Node<SparseCell<T>>>();
        let row = std::mem::size_of::<Vec<SparseCell<T>>>()
            + 2 * std::mem::size_of::<SparseCell<T>>()
            + self.block_len * std::mem::size_of::<T>()
            + 16;

        // Native splitting and merging require an even order of at least four.
        ((crate::schema::SPARSE_NODE_MEMORY - header) / row / 2 * 2).max(4)
    }

    fn validate_key(&self, row: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if row.len() != 2
            || !matches!(row[0], SparseCell::Key(_))
            || !matches!(&row[1], SparseCell::Payload(values) if values.len() == self.block_len)
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
        &self.columns[..1]
    }

    fn values(&self) -> &[String] {
        &self.columns[1..]
    }

    fn primary(&self) -> &Self {
        self
    }

    fn auxiliary(&self) -> &[(String, Self)] {
        &[]
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if key.len() != 1 || !key.iter().all(|v| matches!(v, SparseCell::Key(_))) {
            return Err(invalid());
        }

        Ok(key)
    }

    fn validate_values(&self, v: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if v.len() != 1
            || !matches!(&v[0], SparseCell::Payload(values) if values.len() == self.block_len)
        {
            return Err(invalid());
        }

        Ok(v)
    }
}
