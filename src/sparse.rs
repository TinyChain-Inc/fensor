//! Typed, paged sparse payloads. Serialization belongs to file adapters.

use std::{cmp::Ordering, io, marker::PhantomData};

use b_table::{BTreeSchema, IndexSchema, Schema};
use collate::Collate;
use number_general::{Float, Number, NumberInstance};

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

// NaNs form one semantic equality class after all non-NaN values. Signed zeros
// compare equally. Replacement still writes the caller's supplied representation.
fn float_cmp(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

fn number_cmp(a: Number, b: Number) -> Ordering {
    match (a, b) {
        (Number::Float(a), Number::Float(b)) => float_cmp(a.into(), b.into()),
        (Number::Complex(a), Number::Complex(b)) => {
            let (ar, ai): (Float, Float) = a.into();
            let (br, bi): (Float, Float) = b.into();

            float_cmp(
                f64::from(ar).hypot(ai.into()),
                f64::from(br).hypot(bi.into()),
            )
        }
        (Number::Int(a), Number::Int(b)) => a.cmp(&b),
        (Number::UInt(a), Number::UInt(b)) => a.cmp(&b),
        (Number::Bool(a), Number::Bool(b)) => bool::from(a).cmp(&bool::from(b)),
        (a, b) => a.class().cmp(&b.class()),
    }
}

fn same_number(a: Number, b: Number) -> bool {
    match (a, b) {
        (Number::Complex(a), Number::Complex(b)) => {
            let (ar, ai): (Float, Float) = a.into();
            let (br, bi): (Float, Float) = b.into();

            float_cmp(ar.into(), br.into()).is_eq() && float_cmp(ai.into(), bi.into()).is_eq()
        }
        (a, b) => number_cmp(a, b).is_eq(),
    }
}

impl<T: TensorElement> PartialEq for SparseCell<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Key(a), Self::Key(b)) => a == b,
            (Self::Payload(a), Self::Payload(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(&a, &b)| same_number(a.into(), b.into()))
            }
            _ => false,
        }
    }
}

impl<T: TensorElement> Eq for SparseCell<T> {}

// Collation deliberately groups unequal complex values with the same norm.
// It therefore belongs to Collate, not SparseCell's Eq/Ord implementation.
#[derive(Clone, Copy, Default)]
pub(crate) struct SparseCollator<T>(PhantomData<T>);

impl<T> PartialEq for SparseCollator<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> Eq for SparseCollator<T> {}

impl<T: TensorElement> Collate for SparseCollator<T> {
    type Value = SparseCell<T>;

    fn cmp(&self, a: &Self::Value, b: &Self::Value) -> Ordering {
        match (a, b) {
            (SparseCell::Key(a), SparseCell::Key(b)) => a.cmp(b),
            (SparseCell::Payload(a), SparseCell::Payload(b)) => {
                for (&a, &b) in a.iter().zip(b) {
                    let order = number_cmp(a.into(), b.into());
                    if !order.is_eq() {
                        return order;
                    }
                }

                a.len().cmp(&b.len())
            }
            (SparseCell::Key(_), _) => Ordering::Less,
            _ => Ordering::Greater,
        }
    }
}

/// Caller-owned native payload node. Leaves contain a block key and dense payload;
/// internal separators may retain only the block key.
pub type SparseNode<T> = b_table::Node<SparseCell<T>>;

// One schema serves both the table and its primary index; there are no auxiliary indexes.
#[derive(Clone, Debug)]
pub(crate) struct PayloadSchema<T> {
    block_len: usize,
    dtype: PhantomData<T>,
}

impl<T> PayloadSchema<T> {
    pub(crate) fn new(block_len: usize) -> Self {
        Self {
            block_len,
            dtype: PhantomData,
        }
    }
}

impl<T> PartialEq for PayloadSchema<T> {
    fn eq(&self, b: &Self) -> bool {
        self.block_len == b.block_len
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

        // Packing policy: choose an even order of at least four for sparse pages.
        ((crate::schema::SPARSE_NODE_MEMORY - header) / row / 2 * 2).max(4)
    }

    fn validate_prefix(&self, prefix: &[Self::Value]) -> io::Result<()> {
        match prefix {
            [SparseCell::Key(_)] => Ok(()),
            [SparseCell::Key(_), SparseCell::Payload(values)] if values.len() == self.block_len => {
                Ok(())
            }
            _ => Err(invalid()),
        }
    }

    fn validate_key(&self, row: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if row.len() != 2 {
            return Err(invalid());
        }
        self.validate_prefix(&row)?;

        Ok(row)
    }
}

impl<T: TensorElement> IndexSchema for PayloadSchema<T> {
    type Id = &'static str;

    fn columns(&self) -> &[Self::Id] {
        &["block", "payload"]
    }
}

impl<T: TensorElement> Schema for PayloadSchema<T> {
    type Id = &'static str;
    type Error = io::Error;
    type Value = SparseCell<T>;
    type Index = Self;

    fn key(&self) -> &[Self::Id] {
        &["block"]
    }

    fn values(&self) -> &[Self::Id] {
        &["payload"]
    }

    fn primary(&self) -> &Self {
        self
    }

    fn auxiliary(&self) -> &[(String, Self)] {
        &[]
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if !matches!(key.as_slice(), [SparseCell::Key(_)]) {
            return Err(invalid());
        }

        Ok(key)
    }

    fn validate_values(&self, v: Vec<Self::Value>) -> io::Result<Vec<Self::Value>> {
        if !matches!(v.as_slice(), [SparseCell::Payload(values)] if values.len() == self.block_len)
        {
            return Err(invalid());
        }

        Ok(v)
    }
}
