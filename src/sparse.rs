//! Typed sparse tree rows and their bounded, format-neutral codec.

use std::cmp::Ordering;
use std::io;
use std::marker::PhantomData;

use b_tree::Schema;
use collate::Collate;
use number_general::{Float, Number, NumberCollator, NumberType};

use crate::{Layout, TensorElement};

mod codec;

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

impl<T: TensorElement> PartialEq for SparseCell<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Key(a), Self::Key(b)) => a == b,
            (Self::Payload(a), Self::Payload(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(&a, &b)| Into::<Number>::into(a) == Into::<Number>::into(b))
            }
            _ => false,
        }
    }
}

impl<T: TensorElement> Eq for SparseCell<T> {}

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
                let collator = NumberCollator::default();
                for (&a, &b) in a.iter().zip(b) {
                    let order = collator.cmp(&a.into(), &b.into());
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
#[derive(Clone, Debug)]
pub struct SparseNode<T>(pub b_tree::Node<Vec<Vec<SparseCell<T>>>>);

impl<T> From<b_tree::Node<Vec<Vec<SparseCell<T>>>>> for SparseNode<T> {
    fn from(node: b_tree::Node<Vec<Vec<SparseCell<T>>>>) -> Self {
        Self(node)
    }
}

impl<T: TensorElement> get_size::GetSize for SparseNode<T> {
    fn get_heap_size(&self) -> usize {
        self.0.get_heap_size()
    }
}

// Sparse blocks use one native index keyed by their logical block ID.
#[derive(Clone, Debug)]
pub(crate) struct PayloadSchema<T> {
    geometry: crate::StorageGeometry,
    dtype: PhantomData<T>,
}

impl<T> PayloadSchema<T> {
    pub(crate) fn new(geometry: crate::StorageGeometry) -> Self {
        Self {
            geometry,
            dtype: PhantomData,
        }
    }

    pub(crate) fn geometry(&self) -> &crate::StorageGeometry {
        &self.geometry
    }
}

impl<T> PartialEq for PayloadSchema<T> {
    fn eq(&self, b: &Self) -> bool {
        self.geometry == b.geometry
    }
}

impl<T> Eq for PayloadSchema<T> {}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid sparse payload row")
}

pub(crate) fn has_payload<T: TensorElement>(layout: Layout, values: &[T]) -> bool {
    if values.iter().any(|value| *value != T::ZERO) {
        return true;
    }

    // Absence decodes to canonical zero. Dense transactional blocks with a
    // negative zero component still need a payload row.
    layout == Layout::Dense
        && matches!(T::dtype(), NumberType::Float(_) | NumberType::Complex(_))
        && values.iter().any(|&value| match value.into() {
            Number::Float(value) => f64::from(value).is_sign_negative(),
            Number::Complex(value) => {
                let (re, im): (Float, Float) = value.into();
                f64::from(re).is_sign_negative() || f64::from(im).is_sign_negative()
            }
            _ => false,
        })
}

impl<T: TensorElement> Schema for PayloadSchema<T> {
    type Error = io::Error;

    type Value = SparseCell<T>;

    fn block_size(&self) -> usize {
        crate::schema::SPARSE_INDEX_BLOCK_BYTES
    }

    fn len(&self) -> usize {
        2
    }

    fn order(&self) -> usize {
        let header = std::mem::size_of::<b_tree::Node<Vec<Vec<SparseCell<T>>>>>();
        let row = std::mem::size_of::<Vec<SparseCell<T>>>()
            + 2 * std::mem::size_of::<SparseCell<T>>()
            + self.geometry.block_capacity() * std::mem::size_of::<T>()
            + 16;

        // Choose an even B+ tree order of at least four.
        ((crate::schema::SPARSE_NODE_MEMORY - header) / row / 2 * 2).max(4)
    }

    fn validate_prefix(&self, prefix: &[Self::Value]) -> io::Result<()> {
        match prefix {
            [SparseCell::Key(id)] => self.geometry.validate_id(*id).map_err(io::Error::other),
            [SparseCell::Key(id), SparseCell::Payload(values)] => self
                .geometry
                .validate_block(*id, values)
                .map_err(io::Error::other),
            _ => Err(invalid()),
        }
    }

    fn validate_key(&self, row: &[Self::Value]) -> io::Result<()> {
        let [SparseCell::Key(id), SparseCell::Payload(values)] = row else {
            return Err(invalid());
        };

        self.geometry
            .validate_block(*id, values)
            .map_err(io::Error::other)?;
        if !has_payload(self.geometry.layout(), values) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty sparse payload",
            ));
        }

        Ok(())
    }
}
