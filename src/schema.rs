use number_general::{FloatType, IntType, NumberType, UIntType};
use smallvec::SmallVec;

use crate::{Error, PORTABLE_INLINE_RANK, Result as FResult};

/// Logical dimensions; inline storage is an optimization, not a rank limit.
pub type Shape = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

/// Unsigned logical row-major strides.
pub type Strides = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

// Reusable rank-sized logical coordinate scratch; public payloads remain Vec.
pub(crate) type Coord = SmallVec<[u64; PORTABLE_INLINE_RANK]>;

/// Rank-sized slice metadata. Explicit selections retain caller order and duplicates.
pub type Range = SmallVec<[AxisRange; PORTABLE_INLINE_RANK]>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AxisRange {
    /// Select one coordinate (geometric slicing removes this axis).
    At(u64),
    /// Select start..stop with a nonzero step; stop is exclusive.
    In(u64, u64, u64),
    /// Caller-owned indices, preserving order and duplicates in geometric slices.
    Of(Vec<u64>),
}

/// Maximum tensor values per storage block, independent of execution batches.
pub const MAX_BLOCK_CAPACITY: usize = 4096;

/// Sparse-index leaf size in bytes, as required by b_tree::Schema.
pub(crate) const SPARSE_INDEX_BLOCK_BYTES: usize = 4096;

/// Target retained bytes per sparse node, independent of tensor block capacity.
pub(crate) const SPARSE_NODE_MEMORY: usize = 16384;

/// Base tensor identity: dtype + fixed logical shape + fixed contiguous
/// strides. Held by `Tensor`, never mutated after creation -- the *current*
/// (possibly-transformed) shape/strides live on `TensorView` instead.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct TensorSchema {
    dtype: NumberType,
    shape: Shape,
    strides: Strides,
}

impl TensorSchema {
    pub fn new(dtype: NumberType, shape: Shape) -> FResult<Self> {
        let supported = match dtype {
            NumberType::UInt(UIntType::U8 | UIntType::U16 | UIntType::U32 | UIntType::U64)
            | NumberType::Int(IntType::I8 | IntType::I16 | IntType::I32 | IntType::I64)
            | NumberType::Float(FloatType::F32 | FloatType::F64) => true,
            #[cfg(feature = "complex")]
            NumberType::Complex(
                number_general::ComplexType::C32 | number_general::ComplexType::C64,
            ) => true,
            _ => false,
        };

        if !supported {
            return Err(Error::InvalidSchema(format!(
                "unsupported tensor dtype: {dtype}"
            )));
        }

        let strides = contiguous_strides(shape.as_slice())?;
        Ok(Self {
            dtype,
            shape,
            strides,
        })
    }

    pub fn dtype(&self) -> NumberType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }
}

/// Contiguous chunks within independently indexed dense regions.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct LinearBlocks {
    pub size: u64,
    pub region_len: u64,
    pub capacity: usize,
    pub chunks: u64,
}

impl LinearBlocks {
    pub fn new(shape: &[u64], layout: Layout, capacity: usize) -> FResult<Self> {
        validate_shape_dims(shape)?;
        let size = checked_product(shape)?;
        if capacity == 0 || capacity > MAX_BLOCK_CAPACITY {
            return Err(Error::InvalidLayout("invalid block capacity".into()));
        }

        let prefix = match layout {
            Layout::Dense => 0,
            Layout::Sparse { axis: None } => shape.len(),
            Layout::Sparse { axis: Some(axis) } if axis < shape.len() => axis + 1,
            Layout::Sparse { .. } => {
                return Err(Error::InvalidSchema("sparse axis out of bounds".into()));
            }
        };

        let region_len = checked_product(&shape[prefix..])?;
        let capacity = (capacity as u64).min(region_len) as usize;
        let chunks = region_len.div_ceil(capacity as u64);
        Ok(Self {
            size,
            region_len,
            capacity,
            chunks,
        })
    }

    pub fn count(&self) -> u64 {
        (self.size / self.region_len) * self.chunks
    }

    pub fn position(&self, flat: u64) -> BlockPosition {
        debug_assert!(flat < self.size);
        let region = flat / self.region_len;
        let local = flat % self.region_len;
        BlockPosition {
            block_id: region * self.chunks + local / self.capacity as u64,
            offset_in_block: (local % self.capacity as u64) as usize,
        }
    }

    pub fn range(&self, id: u64) -> FResult<std::ops::Range<u64>> {
        if id >= self.count() {
            return Err(Error::InvalidCoord("logical block out of bounds".into()));
        }

        let local = (id % self.chunks) * self.capacity as u64;
        let start = (id / self.chunks) * self.region_len + local;
        let len = (self.region_len - local).min(self.capacity as u64);
        Ok(start..start + len)
    }

    pub fn split(
        &self,
        start: i128,
        step: i128,
        len: usize,
        mut emit: impl FnMut(u64, usize, i64, usize, usize) -> FResult<()>,
    ) -> FResult<()> {
        if len == 0 {
            return Ok(());
        }

        let last = step
            .checked_mul((len - 1) as i128)
            .and_then(|n| start.checked_add(n))
            .ok_or_else(|| Error::InvalidCoord("mapped progression overflow".into()))?;

        if start.min(last) < 0 || start.max(last) >= self.size as i128 {
            return Err(Error::InvalidCoord(
                "mapped progression out of bounds".into(),
            ));
        }

        let mut done = 0;
        while done < len {
            let flat = (start + done as i128 * step) as u64;
            let position = self.position(flat);
            let range = self.range(position.block_id)?;
            let room = if step < 0 {
                flat - range.start
            } else {
                range.end - 1 - flat
            };

            let count = if len - done == 1 {
                1
            } else if step == 0 {
                len - done
            } else {
                ((room as i128 / step.abs()) + 1).min((len - done) as i128) as usize
            };

            let stride = if count == 1 {
                0
            } else {
                i64::try_from(step)
                    .map_err(|_| Error::InvalidCoord("storage stride exceeds i64".into()))?
            };

            emit(
                position.block_id,
                position.offset_in_block,
                stride,
                count,
                done,
            )?;

            done += count;
        }

        Ok(())
    }
}

/// Lazily cover a flat interval with disjoint row-major rectangles.
/// Retains rank-sized coordinates; no map of storage blocks is built.
pub(crate) fn flat_rectangles(
    schema: &TensorSchema,
    range: std::ops::Range<u64>,
) -> FResult<impl Iterator<Item = Vec<(u64, u64)>> + '_> {
    let shape = schema.shape();
    let strides = schema.strides();
    // TensorSchema validates a nonempty shape and its complete product.
    let size = shape[0] * strides[0];
    if range.start > range.end || range.end > size {
        return Err(Error::InvalidCoord("flat interval out of bounds".into()));
    }

    let mut start = range.start;
    let end = range.end;
    let mut coord = Coord::from_elem(0, shape.len());
    Ok(std::iter::from_fn(move || {
        if start == end {
            return None;
        }

        for ((value, stride), dim) in coord.iter_mut().zip(strides).zip(shape) {
            *value = (start / stride) % dim;
        }

        for (axis, &width) in strides.iter().enumerate() {
            if start.is_multiple_of(width) && end - start >= width {
                let count = (shape[axis] - coord[axis]).min((end - start) / width);
                let mut bounds: Vec<_> = coord.iter().map(|&c| (c, c + 1)).collect();
                bounds[axis].1 = coord[axis] + count;
                for trailing in axis + 1..shape.len() {
                    bounds[trailing] = (0, shape[trailing]);
                }

                start += count * width;
                return Some(bounds);
            }
        }

        unreachable!("last axis has unit stride")
    }))
}

/// Canonical row-major chunks; sparse prefixes never share a chunk.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct StorageSchema {
    pub(crate) layout: Layout,
    pub(crate) blocks: LinearBlocks,
}

impl StorageSchema {
    pub(crate) fn new(shape: &[u64], layout: Layout, capacity: usize) -> FResult<Self> {
        Ok(Self {
            layout,
            blocks: LinearBlocks::new(shape, layout, capacity)?,
        })
    }
}

/// Logical storage address: block ID and offset within its payload.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) struct BlockPosition {
    pub(crate) block_id: u64,
    pub(crate) offset_in_block: usize,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Layout {
    Dense,
    Sparse { axis: Option<usize> },
}

pub(crate) fn validate_shape_dims<T>(shape: &[T]) -> FResult<()>
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

pub fn contiguous_strides(shape: &[u64]) -> FResult<Strides> {
    validate_shape_dims(shape)?;

    checked_product(shape)?;
    let ndim = shape.len();
    let mut strides = Strides::from_elem(1, ndim);

    for i in (0..ndim).rev() {
        if i + 1 < ndim {
            strides[i] = strides[i + 1]
                .checked_mul(shape[i + 1])
                .ok_or_else(|| Error::InvalidSchema("Dimension overflows stride".to_string()))?;
        }
    }

    Ok(strides)
}

pub(crate) fn checked_product(shape: &[u64]) -> FResult<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::InvalidSchema("shape element count overflows u64".to_string()))
}

/// Row-major (C-order) coordinate walk over `shape` for bounded reads and copies.
pub struct RowMajorCoords {
    shape: Shape,
    coord: Coord,
    remaining: u64,
}

/// Stores only the shape and one current coordinate (both rank-sized).
/// Total size is a counter, never a coordinate allocation.
pub fn row_major_coords(shape: &[u64]) -> FResult<RowMajorCoords> {
    validate_shape_dims(shape)?;
    let remaining = checked_product(shape)?;

    Ok(RowMajorCoords {
        shape: shape.iter().copied().collect(),
        coord: Coord::from_elem(0, shape.len()),
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
                if self.coord[axis] < self.shape[axis] {
                    break;
                }

                self.coord[axis] = 0;
                if axis == 0 {
                    break;
                }

                axis -= 1;
            }
        }

        Some(out.into_vec())
    }
}

#[cfg(test)]
#[path = "../tests/unit/schema/tests.rs"]
mod tests;
