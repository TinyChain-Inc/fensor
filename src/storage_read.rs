//! Bounded storage plans. Runs preserve request order without mapped coordinate lists.
//! Every run covers requested elements, so groups and the total run count
//! are bounded by `expression::MAX_BATCH_ELEMENTS`. Affine progressions validate
//! endpoints before any I/O; coordinate carries and block/key boundaries delimit
//! constant-stride runs.
//! Coalescing is descriptive only: it neither deduplicates outputs nor caches data.

use std::collections::BTreeMap;

use crate::mapping::CoordinateMap;
use crate::request::{BatchRequest, Progression, RequestKind, decode_flat, progressions};
use crate::schema::{BlockPosition, Coord};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Run {
    pub output: usize,
    pub offset: usize,
    pub stride: i64,
    pub len: usize,
}

impl Run {
    pub fn validate(&self, block_len: usize, output_len: usize) -> Result<()> {
        let end = self.output.checked_add(self.len).ok_or_else(invalid)?;
        let last = self.offset as i128 + self.stride as i128 * self.len.saturating_sub(1) as i128;
        if self.len == 0
            || end > output_len
            || self.offset >= block_len
            || last < 0
            || last >= block_len as i128
        {
            return Err(invalid());
        }

        Ok(())
    }

    pub fn scatter<T: Copy>(&self, block: &[T], output: &mut [T]) -> Result<()> {
        self.validate(block.len(), output.len())?;
        let out = &mut output[self.output..self.output + self.len];
        if self.stride == 1 {
            out.copy_from_slice(&block[self.offset..self.offset + self.len]);
        } else {
            for (i, value) in out.iter_mut().enumerate() {
                *value = block[(self.offset as i128 + i as i128 * self.stride as i128) as usize];
            }
        }

        Ok(())
    }
}

// Each logical block owns only its requested runs.
pub(crate) type LogicalGroups = BTreeMap<u64, Vec<Run>>;

pub(crate) struct StorageShape<'a> {
    pub shape: &'a [u64],
    pub strides: &'a [u64],
    pub block_shape: &'a [u64],
    pub block_strides: &'a [u64],
    pub grid_strides: &'a [u64],
}

// Two rank-sized buffers, reused across all progression boundaries in a request.
struct Scratch {
    coord: Coord,
    delta: Coord,
}

// Shared address arithmetic for read planning and scalar/copy writes. Coordinates
// and schema cardinality are validated by callers. The block offset is below the
// validated MAX_BLOCK_CAPACITY, so only that offset narrows to usize.
pub(crate) fn block_position(
    coord: impl Iterator<Item = u64>,
    block_shape: &[u64],
    block_strides: &[u64],
    grid_strides: &[u64],
) -> BlockPosition {
    let mut block_id = 0;
    let mut offset_in_block = 0;

    for (i, c) in coord.enumerate() {
        block_id += (c / block_shape[i]) * grid_strides[i];
        offset_in_block += (c % block_shape[i]) * block_strides[i];
    }

    BlockPosition {
        block_id,
        offset_in_block: offset_in_block as usize,
    }
}

fn invalid() -> Error {
    Error::InvalidLayout("invalid storage run bounds or arithmetic overflow".into())
}

fn push_run(runs: &mut Vec<Run>, next: Run) {
    if let Some(previous) = runs.last_mut() {
        let a = previous;
        let b = &next;
        let stride = if a.len == 1 {
            b.offset as i64 - a.offset as i64
        } else {
            a.stride
        };

        if a.output + a.len == b.output
            && (b.len == 1 || b.stride == stride)
            && a.offset as i128 + stride as i128 * a.len as i128 == b.offset as i128
        {
            a.stride = stride;
            a.len += b.len;
            return;
        }
    }

    runs.push(next);
}

impl StorageShape<'_> {
    pub fn plan(
        &self,
        request: &BatchRequest,
        mapping: Option<&CoordinateMap>,
    ) -> Result<LogicalGroups> {
        let shape = mapping.map(|m| m.shape.as_slice()).unwrap_or(self.shape);
        request.validate(shape)?;

        let affine = if matches!(
            request.kind(),
            RequestKind::Explicit(_) | RequestKind::FlatRuns(_)
        ) {
            None
        } else {
            match mapping {
                Some(mapping) => mapping.affine()?,
                None => Some((0, self.strides.iter().map(|s| *s as i128).collect())),
            }
        };

        let mut groups = LogicalGroups::new();
        let block_len = self.block_shape.iter().product::<u64>() as usize;
        let mut end = 0;
        let mut emit = |key, run: Run| {
            run.validate(block_len, request.len())?;
            if run.output != end {
                return Err(invalid());
            }
            end += run.len;
            push_run(groups.entry(key).or_default(), run);
            Ok(())
        };
        // Rank-sized scratch is reused at progression/block boundaries.
        let mut scratch = Scratch {
            coord: Coord::from_elem(0, self.shape.len()),
            delta: Coord::from_elem(0, self.shape.len()),
        };

        match (request.kind(), affine) {
            (RequestKind::FlatRuns(runs), _)
                if mapping.is_none_or(|m| m.is_identity(self.shape, self.strides)) =>
            {
                for run in runs {
                    self.split(&mut emit, &mut scratch, *run)?;
                }
            }
            (_, Some((offset, strides))) => {
                progressions(request, shape, offset, &strides, |progression| {
                    self.split(&mut emit, &mut scratch, progression)
                })?;
            }
            _ => {
                let mut cursor = request.cursor(shape)?;
                let mut coord = Coord::new();
                let mut mapped = Coord::new();
                let mut output = 0;

                while cursor.next_into(&mut coord) {
                    #[cfg(test)]
                    crate::read_metrics::record(|m| m.coordinate_resolutions += 1);
                    let coord = if let Some(mapping) = mapping {
                        mapping.resolve_into(&coord, self.shape, self.strides, &mut mapped)?;
                        &mapped
                    } else {
                        &coord
                    };

                    let (block, offset) = self.position(coord);
                    emit(
                        block,
                        Run {
                            output,
                            offset,
                            stride: 0,
                            len: 1,
                        },
                    )?;
                    output += 1;
                }
            }
        }

        if end != request.len() {
            return Err(invalid());
        }

        #[cfg(test)]
        crate::read_metrics::record(|m| {
            m.runs += groups.values().map(Vec::len).sum::<usize>();
            m.requested += request.len();
        });
        Ok(groups)
    }

    fn position(&self, coord: &[u64]) -> (u64, usize) {
        let position = block_position(
            coord.iter().copied(),
            self.block_shape,
            self.block_strides,
            self.grid_strides,
        );
        (position.block_id, position.offset_in_block)
    }

    fn split(
        &self,
        emit: &mut impl FnMut(u64, Run) -> Result<()>,
        scratch: &mut Scratch,
        progression: Progression,
    ) -> Result<()> {
        let Progression {
            output,
            start,
            step,
            len,
        } = progression;
        let Scratch { coord, delta } = scratch;
        if len == 0 {
            return Ok(());
        }

        let last = step
            .checked_mul((len - 1) as i128)
            .and_then(|d| start.checked_add(d))
            .ok_or_else(invalid)?;
        let size = self
            .shape
            .iter()
            .try_fold(1i128, |n, d| n.checked_mul(*d as i128))
            .ok_or_else(invalid)?;
        if start.min(last) < 0 || start.max(last) >= size {
            return Err(Error::InvalidCoord(
                "mapped progression out of bounds".into(),
            ));
        }

        let magnitude = if len == 1 {
            0
        } else {
            u64::try_from(step.checked_abs().ok_or_else(invalid)?).map_err(|_| invalid())?
        };
        // Mixed-radix step digits are constant until a carry/borrow. The block
        // boundary limits also prevent coordinate carries inside an emitted run.
        for ((delta, stride), dim) in delta.iter_mut().zip(self.strides).zip(self.shape) {
            *delta = (magnitude / stride) % dim;
        }

        let mut done = 0;

        while done < len {
            // Endpoints were checked in widened arithmetic. All intermediate
            // offsets lie between them; native division is safe at boundaries.
            let flat = u64::try_from(start + done as i128 * step).map_err(|_| invalid())?;

            #[cfg(test)]
            crate::read_metrics::record(|m| m.boundary_decodes += 1);
            let mut count = len - done;
            let mut stride = 0i128;

            decode_flat(flat, self.shape, coord)?;

            for (i, c) in coord.iter().enumerate() {
                let low = *c / self.block_shape[i] * self.block_shape[i];
                let high = low + (self.block_shape[i] - 1).min(self.shape[i] - 1 - low);
                let room = if step < 0 { *c - low } else { high - *c };
                if let Some(steps) = room.checked_div(delta[i]) {
                    count = (count as u64).min(steps.saturating_add(1)) as usize;
                }

                stride += delta[i] as i128 * self.block_strides[i] as i128 * step.signum();
            }

            let (block, offset) = self.position(coord);
            let stride = if count == 1 {
                0
            } else {
                i64::try_from(stride).map_err(|_| invalid())?
            };
            emit(
                block,
                Run {
                    output: output + done,
                    offset,
                    stride,
                    len: count,
                },
            )?;
            done += count;
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/storage_read/tests.rs"]
mod tests;
