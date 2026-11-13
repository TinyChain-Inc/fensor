//! Bounded storage plans. Runs preserve request order without mapped coordinate lists.
//! Every run covers requested elements, so groups and the total run count
//! are bounded by `expression::MAX_BATCH_ELEMENTS`. Affine progressions validate
//! endpoints before any I/O; coordinate carries and block/key boundaries delimit
//! constant-stride runs.
//! Coalescing is descriptive only: it neither deduplicates outputs nor caches data.

use std::collections::BTreeMap;

use crate::mapping::CoordinateMap;
use crate::request::{BatchRequest, Progression, RequestKind, progressions};
use crate::schema::Coord;
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
    pub geometry: &'a crate::StorageGeometry,
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
        let shape = mapping
            .map(|m| m.shape())
            .unwrap_or(self.geometry.schema().shape());

        request.validate(shape)?;

        let affine = if matches!(request.kind(), RequestKind::Explicit(_))
            || (matches!(request.kind(), RequestKind::FlatRuns(_))
                && mapping.is_none_or(|m| {
                    m.is_identity(
                        self.geometry.schema().shape(),
                        self.geometry.schema().strides(),
                    )
                })) {
            None
        } else {
            match mapping {
                Some(mapping) => mapping.affine()?,
                None => Some((
                    0,
                    self.geometry
                        .schema()
                        .strides()
                        .iter()
                        .map(|s| *s as i128)
                        .collect(),
                )),
            }
        };

        let mut groups = LogicalGroups::new();

        let mut end = 0;
        let mut emit = |key, run: Run| {
            run.validate(self.geometry.block_len(key)?, request.len())?;
            if run.output != end {
                return Err(invalid());
            }

            end += run.len;
            push_run(groups.entry(key).or_default(), run);
            Ok(())
        };

        match (request.kind(), affine) {
            (RequestKind::FlatRuns(runs), _)
                if mapping.is_none_or(|m| {
                    m.is_identity(
                        self.geometry.schema().shape(),
                        self.geometry.schema().strides(),
                    )
                }) =>
            {
                for run in runs {
                    self.split(&mut emit, *run)?;
                }
            }
            (_, Some((offset, strides))) => {
                progressions(request, shape, offset, &strides, |progression| {
                    self.split(&mut emit, progression)
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
                        mapping.resolve_into(
                            &coord,
                            self.geometry.schema().shape(),
                            self.geometry.schema().strides(),
                            &mut mapped,
                        )?;

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
        self.geometry
            .block_position(coord)
            .expect("validated source coordinates")
    }

    fn split(
        &self,
        emit: &mut impl FnMut(u64, Run) -> Result<()>,
        progression: Progression,
    ) -> Result<()> {
        self.geometry.storage.blocks.split(
            progression.start,
            progression.step,
            progression.len,
            |id, offset, stride, len, done| {
                emit(
                    id,
                    Run {
                        output: progression.output + done,
                        offset,
                        stride,
                        len,
                    },
                )
            },
        )
    }
}

#[cfg(test)]
#[path = "../tests/unit/storage_read/tests.rs"]
mod tests;
