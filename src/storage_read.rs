//! Bounded storage plans. Runs preserve request order without mapped coordinate lists.
//! Every run covers requested elements, so groups and the total run count
//! are bounded by 4096. Affine progressions validate endpoints before any I/O;
//! coordinate carries and block/key boundaries delimit constant-stride runs.
//! Coalescing is descriptive only: it neither deduplicates outputs nor caches data.

use std::collections::BTreeMap;

use crate::mapping::CoordinateMap;
use crate::request::{BatchRequest, RequestKind, decode_flat, linear_segments};
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

// Each logical address owns only its requested runs. Sparse keys include the
// logical block ID; dense addresses use [0, block]. Physical aliases merge
// after one lookup per sparse key.
pub(crate) type LogicalGroups = BTreeMap<[u64; 2], Vec<Run>>;

pub(crate) struct StorageShape<'a> {
    pub shape: &'a [u64],
    pub strides: &'a [u64],
    pub block_shape: &'a [u64],
    pub block_strides: &'a [u64],
    pub grid_strides: &'a [u64],
    pub sparse_axis: Option<usize>,
}

struct Progression {
    output: usize,
    start: i128,
    step: i128,
    len: usize,
}

// Two rank-sized buffers, reused across all progression boundaries in a request.
struct Scratch {
    coord: Coord,
    delta: Coord,
}

// Shared address arithmetic for read planning and scalar/copy writes. Coordinates
// and schema cardinality are validated by callers. The block offset is below the
// validated storage capacity (at most 4096), so only that offset narrows to usize.
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

        let affine = if matches!(request.kind(), RequestKind::Explicit(_)) {
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

        if let Some((offset, strides)) = affine {
            progressions(request, shape, offset, &strides, |progression| {
                self.split(&mut emit, &mut scratch, progression)
            })?;
        } else {
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
                    self.key(coord, block),
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

    fn key(&self, coord: &[u64], block: u64) -> [u64; 2] {
        [self.sparse_axis.map_or(0, |axis| coord[axis]), block]
    }

    fn split(
        &self,
        emit: &mut impl FnMut([u64; 2], Run) -> Result<()>,
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

                if self.sparse_axis == Some(i) && delta[i] != 0 {
                    count = 1;
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
                self.key(coord, block),
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

// Invoke once per fastest-axis arithmetic progression, not per output element.
// Selected-axis scans use caller-bounded metadata and preserve duplicates/order.
fn progressions(
    request: &BatchRequest,
    shape: &[u64],
    offset: i128,
    strides: &[i128],
    mut emit: impl FnMut(Progression) -> Result<()>,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(invalid());
    }

    if shape.is_empty() {
        return emit(Progression {
            output: 0,
            start: offset,
            step: 0,
            len: request.len(),
        });
    }

    let rank = shape.len();
    let mut coord = Coord::from_elem(0, rank);
    let flat = |coord: &[u64]| {
        coord.iter().zip(strides).try_fold(offset, |n, (c, s)| {
            s.checked_mul(*c as i128)
                .and_then(|d| n.checked_add(d))
                .ok_or_else(invalid)
        })
    };
    match request.kind() {
        RequestKind::Linear { start } => {
            let varying = shape.iter().rposition(|d| *d > 1).unwrap_or(0);
            for (index, output, len) in linear_segments(*start, request.len(), shape[varying])? {
                decode_flat(index, shape, &mut coord)?;
                emit(Progression {
                    output,
                    start: flat(&coord)?,
                    step: strides[varying],
                    len,
                })?;
            }
        }
        RequestKind::Rectangles(rectangles) => {
            let mut output = 0;

            for rect in rectangles {
                if rect.len() == 0 {
                    continue;
                }

                let axes = rect.axes();
                // Trailing singleton axes are constants, not one-element rows.
                let varying = axes.iter().rposition(|axis| axis.len() > 1).unwrap_or(0);

                for (c, axis) in coord[varying + 1..].iter_mut().zip(&axes[varying + 1..]) {
                    *c = axis.at(0);
                }

                let last = &axes[varying];

                for prefix in 0..rect.len() / last.len() as usize {
                    let mut index = prefix;

                    for (c, axis) in coord[..varying].iter_mut().zip(&axes[..varying]).rev() {
                        *c = axis.at(index as u64 % axis.len());
                        index /= axis.len() as usize;
                    }

                    for (start, step, len) in last.segments() {
                        let len = len as usize;
                        coord[varying] = start;
                        emit(Progression {
                            output,
                            start: flat(&coord)?,
                            step: step.checked_mul(strides[varying]).ok_or_else(invalid)?,
                            len,
                        })?;
                        output += len;
                    }
                }
            }
        }
        RequestKind::Explicit(_) => unreachable!("explicit requests use the coordinate cursor"),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ha_ndarray::{axes, range, shape};

    use crate::AxisRange;

    use super::*;
    use crate::mapping::AxisContrib;
    use crate::request::{Axis, Cartesian};
    use crate::schema::contiguous_strides;

    fn compare(storage: &StorageShape<'_>, map: &CoordinateMap, request: BatchRequest) {
        let planned = storage.plan(&request, Some(map)).unwrap();
        let mut actual = vec![(0, [0, 0], 0); request.len()];

        for (key, runs) in planned {
            for run in runs {
                for i in 0..run.len {
                    actual[run.output + i] = (
                        key[1],
                        key,
                        (run.offset as i128 + i as i128 * run.stride as i128) as usize,
                    );
                }
            }
        }

        let expected: Vec<_> = request
            .coordinates(&map.shape)
            .unwrap()
            .iter()
            .map(|coord| {
                let base = map.resolve(coord, storage.shape, storage.strides).unwrap();
                let base: Vec<_> = base.into_iter().collect();
                let (id, offset) = storage.position(&base);
                (id, storage.key(&base, id), offset)
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn affine_plans_match_scalar_mapping_across_blocks_and_sparse_axes() {
        let shape = shape![3, 5, 7];
        let strides = contiguous_strides(&shape).unwrap();
        let identity = CoordinateMap::identity(shape.clone(), &strides);
        let maps = [
            identity.clone(),
            identity.clone().transpose(Some(axes![2, 0, 1])).unwrap(),
            identity.clone().flip(0).unwrap().flip(2).unwrap(),
            identity
                .clone()
                .slice(range![
                    AxisRange::In(0, 3, 2),
                    AxisRange::In(1, 5, 2),
                    AxisRange::In(0, 7, 3)
                ])
                .unwrap(),
            identity.clone().reshape(shape![15, 7]).unwrap(),
            identity
                .clone()
                .reshape(shape![105])
                .unwrap()
                .slice(range![AxisRange::In(0, 105, 11)])
                .unwrap(),
            identity
                .clone()
                .slice(range![
                    AxisRange::At(1),
                    AxisRange::In(0, 5, 1),
                    AxisRange::At(3)
                ])
                .unwrap()
                .broadcast(shape![4, 5])
                .unwrap(),
            identity
                .clone()
                .slice(range![
                    AxisRange::Of(vec![2, 0, 2]),
                    AxisRange::In(0, 5, 1),
                    AxisRange::In(0, 7, 1)
                ])
                .unwrap()
                .flip(1)
                .unwrap(),
        ];

        for block_shape in [vec![1, 1, 1], vec![2, 3, 4], vec![3, 5, 7]] {
            let block_strides = contiguous_strides(&block_shape).unwrap();
            let grid: Vec<_> = shape
                .iter()
                .zip(&block_shape)
                .map(|(d, b)| d.div_ceil(*b))
                .collect();
            let grid_strides = contiguous_strides(&grid).unwrap();

            for sparse_axis in [None, Some(0), Some(1), Some(2)] {
                let storage = StorageShape {
                    shape: &shape,
                    strides: &strides,
                    block_shape: &block_shape,
                    block_strides: &block_strides,
                    grid_strides: &grid_strides,
                    sparse_axis,
                };

                for map in &maps {
                    let size = map.shape.iter().product::<u64>();

                    for start in 0..size {
                        compare(
                            &storage,
                            map,
                            BatchRequest::linear(start, (size - start).min(13) as usize).unwrap(),
                        );
                    }

                    let axes = map
                        .shape
                        .iter()
                        .map(|d| Axis::Selected(vec![(*d - 1), 0, 0]))
                        .collect();
                    compare(
                        &storage,
                        map,
                        BatchRequest::rectangles(vec![Cartesian::new(axes).unwrap()]).unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn progressions_handle_arbitrary_signed_steps_and_carries() {
        let shape = [3, 5, 7];
        let strides = [35, 7, 1];
        let storage = StorageShape {
            shape: &shape,
            strides: &strides,
            block_shape: &[2, 3, 4],
            block_strides: &[12, 4, 1],
            grid_strides: &[4, 2, 1],
            sparse_axis: Some(1),
        };

        for start in 0..105 {
            for step in -104..=104 {
                let mut len = 1;

                while len < 16 && (0..105).contains(&(start + len * step)) {
                    len += 1;
                }

                let map = CoordinateMap {
                    base_offset: start,
                    axes: vec![AxisContrib::Stride(step)],
                    shape: shape![len as usize as u64],
                };
                compare(
                    &storage,
                    &map,
                    BatchRequest::linear(0, len as usize).unwrap(),
                );
            }
        }
    }

    #[tokio::test]
    async fn compact_runs_reduce_mapping_work_and_preserve_broadcasts() {
        let storage = StorageShape {
            shape: &[8, 8],
            strides: &[8, 1],
            block_shape: &[8, 8],
            block_strides: &[8, 1],
            grid_strides: &[1, 1],
            sparse_axis: None,
        };
        let identity = CoordinateMap::identity(shape![8, 8], &[8, 1]);

        for (map, expected_runs) in [
            (identity.clone(), 1),
            (identity.clone().transpose(None).unwrap(), 8),
            (identity.clone().flip(1).unwrap(), 8),
            (
                identity
                    .clone()
                    .slice(range![AxisRange::In(0, 8, 1), AxisRange::In(0, 1, 1)])
                    .unwrap()
                    .broadcast(shape![8, 512])
                    .unwrap(),
                8,
            ),
        ] {
            crate::read_metrics::CURRENT
                .scope(Default::default(), async {
                    let plan = storage
                        .plan(
                            &BatchRequest::linear(0, map.shape.iter().product::<u64>() as usize)
                                .unwrap(),
                            Some(&map),
                        )
                        .unwrap();
                    assert_eq!(plan.values().map(Vec::len).sum::<usize>(), expected_runs);
                    crate::read_metrics::CURRENT.with(|m| {
                        let m = m.borrow();
                        assert_eq!(m.coordinate_resolutions, 0);
                        assert_eq!(m.boundary_decodes, 8);
                    });
                })
                .await;
        }

        let huge = StorageShape {
            shape: &[1_000_000_000, 1_000_000_000],
            strides: &[1_000_000_000, 1],
            block_shape: &[1, 4096],
            block_strides: &[4096, 1],
            grid_strides: &[244141, 1],
            sparse_axis: None,
        };
        let request = BatchRequest::linear(999_999_999_999_999_968, 32).unwrap();
        assert!(huge.plan(&request, None).unwrap().len() <= 2);
    }

    #[tokio::test]
    async fn singleton_axes_do_not_expand_narrow_requests() {
        let storage = StorageShape {
            shape: &[4097, 1],
            strides: &[1, 1],
            block_shape: &[128, 1],
            block_strides: &[1, 1],
            grid_strides: &[1, 1],
            sparse_axis: None,
        };

        for request in [
            BatchRequest::linear(0, 4096).unwrap(),
            BatchRequest::rectangles(vec![
                Cartesian::new(vec![Axis::range(0, 4096), Axis::Selected(vec![0])]).unwrap(),
            ])
            .unwrap(),
        ] {
            crate::read_metrics::CURRENT
                .scope(Default::default(), async {
                    let runs = storage.plan(&request, None).unwrap();
                    assert_eq!(runs.len(), 32);
                    crate::read_metrics::CURRENT
                        .with(|m| assert_eq!(m.borrow().boundary_decodes, 32));
                })
                .await;
        }
    }

    #[test]
    fn malformed_progressions_and_runs_fail_closed() {
        let storage = StorageShape {
            shape: &[8, 8],
            strides: &[8, 1],
            block_shape: &[8, 8],
            block_strides: &[8, 1],
            grid_strides: &[1, 1],
            sparse_axis: None,
        };

        for (offset, stride) in [(64, 1), (-1, 1), (1, -1), (i128::MAX, i128::MAX)] {
            let map = CoordinateMap {
                base_offset: offset,
                axes: vec![AxisContrib::Stride(stride)],
                shape: shape![3],
            };
            assert!(
                storage
                    .plan(&BatchRequest::linear(0, 3).unwrap(), Some(&map))
                    .is_err()
            );
        }
        assert!(
            storage
                .plan(&BatchRequest::linear(63, 2).unwrap(), None)
                .is_err()
        );

        for run in [
            Run {
                output: 0,
                offset: 0,
                stride: -1,
                len: 2,
            },
            Run {
                output: usize::MAX,
                offset: 0,
                stride: 0,
                len: 2,
            },
            Run {
                output: 0,
                offset: 63,
                stride: 1,
                len: 2,
            },
        ] {
            let mut output = vec![0; 64];
            assert!(run.scatter(&[1; 64], &mut output).is_err());
            assert!(output.iter().all(|v| *v == 0));
        }
    }
}
