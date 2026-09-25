//! Bounded storage plans. Runs preserve request order without mapped coordinate lists.
//! Every emitted run covers at least one requested element, so both collections
//! are bounded by 4096. Affine progressions validate endpoints before any I/O;
//! coordinate carries and block/key boundaries delimit constant-stride runs.
//! Coalescing is descriptive only: it neither deduplicates outputs nor caches data.

use crate::expression::MAX_BATCH_ELEMENTS;
use crate::mapping::CoordinateMap;
use crate::request::{Axis, BatchRequest, RequestKind};
use crate::schema::BlockPosition;
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

#[derive(Debug)]
pub(crate) struct PlannedRun {
    pub block: u64,
    pub key: Option<[u64; 2]>,
    pub run: Run,
}

pub(crate) struct StorageShape<'a> {
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub block_shape: &'a [usize],
    pub block_strides: &'a [usize],
    pub grid_strides: &'a [usize],
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
    coord: Vec<usize>,
    delta: Vec<usize>,
}

// Shared address arithmetic for read planning and scalar/copy writes.
pub(crate) fn block_position(
    coord: impl Iterator<Item = usize>,
    block_shape: &[usize],
    block_strides: &[usize],
    grid_strides: &[usize],
) -> BlockPosition {
    let mut block_id = 0;
    let mut offset_in_block = 0;

    for (i, c) in coord.enumerate() {
        block_id += (c / block_shape[i]) as u64 * grid_strides[i] as u64;
        offset_in_block += (c % block_shape[i]) * block_strides[i];
    }

    BlockPosition {
        block_id,
        offset_in_block,
    }
}

fn invalid() -> Error {
    Error::InvalidLayout("invalid storage run bounds or arithmetic overflow".into())
}

fn push_run(runs: &mut Vec<PlannedRun>, next: PlannedRun) {
    if let Some(previous) = runs.last_mut() {
        let a = &mut previous.run;
        let b = &next.run;
        let stride = if a.len == 1 {
            b.offset as i64 - a.offset as i64
        } else {
            a.stride
        };

        if previous.block == next.block
            && previous.key == next.key
            && a.output + a.len == b.output
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
    ) -> Result<Vec<PlannedRun>> {
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

        let mut runs = Vec::new();
        // Rank-sized scratch is reused at progression/block boundaries.
        let mut scratch = Scratch {
            coord: vec![0; self.shape.len()],
            delta: vec![0; self.shape.len()],
        };

        if let Some((offset, strides)) = affine {
            progressions(request, shape, offset, &strides, |progression| {
                self.split(&mut runs, &mut scratch, progression)
            })?;
        } else {
            let mut cursor = request.cursor(shape)?;
            let mut coord = Vec::new();
            let mut mapped = Vec::new();
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

                for (out, c) in scratch.coord.iter_mut().zip(coord) {
                    *out = *c as usize;
                }

                let (block, offset) = self.position(&scratch.coord);
                push_run(
                    &mut runs,
                    PlannedRun {
                        block,
                        key: self.key(&scratch.coord, block),
                        run: Run {
                            output,
                            offset,
                            stride: 0,
                            len: 1,
                        },
                    },
                );
                output += 1;
            }
        }

        let block_len = self.block_shape.iter().product();
        let mut end = 0;

        for planned in &runs {
            planned.run.validate(block_len, request.len())?;
            if planned.run.output != end {
                return Err(invalid());
            }
            end += planned.run.len;
        }

        if end != request.len() || runs.len() > MAX_BATCH_ELEMENTS {
            return Err(invalid());
        }

        #[cfg(test)]
        crate::read_metrics::record(|m| {
            m.runs += runs.len();
            m.requested += request.len();
        });
        Ok(runs)
    }

    fn position(&self, coord: &[usize]) -> (u64, usize) {
        let position = block_position(
            coord.iter().copied(),
            self.block_shape,
            self.block_strides,
            self.grid_strides,
        );
        (position.block_id, position.offset_in_block)
    }

    fn key(&self, coord: &[usize], block: u64) -> Option<[u64; 2]> {
        self.sparse_axis.map(|axis| [coord[axis] as u64, block])
    }

    fn split(
        &self,
        runs: &mut Vec<PlannedRun>,
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
            usize::try_from(step.checked_abs().ok_or_else(invalid)?).map_err(|_| invalid())?
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
            let flat = usize::try_from(start + done as i128 * step).map_err(|_| invalid())?;

            #[cfg(test)]
            crate::read_metrics::record(|m| m.boundary_decodes += 1);
            let mut count = len - done;
            let mut stride = 0i128;

            for (i, c) in coord.iter_mut().enumerate() {
                *c = (flat / self.strides[i]) % self.shape[i];
                let low = *c / self.block_shape[i] * self.block_shape[i];
                let high = low + (self.block_shape[i] - 1).min(self.shape[i] - 1 - low);
                let room = if step < 0 { *c - low } else { high - *c };
                if let Some(steps) = room.checked_div(delta[i]) {
                    count = count.min(steps + 1);
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
            push_run(
                runs,
                PlannedRun {
                    block,
                    key: self.key(coord, block),
                    run: Run {
                        output: output + done,
                        offset,
                        stride,
                        len: count,
                    },
                },
            );
            done += count;
        }

        Ok(())
    }
}

// Invoke once per fastest-axis arithmetic progression, not per output element.
// Selected-axis scans use caller-bounded metadata and preserve duplicates/order.
fn progressions(
    request: &BatchRequest,
    shape: &[usize],
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
    let mut coord = vec![0u64; rank];
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
            let mut done = 0;

            while done < request.len() {
                let mut index = start + done;

                for (c, d) in coord.iter_mut().zip(shape).rev() {
                    *c = (index % d) as u64;
                    index /= d;
                }

                let len = (request.len() - done).min(shape[varying] - coord[varying] as usize);
                emit(Progression {
                    output: done,
                    start: flat(&coord)?,
                    step: strides[varying],
                    len,
                })?;
                done += len;
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
                let segments = axis_segments(last);

                for prefix in 0..rect.len() / last.len() {
                    let mut index = prefix;

                    for (c, axis) in coord[..varying].iter_mut().zip(&axes[..varying]).rev() {
                        *c = axis.at(index % axis.len());
                        index /= axis.len();
                    }

                    for &(start, step, len) in &segments {
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

fn axis_segments(axis: &Axis) -> Vec<(u64, i128, usize)> {
    match axis {
        Axis::Span { start, step, len } => vec![(*start as u64, *step as i128, *len)],
        Axis::Selected(values) => {
            let mut segments = Vec::new();
            let mut i = 0;

            while i < values.len() {
                let step = values
                    .get(i + 1)
                    .map(|v| *v as i128 - values[i] as i128)
                    .unwrap_or(0);
                let mut end = i + 1;

                while end < values.len() && values[end] as i128 - values[end - 1] as i128 == step {
                    end += 1;
                }
                segments.push((values[i], step, end - i));
                i = end;
            }
            segments
        }
    }
}

#[cfg(test)]
mod tests {
    use ha_ndarray::{AxisRange, axes, range, shape};

    use super::*;
    use crate::mapping::AxisContrib;
    use crate::request::Cartesian;
    use crate::schema::contiguous_strides;

    fn compare(storage: &StorageShape<'_>, map: &CoordinateMap, request: BatchRequest) {
        let planned = storage.plan(&request, Some(map)).unwrap();
        let mut actual = Vec::new();

        for p in planned {
            for i in 0..p.run.len {
                actual.push((
                    p.block,
                    p.key,
                    (p.run.offset as i128 + i as i128 * p.run.stride as i128) as usize,
                ));
            }
        }

        let expected: Vec<_> = request
            .coordinates(&map.shape)
            .unwrap()
            .iter()
            .map(|coord| {
                let base = map.resolve(coord, storage.shape, storage.strides).unwrap();
                let base: Vec<_> = base.into_iter().map(|c| c as usize).collect();
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
                    AxisRange::Of(vec![2, 0, 2].into()),
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
                    let size = map.shape.iter().product::<usize>();

                    for start in 0..size {
                        compare(
                            &storage,
                            map,
                            BatchRequest::linear(start, (size - start).min(13)).unwrap(),
                        );
                    }

                    let axes = map
                        .shape
                        .iter()
                        .map(|d| Axis::Selected(vec![(*d - 1) as u64, 0, 0]))
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
                    axes: vec![AxisContrib::Stride(step)].into(),
                    shape: shape![len as usize],
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
                            &BatchRequest::linear(0, map.shape.iter().product()).unwrap(),
                            Some(&map),
                        )
                        .unwrap();
                    assert_eq!(plan.len(), expected_runs);
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

        for (offset, stride) in [(64, 1), (-1, 1), (1, -1), (i64::MAX, i64::MAX)] {
            let map = CoordinateMap {
                base_offset: offset,
                axes: vec![AxisContrib::Stride(stride)].into(),
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
