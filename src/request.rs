//! Compact, bounded execution requests. Coordinates are expanded only by consumers
//! which need explicit payloads; cursors reuse one rank-sized scratch buffer.

use std::borrow::Borrow;
use std::sync::Arc;

use crate::expression::MAX_BATCH_ELEMENTS;
use crate::mapping::CoordinateMap;
use crate::schema::Coord;
use crate::{Error, Result, schema, validate};

/// Decode one logical position into reusable rank-sized scratch. No geometry
/// narrows to machine-sized indexing, including dimensions above i64::MAX.
pub(crate) fn decode_flat(mut flat: u64, shape: &[u64], out: &mut Coord) -> Result<()> {
    out.resize(shape.len(), 0);

    for (dim, value) in shape.iter().zip(out.iter_mut()).rev() {
        if *dim == 0 {
            return Err(Error::InvalidSchema(
                "zero dimension in coordinate decoding".into(),
            ));
        }
        *value = flat % dim;
        flat /= dim;
    }

    if flat != 0 {
        return Err(Error::InvalidCoord("flat coordinate out of bounds".into()));
    }

    Ok(())
}

/// Lazy row intersections, retaining flat position, output offset, and length.
/// Callers choose row width: storage can fold trailing singleton axes, whereas
/// matrix planning retains matrix rows. The request has already been bounded.
pub(crate) fn linear_segments(
    start: u64,
    len: usize,
    width: u64,
) -> Result<impl Iterator<Item = (u64, usize, usize)>> {
    if width == 0 || start.checked_add(len as u64).is_none() {
        return Err(Error::InvalidCoord("invalid linear segment bounds".into()));
    }

    let mut done = 0;
    Ok(std::iter::from_fn(move || {
        if done == len {
            return None;
        }

        let flat = start + done as u64;
        let count = ((len - done) as u64).min(width - flat % width) as usize;
        let segment = (flat, done, count);
        done += count;
        Some(segment)
    }))
}

#[derive(Clone, Debug)]
pub(crate) enum Axis {
    Span { start: u64, step: u64, len: u64 },
    Selected(Vec<u64>),
}

impl Axis {
    pub(crate) fn range(start: u64, len: u64) -> Self {
        Self::Span {
            start,
            step: 1,
            len,
        }
    }

    pub(crate) fn len(&self) -> u64 {
        match self {
            Self::Span { len, .. } => *len,
            Self::Selected(values) => values.len() as u64,
        }
    }

    pub(crate) fn at(&self, index: u64) -> u64 {
        match self {
            Self::Span { start, step, .. } => start + step * index,
            Self::Selected(values) => values[index as usize],
        }
    }

    /// Constant-step subsequences preserve caller selection order and duplicates.
    /// Only the iterator cursor is retained; no segment collection is allocated.
    pub(crate) fn segments(&self) -> impl Iterator<Item = (u64, i128, u64)> + '_ {
        let mut i = 0;
        std::iter::from_fn(move || {
            if i == self.len() {
                return None;
            }

            let start = self.at(i);
            let (step, end) = match self {
                Self::Span { step, len, .. } => (*step as i128, *len),
                Self::Selected(values) => {
                    let step = values
                        .get(i as usize + 1)
                        .map(|v| *v as i128 - start as i128)
                        .unwrap_or(0);
                    let mut end = i + 1;

                    while end < self.len()
                        && self.at(end) as i128 - self.at(end - 1) as i128 == step
                    {
                        end += 1;
                    }
                    (step, end)
                }
            };

            let len = end - i;
            i = end;
            Some((start, step, len))
        })
    }

    pub(crate) fn validate(&self, dim: u64) -> Result<()> {
        match self {
            Self::Span { start, step, len } => {
                if *step == 0 || *start > dim {
                    return Err(Error::InvalidCoord("invalid slice start or step".into()));
                }

                if *len > 0 {
                    let last = step
                        .checked_mul(len - 1)
                        .and_then(|n| start.checked_add(n))
                        .ok_or_else(|| Error::InvalidCoord("request axis overflow".into()))?;
                    if last >= dim {
                        return Err(Error::InvalidCoord("request axis out of bounds".into()));
                    }
                }
            }
            Self::Selected(values) => {
                if values.iter().any(|v| *v >= dim) {
                    return Err(Error::InvalidCoord(
                        "request selection out of bounds".into(),
                    ));
                }
            }
        }

        Ok(())
    }
}

// Large descriptors stay heap-backed so each request does not embed eight of them.
#[derive(Clone, Debug)]
pub(crate) struct Cartesian {
    axes: Vec<Axis>,
    len: usize,
}

impl Cartesian {
    pub(crate) fn new(axes: Vec<Axis>) -> Result<Self> {
        for axis in &axes {
            if let Axis::Selected(values) = axis {
                bound(values.len())?;
            }
        }

        let len = if axes.iter().any(|axis| axis.len() == 0) {
            0
        } else {
            axes.iter()
                .try_fold(1u64, |n, a| n.checked_mul(a.len()))
                .ok_or_else(|| Error::InvalidLayout("request cardinality overflow".into()))?
        };

        let len = usize::try_from(len)
            .map_err(|_| Error::InvalidLayout("request cardinality exceeds usize".into()))?;
        bound(len)?;
        Ok(Self { axes, len })
    }

    pub(crate) fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn validate(&self, shape: &[u64]) -> Result<()> {
        if self.axes.len() != shape.len() {
            return Err(Error::InvalidCoord("request rank mismatch".into()));
        }

        for (axis, dim) in self.axes.iter().zip(shape) {
            axis.validate(*dim)?;
        }

        Ok(())
    }

    fn coordinate(&self, mut index: usize, out: &mut Coord) {
        out.resize(self.axes.len(), 0);

        for (axis, value) in self.axes.iter().zip(out.iter_mut()).rev() {
            *value = axis.at(index as u64 % axis.len());
            index /= axis.len() as usize;
        }
    }
}

/// Ordered flat addresses; signed steps retain reversals and zero steps broadcasts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Progression {
    pub output: usize,
    pub start: i128,
    pub step: i128,
    pub len: usize,
}

fn invalid() -> Error {
    Error::InvalidLayout("invalid request run bounds or arithmetic overflow".into())
}

fn push_progression(runs: &mut Vec<Progression>, next: Progression) {
    if let Some(last) = runs.last_mut() {
        let step = if last.len == 1 {
            match next.start.checked_sub(last.start) {
                Some(step) => step,
                None => {
                    runs.push(next);
                    return;
                }
            }
        } else {
            last.step
        };

        if last.output + last.len == next.output
            && (next.len == 1 || next.step == step)
            && step
                .checked_mul(last.len as i128)
                .and_then(|n| last.start.checked_add(n))
                == Some(next.start)
        {
            last.step = step;
            last.len += next.len;
            return;
        }
    }
    runs.push(next);
}

#[derive(Clone, Debug)]
pub(crate) enum RequestKind {
    Linear { start: u64 },
    Rectangles(Vec<Cartesian>),
    Explicit(Vec<Vec<u64>>),
    FlatRuns(Vec<Progression>),
}

#[derive(Clone, Debug)]
pub struct BatchRequest {
    kind: RequestKind,
    len: usize,
}

fn bound(len: usize) -> Result<()> {
    if len > MAX_BATCH_ELEMENTS {
        return Err(Error::InvalidLayout(format!(
            "coordinate batch: expected at most {MAX_BATCH_ELEMENTS} elements, got {len}"
        )));
    }

    Ok(())
}

impl BatchRequest {
    pub(crate) fn linear(start: u64, len: usize) -> Result<Self> {
        bound(len)?;
        start
            .checked_add(len as u64)
            .ok_or_else(|| Error::InvalidLayout("request span overflow".into()))?;
        Ok(Self {
            kind: RequestKind::Linear { start },
            len,
        })
    }

    pub(crate) fn rectangles(rectangles: Vec<Cartesian>) -> Result<Self> {
        // Bound metadata as well as logical elements, even for empty rectangles.
        bound(rectangles.len())?;
        let len = rectangles
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(r.len()))
            .ok_or_else(|| Error::InvalidLayout("request cardinality overflow".into()))?;
        bound(len)?;
        Ok(Self {
            kind: RequestKind::Rectangles(rectangles),
            len,
        })
    }

    pub(crate) fn explicit(coords: Vec<Vec<u64>>) -> Result<Self> {
        bound(coords.len())?;
        Ok(Self {
            len: coords.len(),
            kind: RequestKind::Explicit(coords),
        })
    }

    pub(crate) fn point(coord: &[u64]) -> Self {
        Self {
            len: 1,
            kind: RequestKind::Explicit(vec![coord.to_vec()]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn kind(&self) -> &RequestKind {
        &self.kind
    }

    /// Map addresses without materializing coordinates. Affine geometry shares
    /// storage's progression planner; irregular geometry reuses rank-sized scratch.
    pub(crate) fn mapped(
        &self,
        mapping: &CoordinateMap,
        base_shape: &[u64],
        base_strides: &[u64],
    ) -> Result<Self> {
        self.validate(&mapping.shape)?;
        let mut runs = Vec::new();
        let affine = match self.kind {
            RequestKind::Explicit(_) | RequestKind::FlatRuns(_) => None,
            _ => mapping.affine()?,
        };

        if let Some((offset, strides)) = affine {
            progressions(self, &mapping.shape, offset, &strides, |run| {
                push_progression(&mut runs, run);
                Ok(())
            })?;
        } else {
            mapping.visit_mapped(self, base_shape, base_strides, |output, coord| {
                let start = coord
                    .iter()
                    .zip(base_strides)
                    .try_fold(0i128, |n, (c, s)| {
                        (*c as i128)
                            .checked_mul(*s as i128)
                            .and_then(|v| n.checked_add(v))
                            .ok_or_else(invalid)
                    })?;
                push_progression(
                    &mut runs,
                    Progression {
                        output,
                        start,
                        step: 0,
                        len: 1,
                    },
                );
                Ok(())
            })?;
        }

        let mapped = Self {
            kind: RequestKind::FlatRuns(runs),
            len: self.len,
        };
        mapped.validate(base_shape)?;
        #[cfg(test)]
        if let RequestKind::FlatRuns(runs) = &mapped.kind {
            crate::read_metrics::record(|m| m.mapped_runs += runs.len());
        }

        Ok(mapped)
    }

    pub(crate) fn validate(&self, shape: &[u64]) -> Result<()> {
        match &self.kind {
            RequestKind::Linear { start } => {
                let total = crate::schema::checked_product(shape)?;
                if start + self.len as u64 > total {
                    return Err(Error::InvalidCoord("request span out of bounds".into()));
                }
            }
            RequestKind::Rectangles(rectangles) => {
                for rect in rectangles {
                    rect.validate(shape)?;
                }
            }
            RequestKind::FlatRuns(runs) => {
                bound(self.len)?;
                let total = schema::checked_product(shape)? as i128;
                let mut end = 0usize;

                for run in runs {
                    if run.len == 0 || run.output != end {
                        return Err(invalid());
                    }

                    let last = run
                        .step
                        .checked_mul((run.len - 1) as i128)
                        .and_then(|n| run.start.checked_add(n))
                        .ok_or_else(invalid)?;
                    end = end.checked_add(run.len).ok_or_else(invalid)?;
                    if end > self.len {
                        return Err(invalid());
                    }

                    if run.start < 0 || run.start >= total || last < 0 || last >= total {
                        return Err(Error::InvalidCoord(
                            "mapped progression out of bounds".into(),
                        ));
                    }
                }

                if end != self.len {
                    return Err(invalid());
                }
            }
            RequestKind::Explicit(coords) => {
                for coord in coords {
                    validate::validate_coord(shape, coord)?;
                }
            }
        }

        Ok(())
    }

    pub(crate) fn cursor<'a>(&'a self, shape: &'a [u64]) -> Result<Cursor<&'a Self, &'a [u64]>> {
        Cursor::new(self, shape)
    }

    /// Retain compact request metadata across stream polls without expanding coordinates.
    pub(crate) fn into_cursor(self, shape: Arc<[u64]>) -> Result<Cursor<Self, Arc<[u64]>>> {
        Cursor::new(self, shape)
    }

    /// Transfer an already evaluated request to a public coordinate payload.
    /// Explicit requests were validated by evaluation and need no second copy.
    pub(crate) fn into_coordinates(self, shape: &[u64]) -> Result<Vec<Vec<u64>>> {
        match self {
            Self {
                kind: RequestKind::Explicit(coords),
                ..
            } => Ok(coords),
            other => other.coordinates(shape),
        }
    }

    /// Expand at a public stream boundary, or for a test's reference coordinates.
    pub(crate) fn coordinates(&self, shape: &[u64]) -> Result<Vec<Vec<u64>>> {
        let mut cursor = self.cursor(shape)?;
        #[cfg(test)]
        crate::read_metrics::record(|m| m.expanded_coordinates += self.len);
        let mut scratch = Coord::new();
        let mut out = Vec::with_capacity(self.len);

        while cursor.next_into(&mut scratch) {
            out.push(scratch.to_vec());
        }

        Ok(out)
    }
}

pub(crate) struct Cursor<R, S> {
    request: R,
    shape: S,
    position: usize,
    rectangle: usize,
    local: usize,
}

impl<R: Borrow<BatchRequest>, S: AsRef<[u64]>> Cursor<R, S> {
    fn new(request: R, shape: S) -> Result<Self> {
        request.borrow().validate(shape.as_ref())?;
        Ok(Self {
            request,
            shape,
            position: 0,
            rectangle: 0,
            local: 0,
        })
    }

    pub(crate) fn next_into(&mut self, out: &mut Coord) -> bool {
        let request = self.request.borrow();
        let shape = self.shape.as_ref();
        if self.position == request.len {
            return false;
        }

        match &request.kind {
            RequestKind::Linear { start } => {
                decode_flat(start + self.position as u64, shape, out)
                    .expect("validated linear request");
            }
            RequestKind::Rectangles(rectangles) => {
                while self.local == rectangles[self.rectangle].len {
                    self.rectangle += 1;
                    self.local = 0;
                }
                rectangles[self.rectangle].coordinate(self.local, out);
                self.local += 1;
            }
            RequestKind::FlatRuns(runs) => {
                while self.local == runs[self.rectangle].len {
                    self.rectangle += 1;
                    self.local = 0;
                }

                let run = &runs[self.rectangle];
                let flat = run.start + run.step * self.local as i128;
                decode_flat(flat as u64, shape, out).expect("validated flat run");
                self.local += 1;
            }
            RequestKind::Explicit(coords) => {
                out.clear();
                out.extend_from_slice(&coords[self.position]);
            }
        }

        self.position += 1;
        true
    }
}

pub(crate) fn linear_requests(
    shape: &[u64],
) -> Result<impl Iterator<Item = BatchRequest> + Send + use<>> {
    schema::validate_shape_dims(shape)?;
    let total = crate::schema::checked_product(shape)?;
    Ok((0..total).step_by(MAX_BATCH_ELEMENTS).map(move |start| {
        BatchRequest::linear(
            start,
            (total - start).min(MAX_BATCH_ELEMENTS as u64) as usize,
        )
        .expect("bounded span")
    }))
}

/// Pack whole spatial tiles. Descriptors use rank-sized axes; at most
/// `MAX_BATCH_ELEMENTS` nonempty tiles fit a request, including singleton matrices.
pub(crate) fn tiled_requests(
    shape: &[u64],
) -> Result<impl Iterator<Item = BatchRequest> + Send + use<>> {
    schema::validate_shape_dims(shape)?;
    let shape = crate::Shape::from_slice(shape);
    let rank = shape.len();
    let mut grid = shape.clone();

    for dim in grid.iter_mut().rev().take(2) {
        *dim = dim.div_ceil(crate::matmul::TILE_SIDE as u64);
    }

    let mut tiles = schema::row_major_coords(&grid)?.peekable();
    Ok(std::iter::from_fn(move || {
        let mut rectangles = Vec::new();
        let mut size = 0;

        while let Some(tile) = tiles.peek() {
            let axes: Vec<_> = tile
                .iter()
                .enumerate()
                .map(|(i, index)| {
                    let side = if i + 2 >= rank {
                        crate::matmul::TILE_SIDE as u64
                    } else {
                        1
                    };

                    let start = *index * side;
                    Axis::range(start, (shape[i] - start).min(side))
                })
                .collect();
            let rect = Cartesian::new(axes).expect("bounded spatial tile");
            if size + rect.len() > MAX_BATCH_ELEMENTS {
                break;
            }
            size += rect.len();
            rectangles.push(rect);
            tiles.next();
        }

        if rectangles.is_empty() {
            None
        } else {
            Some(BatchRequest::rectangles(rectangles).expect("bounded tile batch"))
        }
    }))
}

// Invoke once per fastest-axis arithmetic progression, not per output element.
// Selected-axis scans use caller-bounded metadata and preserve duplicates/order.
pub(crate) fn progressions(
    request: &BatchRequest,
    shape: &[u64],
    offset: i128,
    strides: &[i128],
    mut emit: impl FnMut(Progression) -> Result<()>,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(invalid());
    }

    if request.len() == 0 {
        return Ok(());
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
        RequestKind::Explicit(_) | RequestKind::FlatRuns(_) => {
            unreachable!("non-affine requests use the coordinate cursor")
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/request/tests.rs"]
mod tests;
