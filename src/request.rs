//! Compact, bounded execution requests. Coordinates are expanded only by consumers
//! which need explicit payloads; cursors reuse one rank-sized scratch buffer.

use crate::expression::MAX_BATCH_ELEMENTS;
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

#[derive(Debug)]
pub(crate) enum RequestKind {
    Linear { start: u64 },
    Rectangles(Vec<Cartesian>),
    Explicit(Vec<Vec<u64>>),
}

#[derive(Debug)]
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
            RequestKind::Explicit(coords) => {
                for coord in coords {
                    validate::validate_coord(shape, coord)?;
                }
            }
        }

        Ok(())
    }

    pub(crate) fn cursor<'a>(&'a self, shape: &'a [u64]) -> Result<Cursor<'a>> {
        self.validate(shape)?;
        Ok(Cursor {
            request: self,
            shape,
            position: 0,
            rectangle: 0,
            local: 0,
        })
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
        let mut scratch = Coord::new();
        let mut out = Vec::with_capacity(self.len);

        while cursor.next_into(&mut scratch) {
            out.push(scratch.to_vec());
        }

        Ok(out)
    }
}

pub(crate) struct Cursor<'a> {
    request: &'a BatchRequest,
    shape: &'a [u64],
    position: usize,
    rectangle: usize,
    local: usize,
}

impl Cursor<'_> {
    pub(crate) fn next_into(&mut self, out: &mut Coord) -> bool {
        if self.position == self.request.len {
            return false;
        }
        match &self.request.kind {
            RequestKind::Linear { start } => {
                decode_flat(start + self.position as u64, self.shape, out)
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

pub(crate) fn explicit_requests(
    coords: impl Iterator<Item = Vec<u64>>,
) -> impl Iterator<Item = BatchRequest> {
    crate::traits::coordinate_batches(coords)
        .map(|coords| BatchRequest::explicit(coords).expect("bounded coordinate iterator"))
}

/// Pack whole spatial tiles. Descriptors use rank-sized axes; at most 4096
/// nonempty tiles fit a request, including batches of singleton matrices.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_segments_preserve_order_bounds_and_exhaustion() {
        for axis in [
            Axis::range(8, 0),
            Axis::range(8, 7),
            Axis::Selected(vec![]),
            Axis::Selected(vec![9, 7, 5, 5, 5, 8, 2]),
            Axis::Selected(vec![u64::MAX, 0, u64::MAX]),
        ] {
            let values: Vec<_> = axis
                .segments()
                .flat_map(|(start, step, len)| {
                    (0..len).map(move |i| (start as i128 + step * i as i128) as u64)
                })
                .collect();
            assert_eq!(
                values,
                (0..axis.len()).map(|i| axis.at(i)).collect::<Vec<_>>()
            );
        }
        let mut scratch = Coord::new();
        for (flat, offset, len) in linear_segments(5, 17, 7).unwrap() {
            decode_flat(flat, &[4, 7], &mut scratch).unwrap();
            assert_eq!(flat, 5 + offset as u64);
            assert!(scratch[1] + len as u64 <= 7);
        }
        let mut huge = linear_segments(0, usize::MAX, 1).unwrap();
        assert_eq!(
            huge.by_ref().take(3).collect::<Vec<_>>(),
            vec![(0, 0, 1), (1, 1, 1), (2, 2, 1)]
        );
        let mut empty = linear_segments(0, 0, 7).unwrap();
        assert_eq!(empty.next(), None);
        assert_eq!(empty.next(), None);
        assert!(linear_segments(u64::MAX, 1, 7).is_err());
        assert!(linear_segments(0, 1, 0).is_err());
        assert!(decode_flat(0, &[0], &mut scratch).is_err());
        assert!(decode_flat(6, &[2, 3], &mut scratch).is_err());
        decode_flat(u64::MAX - 1, &[u64::MAX], &mut scratch).unwrap();
        assert_eq!(scratch.as_slice(), &[u64::MAX - 1]);
    }

    #[test]
    fn linear_batches_preserve_boundaries_and_validate_shape() {
        for len in [
            1,
            MAX_BATCH_ELEMENTS - 1,
            MAX_BATCH_ELEMENTS,
            MAX_BATCH_ELEMENTS + 1,
        ] {
            let batches = linear_requests(&[len as u64]).unwrap().collect::<Vec<_>>();
            assert_eq!(batches.len(), len.div_ceil(MAX_BATCH_ELEMENTS));
            let mut next = 0;

            for batch in batches {
                assert!(
                    matches!(batch.kind(), RequestKind::Linear { start } if *start == next as u64)
                );
                assert_eq!(batch.len(), (len - next).min(MAX_BATCH_ELEMENTS));
                next += batch.len();
            }
            assert_eq!(next, len);
        }

        for shape in [vec![], vec![0], vec![2, 0], vec![u64::MAX, 2]] {
            assert!(linear_requests(&shape).is_err());
        }

        // Taking a prefix retains compact descriptors, not the logical tensor.
        let prefix = linear_requests(&[u64::MAX])
            .unwrap()
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(prefix.len(), 2);
        assert!(prefix.iter().all(|batch| batch.len() == MAX_BATCH_ELEMENTS));
        assert!(matches!(
            prefix[1].kind(),
            RequestKind::Linear {
                start
            } if *start == MAX_BATCH_ELEMENTS as u64
        ));
    }

    #[test]
    fn request_validation_and_cursor_reuse() {
        assert!(BatchRequest::linear(u64::MAX, 1).is_err());
        assert!(BatchRequest::linear(0, MAX_BATCH_ELEMENTS + 1).is_err());
        assert!(
            BatchRequest::linear(5, 2)
                .unwrap()
                .validate(&[2, 3])
                .is_err()
        );
        assert!(Cartesian::new(vec![Axis::range(0, u64::MAX), Axis::range(0, 2)]).is_err());
        // Reject an unallocatable extent before narrowing it to a buffer length.
        assert!(matches!(
            Cartesian::new(vec![Axis::range(0, u64::MAX)]),
            Err(Error::InvalidLayout(_))
        ));
        let bad = BatchRequest::rectangles(vec![
            Cartesian::new(vec![Axis::Span {
                start: u64::MAX,
                step: 2,
                len: 2,
            }])
            .unwrap(),
        ])
        .unwrap();
        assert!(bad.validate(&[10]).is_err());
        assert!(BatchRequest::point(&[0, 0]).validate(&[1]).is_err());
        let request = BatchRequest::rectangles(vec![
            Cartesian::new(vec![
                Axis::Selected(vec![2, 0, 2]),
                Axis::Span {
                    start: 1,
                    step: 2,
                    len: 2,
                },
            ])
            .unwrap(),
        ])
        .unwrap();
        let expected = vec![
            vec![2, 1],
            vec![2, 3],
            vec![0, 1],
            vec![0, 3],
            vec![2, 1],
            vec![2, 3],
        ];
        assert_eq!(request.coordinates(&[3, 4]).unwrap(), expected);
        let mut cursor = request.cursor(&[3, 4]).unwrap();
        let mut scratch = Coord::with_capacity(2);
        let ptr = scratch.as_ptr();

        for expected in expected {
            assert!(cursor.next_into(&mut scratch));
            assert_eq!(scratch.as_slice(), expected.as_slice());
            assert_eq!(scratch.as_ptr(), ptr);
        }
        assert!(!cursor.next_into(&mut scratch));
    }

    #[test]
    fn spans_and_packed_tiles_preserve_coverage() {
        assert_eq!(
            BatchRequest::linear(2, 5)
                .unwrap()
                .coordinates(&[3, 3])
                .unwrap(),
            vec![vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2], vec![2, 0]]
        );

        for shape in [vec![2, 33, 35], vec![3, 1, 1], vec![1, 1, 129]] {
            let mut all = Vec::new();

            for request in tiled_requests(&shape).unwrap() {
                assert!(request.len() <= MAX_BATCH_ELEMENTS);
                assert!(matches!(request.kind(), RequestKind::Rectangles(_)));
                all.extend(request.coordinates(&shape).unwrap());
            }
            all.sort();
            assert_eq!(
                all,
                schema::row_major_coords(&shape)
                    .unwrap()
                    .collect::<Vec<_>>()
            );
        }

        let first = tiled_requests(&[1_000_000_000, 1_000_000_000])
            .unwrap()
            .next()
            .unwrap();
        let RequestKind::Rectangles(rectangles) = first.kind() else {
            panic!("expected rectangles")
        };
        assert_eq!(rectangles.len(), 4);
        assert_eq!(first.len(), MAX_BATCH_ELEMENTS);
    }

    #[test]
    fn range_requests_tile_without_expanding_coordinates() {
        let axes = vec![Axis::range(3, 2), Axis::range(0, 10_000)];
        let requests: Vec<_> = crate::slice::Slice::new(&[10, 10_000], axes)
            .unwrap()
            .requests()
            .collect();
        assert!(
            requests
                .iter()
                .all(|request| request.len() <= MAX_BATCH_ELEMENTS)
        );
        let mut coordinates = Vec::new();

        for request in requests {
            coordinates.extend(request.coordinates(&[10, 10_000]).unwrap());
        }
        assert_eq!(coordinates.len(), 20_000);
        assert_eq!(coordinates.first(), Some(&vec![3, 0]));
        assert_eq!(coordinates.last(), Some(&vec![4, 9_999]));
    }
}
