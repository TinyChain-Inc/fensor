//! Compact, bounded execution requests. Coordinates are expanded only by consumers
//! which need explicit payloads; cursors reuse one rank-sized scratch buffer.

use crate::expression::MAX_BATCH_ELEMENTS;
use crate::{Error, Result, schema, validate};

#[derive(Clone, Debug)]
pub(crate) enum Axis {
    Span {
        start: usize,
        step: usize,
        len: usize,
    },
    Selected(Vec<u64>),
}

impl Axis {
    pub(crate) fn range(start: usize, len: usize) -> Self {
        Self::Span {
            start,
            step: 1,
            len,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Span { len, .. } => *len,
            Self::Selected(values) => values.len(),
        }
    }

    pub(crate) fn at(&self, index: usize) -> u64 {
        match self {
            Self::Span { start, step, .. } => (start + step * index) as u64,
            Self::Selected(values) => values[index],
        }
    }

    pub(crate) fn validate(&self, dim: usize) -> Result<()> {
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
                if values.iter().any(|v| *v >= dim as u64) {
                    return Err(Error::InvalidCoord(
                        "request selection out of bounds".into(),
                    ));
                }
            }
        }

        Ok(())
    }
}

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

        let len = axes
            .iter()
            .try_fold(1usize, |n, a| n.checked_mul(a.len()))
            .ok_or_else(|| Error::InvalidLayout("request cardinality overflow".into()))?;
        bound(len)?;
        Ok(Self { axes, len })
    }

    pub(crate) fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn validate(&self, shape: &[usize]) -> Result<()> {
        if self.axes.len() != shape.len() {
            return Err(Error::InvalidCoord("request rank mismatch".into()));
        }

        for (axis, dim) in self.axes.iter().zip(shape) {
            axis.validate(*dim)?;
        }

        Ok(())
    }

    fn coordinate(&self, mut index: usize, out: &mut Vec<u64>) {
        out.resize(self.axes.len(), 0);

        for (axis, value) in self.axes.iter().zip(out.iter_mut()).rev() {
            *value = axis.at(index % axis.len());
            index /= axis.len();
        }
    }
}

#[derive(Debug)]
pub(crate) enum RequestKind {
    Linear { start: usize },
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
    pub(crate) fn linear(start: usize, len: usize) -> Result<Self> {
        bound(len)?;
        start
            .checked_add(len)
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

    pub(crate) fn validate(&self, shape: &[usize]) -> Result<()> {
        match &self.kind {
            RequestKind::Linear { start } => {
                let total = shape
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| Error::InvalidLayout("request shape overflow".into()))?;
                if start + self.len > total {
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

    pub(crate) fn cursor<'a>(&'a self, shape: &'a [usize]) -> Result<Cursor<'a>> {
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
    pub(crate) fn into_coordinates(self, shape: &[usize]) -> Result<Vec<Vec<u64>>> {
        match self {
            Self {
                kind: RequestKind::Explicit(coords),
                ..
            } => Ok(coords),
            other => other.coordinates(shape),
        }
    }

    /// Expand at a public stream boundary, or for a test's reference coordinates.
    pub(crate) fn coordinates(&self, shape: &[usize]) -> Result<Vec<Vec<u64>>> {
        let mut cursor = self.cursor(shape)?;
        let mut scratch = Vec::new();
        let mut out = Vec::with_capacity(self.len);

        while cursor.next_into(&mut scratch) {
            out.push(scratch.clone());
        }

        Ok(out)
    }
}

pub(crate) struct Cursor<'a> {
    request: &'a BatchRequest,
    shape: &'a [usize],
    position: usize,
    rectangle: usize,
    local: usize,
}

impl Cursor<'_> {
    pub(crate) fn next_into(&mut self, out: &mut Vec<u64>) -> bool {
        if self.position == self.request.len {
            return false;
        }
        match &self.request.kind {
            RequestKind::Linear { start } => {
                let mut index = start + self.position;
                out.resize(self.shape.len(), 0);

                for (dim, value) in self.shape.iter().zip(out.iter_mut()).rev() {
                    *value = (index % dim) as u64;
                    index /= dim;
                }
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
    shape: &[usize],
) -> Result<impl Iterator<Item = BatchRequest> + Send + use<>> {
    schema::validate_shape_dims(shape)?;
    let total = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| Error::InvalidLayout("request shape overflow".into()))?;
    Ok((0..total).step_by(MAX_BATCH_ELEMENTS).map(move |start| {
        BatchRequest::linear(start, (total - start).min(MAX_BATCH_ELEMENTS)).expect("bounded span")
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
    shape: &[usize],
) -> Result<impl Iterator<Item = BatchRequest> + Send + use<>> {
    schema::validate_shape_dims(shape)?;
    let shape = shape.to_vec();
    let rank = shape.len();
    let mut grid = shape.clone();

    for dim in grid.iter_mut().rev().take(2) {
        *dim = dim.div_ceil(crate::matmul::TILE_SIDE);
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
                        crate::matmul::TILE_SIDE
                    } else {
                        1
                    };
                    let start = *index as usize * side;
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
    fn linear_batches_preserve_boundaries_and_validate_shape() {
        for len in [
            1,
            MAX_BATCH_ELEMENTS - 1,
            MAX_BATCH_ELEMENTS,
            MAX_BATCH_ELEMENTS + 1,
        ] {
            let batches = linear_requests(&[len]).unwrap().collect::<Vec<_>>();
            assert_eq!(batches.len(), len.div_ceil(MAX_BATCH_ELEMENTS));
            let mut next = 0;

            for batch in batches {
                assert!(matches!(batch.kind(), RequestKind::Linear { start } if *start == next));
                assert_eq!(batch.len(), (len - next).min(MAX_BATCH_ELEMENTS));
                next += batch.len();
            }
            assert_eq!(next, len);
        }

        for shape in [vec![], vec![0], vec![2, 0], vec![usize::MAX, 2]] {
            assert!(linear_requests(&shape).is_err());
        }

        // Taking a prefix retains compact descriptors, not the logical tensor.
        let prefix = linear_requests(&[usize::MAX])
            .unwrap()
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(prefix.len(), 2);
        assert!(prefix.iter().all(|batch| batch.len() == MAX_BATCH_ELEMENTS));
        assert!(matches!(
            prefix[1].kind(),
            RequestKind::Linear {
                start: MAX_BATCH_ELEMENTS
            }
        ));
    }

    #[test]
    fn request_validation_and_cursor_reuse() {
        assert!(BatchRequest::linear(usize::MAX, 1).is_err());
        assert!(BatchRequest::linear(0, MAX_BATCH_ELEMENTS + 1).is_err());
        assert!(
            BatchRequest::linear(5, 2)
                .unwrap()
                .validate(&[2, 3])
                .is_err()
        );
        assert!(Cartesian::new(vec![Axis::range(0, usize::MAX), Axis::range(0, 2)]).is_err());
        let bad = BatchRequest::rectangles(vec![
            Cartesian::new(vec![Axis::Span {
                start: usize::MAX,
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
        let mut scratch = Vec::with_capacity(2);
        let ptr = scratch.as_ptr();

        for expected in expected {
            assert!(cursor.next_into(&mut scratch));
            assert_eq!(scratch, expected);
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
