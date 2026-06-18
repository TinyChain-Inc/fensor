use std::sync::Arc;

use ha_ndarray::{Axes, AxisRange, Range, Shape, Strides};

use crate::{
    Error, Result, TensorSchema, TensorShape, ViewAxisMapSchema, ViewAxisSchema, ViewSchema,
};

#[derive(Clone)]
pub struct TensorView {
    base_rank: usize,
    axes: Vec<ViewAxis>,
    base_fixed: Vec<Option<u64>>,
}

#[derive(Clone)]
struct ViewAxis {
    base_axis: usize,
    map: AxisMap,
}

#[derive(Clone)]
enum AxisMap {
    Identity,
    Affine { start: usize, step: usize },
    Gather(Arc<[usize]>),
}

impl TensorView {
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    pub fn base_rank(&self) -> usize {
        self.base_rank
    }

    pub fn identity(schema: &TensorSchema) -> Self {
        let base_rank = schema.shape().len();
        let axes = (0..base_rank)
            .map(|base_axis| ViewAxis {
                base_axis,
                map: AxisMap::Identity,
            })
            .collect();

        Self {
            base_rank,
            axes,
            base_fixed: vec![None; base_rank],
        }
    }

    pub fn resolve_coord(&self, coord: &[u64]) -> Result<Vec<u64>> {
        if coord.len() != self.axes.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }

        let mut resolved = vec![0u64; self.base_rank()];
        let mut assigned = vec![false; self.base_rank()];

        for (base_axis, fixed) in self.base_fixed.iter().enumerate() {
            if let Some(value) = fixed {
                resolved[base_axis] = *value;
                assigned[base_axis] = true;
            }
        }

        for (axis, coord_value) in self.axes.iter().zip(coord.iter()) {
            let coord_value: usize = (*coord_value)
                .try_into()
                .map_err(|_| Error::InvalidCoord("coordinate does not fit in usize".to_string()))?;

            resolved[axis.base_axis] = u64::try_from(axis.map.resolve(coord_value)?)
                .map_err(|_| Error::InvalidCoord("coordinate does not fit in u64".to_string()))?;
            assigned[axis.base_axis] = true;
        }

        if assigned.iter().any(|is_set| !is_set) {
            return Err(Error::InvalidLayout(
                "view does not map all base tensor axes".to_string(),
            ));
        }

        Ok(resolved)
    }

    pub fn transpose(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.len() != self.axes.len() {
            return Err(Error::InvalidLayout(
                "transpose permutation rank must match tensor rank".to_string(),
            ));
        }

        let mut seen = vec![false; self.axes.len()];
        let mut axes = Vec::with_capacity(self.axes.len());

        for permuted_axis in permutation {
            if *permuted_axis >= self.axes.len() || seen[*permuted_axis] {
                return Err(Error::InvalidLayout(
                    "transpose permutation must be a valid axis permutation".to_string(),
                ));
            }

            seen[*permuted_axis] = true;
            axes.push(self.axes[*permuted_axis].clone());
        }

        Ok(Self {
            base_rank: self.base_rank,
            axes,
            base_fixed: self.base_fixed.clone(),
        })
    }

    pub fn slice(&self, shape: &[usize], range: &Range) -> Result<(Self, Shape, Strides)> {
        if range.len() != shape.len() {
            return Err(Error::InvalidLayout(
                "slice range rank must match tensor rank".to_string(),
            ));
        }

        let mut axes = Vec::with_capacity(self.axes.len());
        let mut base_fixed = self.base_fixed.clone();
        let mut new_shape = Shape::with_capacity(self.axes.len());
        let mut new_strides = Strides::with_capacity(self.axes.len());

        for (axis_index, (bound, dim)) in range.iter().zip(shape.iter()).enumerate() {
            let view_axis = &self.axes[axis_index];
            let stride = match &view_axis.map {
                AxisMap::Identity => 1,
                AxisMap::Affine { step, .. } => *step,
                AxisMap::Gather(_) => 0,
            };

            match bound {
                AxisRange::At(i) => {
                    if *i >= *dim {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }

                    let base_coord = view_axis.map.resolve(*i)?;
                    base_fixed[view_axis.base_axis] = Some(
                        u64::try_from(base_coord)
                            .map_err(|_| Error::InvalidCoord("coordinate does not fit in u64".to_string()))?,
                    );
                }
                AxisRange::In(start, stop, step) => {
                    if *step == 0 || *start > *stop || *stop > *dim {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }

                    let extent = if start == stop {
                        0
                    } else {
                        (stop - start).div_ceil(*step)
                    };

                    let map = compose_range(&view_axis.map, *start, *step, extent)?;
                    axes.push(ViewAxis {
                        base_axis: view_axis.base_axis,
                        map,
                    });
                    new_shape.push(extent);
                    new_strides.push(stride.saturating_mul(*step));
                }
                AxisRange::Of(indices) => {
                    if indices.iter().any(|index| *index >= *dim) {
                        return Err(Error::InvalidLayout(format!(
                            "slice bound at axis {axis_index} is out of bounds"
                        )));
                    }

                    let map = compose_gather(&view_axis.map, indices)?;
                    axes.push(ViewAxis {
                        base_axis: view_axis.base_axis,
                        map,
                    });
                    new_shape.push(indices.len());
                    new_strides.push(0);
                }
            }
        }

        Ok((
            Self {
                base_rank: self.base_rank,
                axes,
                base_fixed,
            },
            new_shape,
            new_strides,
        ))
    }

    pub fn to_schema(&self) -> Result<ViewSchema> {
        let axes = self
            .axes
            .iter()
            .map(ViewAxis::to_schema)
            .collect::<Result<_>>()?;

        Ok(ViewSchema {
            base_rank: self.base_rank(),
            axes,
            base_fixed: self.base_fixed.iter().copied().collect(),
        })
    }

    pub fn from_schema(schema: &ViewSchema) -> Result<Self> {
        let axes = schema
            .axes
            .iter()
            .map(ViewAxis::from_schema)
            .collect::<Result<Vec<_>>>()?;

        let base_fixed = schema.base_fixed.iter().copied().collect::<Vec<_>>();

        if base_fixed.len() != schema.base_rank {
            return Err(Error::InvalidSchema(
                "view base_fixed rank must match base_rank".to_string(),
            ));
        }

        for coord in base_fixed.iter().flatten() {
            let _ = usize::try_from(*coord)
                .map_err(|_| Error::InvalidSchema("view coord overflow".to_string()))?;
        }

        Ok(Self {
            base_rank: schema.base_rank,
            axes,
            base_fixed,
        })
    }
}

impl ViewAxis {
    fn to_schema(&self) -> Result<ViewAxisSchema> {
        Ok(ViewAxisSchema {
            base_axis: self.base_axis,
            map: self.map.to_schema()?,
        })
    }

    fn from_schema(schema: &ViewAxisSchema) -> Result<Self> {
        Ok(Self {
            base_axis: schema.base_axis,
            map: AxisMap::from_schema(&schema.map)?,
        })
    }
}

impl AxisMap {
    fn resolve(&self, index: usize) -> Result<usize> {
        match self {
            Self::Identity => Ok(index),
            Self::Affine { start, step } => start
                .checked_add(index.checked_mul(*step).ok_or_else(|| {
                    Error::InvalidCoord("affine axis mapping overflow".to_string())
                })?)
                .ok_or_else(|| Error::InvalidCoord("affine axis mapping overflow".to_string())),
            Self::Gather(indices) => indices.get(index).copied().ok_or_else(|| {
                Error::InvalidCoord("coordinate out of bounds for gathered axis".to_string())
            }),
        }
    }

    fn to_schema(&self) -> Result<ViewAxisMapSchema> {
        match self {
            Self::Identity => Ok(ViewAxisMapSchema::Identity),
            Self::Affine { start, step } => {
                let start = u64::try_from(*start)
                    .map_err(|_| Error::InvalidSchema("view start overflow".to_string()))?;
                let step = u64::try_from(*step)
                    .map_err(|_| Error::InvalidSchema("view step overflow".to_string()))?;
                Ok(ViewAxisMapSchema::Affine { start, step })
            }
            Self::Gather(indices) => {
                let gathered = indices
                    .iter()
                    .map(|index| {
                        u64::try_from(*index)
                            .map_err(|_| Error::InvalidSchema("view gather overflow".to_string()))
                    })
                    .collect::<Result<TensorShape>>()?;
                Ok(ViewAxisMapSchema::Gather(gathered))
            }
        }
    }

    fn from_schema(schema: &ViewAxisMapSchema) -> Result<Self> {
        match schema {
            ViewAxisMapSchema::Identity => Ok(Self::Identity),
            ViewAxisMapSchema::Affine { start, step } => {
                let start = usize::try_from(*start)
                    .map_err(|_| Error::InvalidSchema("view start overflow".to_string()))?;
                let step = usize::try_from(*step)
                    .map_err(|_| Error::InvalidSchema("view step overflow".to_string()))?;
                Ok(Self::Affine { start, step })
            }
            ViewAxisMapSchema::Gather(indices) => {
                let gathered = indices
                    .iter()
                    .map(|index| {
                        usize::try_from(*index)
                            .map_err(|_| Error::InvalidSchema("view gather overflow".to_string()))
                    })
                    .collect::<Result<Vec<usize>>>()?;
                Ok(Self::Gather(gathered.into()))
            }
        }
    }
}

fn compose_range(map: &AxisMap, start: usize, step: usize, extent: usize) -> Result<AxisMap> {
    if extent == 0 {
        return Ok(AxisMap::Gather(Arc::new([])));
    }

    match map {
        AxisMap::Identity => {
            if start == 0 && step == 1 {
                Ok(AxisMap::Identity)
            } else {
                Ok(AxisMap::Affine { start, step })
            }
        }
        AxisMap::Affine {
            start: base_start,
            step: base_step,
        } => Ok(AxisMap::Affine {
            start: base_start + base_step * start,
            step: base_step * step,
        }),
        AxisMap::Gather(_) => {
            let mut gathered = Vec::with_capacity(extent);
            for i in 0..extent {
                let source = start + i * step;
                gathered.push(map.resolve(source)?);
            }

            Ok(AxisMap::Gather(gathered.into()))
        }
    }
}

fn compose_gather(map: &AxisMap, indices: &[usize]) -> Result<AxisMap> {
    let mut gathered = Vec::with_capacity(indices.len());
    for index in indices {
        gathered.push(map.resolve(*index)?);
    }

    Ok(AxisMap::Gather(gathered.into()))
}

pub fn default_permutation(ndim: usize, permutation: Option<Axes>) -> Result<Vec<usize>> {
    let axes: Vec<usize> = permutation
        .map(|axes| axes.into_iter().collect())
        .unwrap_or_else(|| (0..ndim).rev().collect());

    if axes.len() != ndim {
        return Err(Error::InvalidLayout(
            "transpose permutation rank must match tensor rank".to_string(),
        ));
    }

    Ok(axes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ha_ndarray::{range, shape};

    fn schema(shape: &[usize]) -> TensorSchema {
        TensorSchema::dense(
            shape.iter().copied().collect(),
            shape.iter().copied().collect(),
        )
        .expect("schema")
    }

    #[test]
    fn resolve_identity() {
        let schema = schema(&[4, 5, 6]);
        let view = TensorView::identity(&schema);

        let resolved = view.resolve_coord(&[2, 1, 3]).expect("resolve");
        assert_eq!(resolved, vec![2, 1, 3]);
    }

    #[test]
    fn slice_then_resolve() {
        let schema = schema(&[4, 5, 6]);
        let view = TensorView::identity(&schema);

        let range = range![
            AxisRange::At(2),
            AxisRange::In(1, 5, 2),
            AxisRange::Of(shape![0, 3, 4])
        ];

        let (sliced, shape, _strides) = view.slice(schema.shape(), &range).expect("slice");
        assert_eq!(shape.as_slice(), &[2, 3]);

        let resolved = sliced.resolve_coord(&[1, 2]).expect("resolve");
        assert_eq!(resolved, vec![2, 3, 4]);
    }

    #[test]
    fn transpose_arbitrary_permutation() {
        let schema = schema(&[3, 4, 5]);
        let view = TensorView::identity(&schema);
        let perm = vec![2, 0, 1];

        let transposed = view.transpose(&perm).expect("transpose");
        let resolved = transposed.resolve_coord(&[1, 2, 3]).expect("resolve");
        assert_eq!(resolved, vec![2, 3, 1]);
    }

    #[test]
    fn compose_transpose_and_slice() {
        let schema = schema(&[3, 4, 5]);
        let view = TensorView::identity(&schema);

        let transposed = view.transpose(&[2, 0, 1]).expect("transpose");
        let shape: Shape = shape![5, 3, 4];
        let range: Range = range![
            AxisRange::In(1, 5, 2),
            AxisRange::At(2),
            AxisRange::In(0, 4, 2)
        ];

        let (sliced, new_shape, _) = transposed.slice(&shape, &range).expect("slice");
        assert_eq!(new_shape.as_slice(), &[2, 2]);

        let resolved = sliced.resolve_coord(&[1, 1]).expect("resolve");
        assert_eq!(resolved, vec![2, 2, 3]);
    }

    #[test]
    fn view_schema_roundtrip() {
        let schema = schema(&[5, 4, 3]);
        let identity = TensorView::identity(&schema);
        let transposed = identity.transpose(&[2, 0, 1]).expect("transpose");
        let view_schema = transposed.to_schema().expect("schema");

        let restored = TensorView::from_schema(&view_schema).expect("restore");
        assert_eq!(restored.resolve_coord(&[1, 2, 3]).expect("coord"), vec![2, 3, 1]);
    }
}
