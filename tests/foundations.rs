use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use fensor::{
    BoxFuture, DType, Error, Layout, TensorArray, TensorRead, TensorSchema, TensorTransform,
    TensorWrite, contiguous_strides,
};
use ha_ndarray::{Axes, AxisRange, Range, Shape, Strides, axes, range, shape};

mod common;

fn block_on<F: Future>(future: F) -> F::Output {
    fn noop_raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        fn wake(_: *const ()) {}
        fn wake_by_ref(_: *const ()) {}
        fn drop(_: *const ()) {}

        RawWaker::new(
            std::ptr::null(),
            &RawWakerVTable::new(clone, wake, wake_by_ref, drop),
        )
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);

    loop {
        match Future::poll(Pin::as_mut(&mut future), &mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[derive(Clone)]
struct TestTensor {
    schema: TensorSchema,
    layout: Layout,
    shape: Shape,
    strides: Strides,
    values: Vec<f32>,
    base_offset: usize,
}

impl TestTensor {
    fn new(layout: Layout) -> Self {
        let shape: Shape = shape![2, 3, 4];
        let schema = TensorSchema::new(DType::F32, shape.clone()).expect("valid schema");
        let strides = schema.strides().clone();
        let values = vec![0.0; shape.iter().product()];

        Self {
            schema,
            layout,
            shape,
            strides,
            values,
            base_offset: 0,
        }
    }

    fn validate_coord(&self, coord: &[u64]) -> fensor::Result<()> {
        if coord.len() != self.shape.len() {
            return Err(Error::InvalidCoord(
                "incorrect number of coordinates".to_string(),
            ));
        }
        for (c, dim) in coord.iter().zip(self.shape.iter()) {
            if (*c as usize) >= *dim {
                return Err(Error::InvalidCoord("coordinate out of bounds".to_string()));
            }
        }
        Ok(())
    }

    fn linear_offset(&self, coord: &[u64]) -> usize {
        self.base_offset
            + coord
                .iter()
                .zip(self.strides.iter())
                .map(|(c, s)| (*c as usize) * *s)
                .sum::<usize>()
    }
}

impl TensorArray for TestTensor {
    type DType = f32;

    fn schema(&self) -> &TensorSchema {
        &self.schema
    }

    fn dtype(&self) -> Self::DType {
        0.0
    }

    fn layout(&self) -> Layout {
        self.layout
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl TensorRead for TestTensor {
    fn read_value<'a>(&'a self, coord: &'a [u64]) -> BoxFuture<'a, fensor::Result<Self::DType>> {
        Box::pin(async move {
            self.validate_coord(coord)?;
            Ok(self.values[self.linear_offset(coord)])
        })
    }
}

impl TensorWrite for TestTensor {
    fn write_value<'a>(
        &'a self,
        coord: &'a [u64],
        value: Self::DType,
    ) -> BoxFuture<'a, fensor::Result<()>> {
        Box::pin(async move {
            self.validate_coord(coord)?;
            let offset = self.linear_offset(coord);

            // Interior mutability is not needed for these tests; clone/write/forget keeps test code minimal.
            let mut clone = self.values.clone();
            clone[offset] = value;

            let this = self as *const Self as *mut Self;
            unsafe {
                (*this).values = clone;
            }

            Ok(())
        })
    }
}

impl TensorTransform for TestTensor {
    fn reshape(mut self, shape: Shape) -> fensor::Result<Self> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = shape.iter().product();

        if old_size != new_size {
            return Err(Error::InvalidLayout(
                "reshape requires an equal number of elements".to_string(),
            ));
        }

        self.strides = contiguous_strides(&shape)?;
        self.shape = shape;
        self.base_offset = 0;
        Ok(self)
    }

    fn slice(mut self, range: Range) -> fensor::Result<Self> {
        if range.len() != self.shape.len() {
            return Err(Error::InvalidLayout(
                "slice range rank must match tensor rank".to_string(),
            ));
        }

        let mut next_shape = Shape::with_capacity(range.len());
        let mut next_strides = Strides::with_capacity(range.len());
        let shape = self.shape.clone();
        let strides = self.strides.clone();

        for (axis, (bound, dim)) in range.iter().zip(shape.iter()).enumerate() {
            match bound {
                AxisRange::In(start, stop, step)
                    if *step > 0 && *start <= *stop && *stop <= *dim =>
                {
                    self.base_offset += start * strides[axis];
                    next_shape.push((stop - start) / step);
                    next_strides.push(strides[axis] * step);
                }
                AxisRange::At(i) if *i < *dim => {
                    self.base_offset += i * strides[axis];
                    next_shape.push(1);
                    next_strides.push(strides[axis]);
                }
                _ => {
                    return Err(Error::InvalidLayout(format!(
                        "slice bound at axis {axis} is out of bounds"
                    )));
                }
            }
        }

        self.shape = next_shape;
        self.strides = next_strides;
        Ok(self)
    }

    fn transpose(mut self, permutation: Option<Axes>) -> fensor::Result<Self> {
        let shape = self.shape.clone();
        let base_strides = self.strides.clone();
        let ndim = shape.len();
        let axes = permutation.unwrap_or_else(|| (0..ndim).collect());

        if axes.len() != ndim {
            return Err(Error::InvalidLayout(
                "transpose permutation rank must match tensor rank".to_string(),
            ));
        }

        let mut seen = vec![false; ndim];
        for axis in &axes {
            if *axis >= ndim || seen[*axis] {
                return Err(Error::InvalidLayout(
                    "transpose permutation must be a valid axis permutation".to_string(),
                ));
            }
            seen[*axis] = true;
        }

        let mut shape = Shape::with_capacity(ndim);
        let mut strides = Strides::with_capacity(ndim);
        for axis in axes {
            shape.push(self.shape[axis]);
            strides.push(base_strides[axis]);
        }

        self.shape = shape;
        self.strides = strides;
        Ok(self)
    }
}

fn iter_coords(shape: &[usize]) -> Vec<Vec<u64>> {
    fn rec(shape: &[usize], out: &mut Vec<Vec<u64>>, prefix: &mut Vec<u64>, axis: usize) {
        if axis == shape.len() {
            out.push(prefix.clone());
            return;
        }

        for i in 0..shape[axis] {
            prefix.push(i as u64);
            rec(shape, out, prefix, axis + 1);
            prefix.pop();
        }
    }

    let mut out = Vec::new();
    rec(shape, &mut out, &mut Vec::new(), 0);
    out
}

fn seed_values(tensor: &TestTensor) {
    for coord in iter_coords(tensor.schema.shape()) {
        let value = (coord[0] * 100 + coord[1] * 10 + coord[2]) as f32;
        block_on(tensor.write_value(&coord, value)).expect("write value");
    }
}

fn transpose_range(range: &Range, permutation: &[usize]) -> Range {
    let mut remapped = Vec::with_capacity(permutation.len());
    for axis in permutation {
        remapped.push(range[*axis].clone());
    }

    remapped.into()
}

#[test]
fn accessor_coordinate_offset_and_read_write_dense() {
    let tensor = TestTensor::new(Layout::Dense);

    assert_eq!(tensor.linear_offset(&[1, 2, 3]), 23);

    block_on(tensor.write_value(&[1, 2, 3], 7.5)).expect("write");
    block_on(tensor.write_value(&[0, 0, 0], 1.25)).expect("write");

    let v_a = block_on(tensor.read_value(&[1, 2, 3])).expect("read");
    let v_b = block_on(tensor.read_value(&[0, 0, 0])).expect("read");
    let v_c = block_on(tensor.read_value(&[1, 1, 1])).expect("read");

    assert_eq!(v_a, 7.5);
    assert_eq!(v_b, 1.25);
    assert_eq!(v_c, 0.0);
}

#[test]
fn standalone_transpose_arbitrary_permutation() {
    let tensor = TestTensor::new(Layout::Dense);
    seed_values(&tensor);

    let perm = axes![2, 0, 1];
    let transposed = tensor
        .clone()
        .transpose(Some(perm.clone()))
        .expect("transpose");

    assert_eq!(transposed.shape(), &[4, 2, 3]);

    let mut inverse = vec![0usize; perm.len()];
    for (i, axis) in perm.iter().enumerate() {
        inverse[*axis] = i;
    }

    for t_coord in iter_coords(transposed.shape()) {
        let mut src = vec![0u64; t_coord.len()];
        for old_axis in 0..t_coord.len() {
            src[old_axis] = t_coord[inverse[old_axis]];
        }

        let expected = block_on(tensor.read_value(&src)).expect("read original");
        let actual = block_on(transposed.read_value(&t_coord)).expect("read transposed");
        assert_eq!(actual, expected, "coord {:?}", t_coord);
    }
}

#[test]
fn standalone_slice_range_selection() {
    let tensor = TestTensor::new(Layout::Dense);
    seed_values(&tensor);

    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(1, 3, 1),
        AxisRange::In(0, 4, 2)
    ];

    let sliced = tensor.clone().slice(r).expect("slice");
    assert_eq!(sliced.shape(), &[2, 2, 2]);

    for s_coord in iter_coords(sliced.shape()) {
        let src = vec![s_coord[0], s_coord[1] + 1, s_coord[2] * 2];
        let expected = block_on(tensor.read_value(&src)).expect("read original");
        let actual = block_on(sliced.read_value(&s_coord)).expect("read slice");
        assert_eq!(actual, expected, "coord {:?}", s_coord);
    }
}

#[test]
fn composition_transpose_slice_and_slice_transpose_consistency() {
    let tensor = TestTensor::new(Layout::Dense);
    seed_values(&tensor);

    let perm = axes![2, 0, 1];
    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(1, 3, 1),
        AxisRange::In(0, 4, 2)
    ];

    let left = tensor
        .clone()
        .slice(r.clone())
        .expect("slice")
        .transpose(Some(perm.clone()))
        .expect("transpose");

    let remapped = transpose_range(&r, &perm);
    let right = tensor
        .clone()
        .transpose(Some(perm))
        .expect("transpose")
        .slice(remapped)
        .expect("slice");

    assert_eq!(left.shape(), right.shape());

    for coord in iter_coords(left.shape()) {
        let left_v = block_on(left.read_value(&coord)).expect("left read");
        let right_v = block_on(right.read_value(&coord)).expect("right read");
        assert_eq!(left_v, right_v, "coord {:?}", coord);
    }
}

#[test]
fn dense_sparse_parity_for_supported_operations() {
    let dense = TestTensor::new(Layout::Dense);
    let sparse = TestTensor::new(Layout::Sparse { axis: Some(1) });

    let writes: HashMap<Vec<u64>, f32> = [
        (vec![0, 0, 0], 1.0),
        (vec![1, 2, 3], 9.0),
        (vec![1, 0, 1], 4.0),
        (vec![0, 2, 2], 7.0),
    ]
    .into_iter()
    .collect();

    for (coord, value) in &writes {
        block_on(dense.write_value(coord, *value)).expect("dense write");
        block_on(sparse.write_value(coord, *value)).expect("sparse write");
    }

    for coord in iter_coords(dense.shape()) {
        let d = block_on(dense.read_value(&coord)).expect("dense read");
        let s = block_on(sparse.read_value(&coord)).expect("sparse read");
        assert_eq!(d, s, "base coord {:?}", coord);
    }

    let permuted_dense = dense
        .clone()
        .transpose(Some(axes![1, 2, 0]))
        .expect("transpose");
    let permuted_sparse = sparse
        .clone()
        .transpose(Some(axes![1, 2, 0]))
        .expect("transpose");

    for coord in iter_coords(permuted_dense.shape()) {
        let d = block_on(permuted_dense.read_value(&coord)).expect("dense read");
        let s = block_on(permuted_sparse.read_value(&coord)).expect("sparse read");
        assert_eq!(d, s, "transposed coord {:?}", coord);
    }

    let r: Range = range![
        AxisRange::In(0, 2, 1),
        AxisRange::In(0, 3, 2),
        AxisRange::In(1, 4, 1)
    ];

    let sliced_dense = dense.clone().slice(r.clone()).expect("slice");
    let sliced_sparse = sparse.clone().slice(r).expect("slice");

    for coord in iter_coords(sliced_dense.shape()) {
        let d = block_on(sliced_dense.read_value(&coord)).expect("dense read");
        let s = block_on(sliced_sparse.read_value(&coord)).expect("sparse read");
        assert_eq!(d, s, "sliced coord {:?}", coord);
    }
}

#[tokio::test]
async fn public_create_rejects_invalid_sparse_axis_hint() {
    let (root, dir) = common::new_dir("invalid_sparse_axis").await;
    let schema = TensorSchema::new(DType::F32, shape![2, 3]).expect("schema");
    let result = fensor::Tensor::<common::FsEntry, f32>::create(
        dir,
        schema,
        Layout::Sparse { axis: Some(99) },
        1000,
    )
    .await;
    assert!(
        matches!(result, Err(Error::InvalidSchema(_))),
        "out-of-bounds sparse axis must be rejected"
    );
    common::cleanup(&root).await;
}

#[test]
fn public_schema_contiguous_strides_match_expected() {
    assert_eq!(
        contiguous_strides(&[2, 3, 4]).expect("strides").as_slice(),
        &[12, 4, 1]
    );
}

#[test]
fn sparse_incompatible_order_error_is_structured() {
    let tensor = TestTensor::new(Layout::Sparse { axis: None });

    let err = block_on(tensor.read_sparse_elements_in_order(
        range![
            AxisRange::In(0, 2, 1),
            AxisRange::In(0, 3, 1),
            AxisRange::In(0, 4, 1)
        ],
        axes![2, 0, 1],
    ))
    .expect_err("expected unsupported sparse order");

    match err {
        Error::UnsupportedSparseIterationOrder {
            requested_order,
            base_order,
            hint: _,
        } => {
            assert_eq!(requested_order, vec![2, 0, 1]);
            assert_eq!(base_order, vec![0, 1, 2]);
        }
        other => panic!("unexpected error variant: {other}"),
    }
}
