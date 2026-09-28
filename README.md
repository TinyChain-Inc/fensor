# fensor

Filesystem-backed tensors with dense and sparse storage, lazy arithmetic, and
bounded streaming execution.

## Composition and consumption

Chaining operations constructs typed view descriptions, following ha-ndarray's
composition model. Awaiting a view-construction method does not evaluate or
persist its result. Import the relevant operation traits to use their methods.
The [checked crate example](src/lib.rs) composes matrix multiplication, scalar
addition, and exponentiation, then consumes coordinate-bearing batches without
creating result storage.

`Tensor<FE, T>` owns storage; `tensor.view()` creates a geometric `TensorView`.
Unary, binary, conditional, reduction, and matrix expressions remain lazy and
read-only. Computed views implement neither `TensorWrite` nor `TensorArray`;
geometric views retain constrained write-through access. View descriptions can
be cloned without requiring the filesystem adapter to implement `Clone`.

Elementwise chains build nested ha-ndarray expressions over each batch.
Reductions and matrix products introduce bounded **in-memory evaluation
boundaries**, retaining batches, tiles, or accumulators. They never persist
computed intermediates. This does not promise a single fused backend operation:
nested expressions can recompute values, and source reads/cache spill can do I/O.

`Tensor::copy_from(dir, &expression, max_capacity).await?` explicitly creates
independent filesystem storage. Copying is never necessary between operations.
The destination adapter can differ from the source adapter and must support the
output dtype. `max_capacity` limits storage-block capacity; it does not select an
exact block shape. Sparse copies omit final zeros and reset the axis hint to `None`.

## Supported operations

Stored types are `u8/u16/u32/u64`, `i8/i16/i32/i64`, and `f32/f64`.
The optional `complex` feature adds `fensor::complex::{Complex32, Complex64}`,
with f32 and f64 components respectively. Boolean outputs use u8, which accepts
the full range 0–255; native bool and abstract number classes are not storage types. Schema
and geometry metadata use number-general's `NumberType`, re-exported by fensor.
`TensorGeometry::DType` names the Rust element type; `dtype()` returns its class.
Unsupported or abstract number classes are rejected during schema construction.

| Trait | Operations | Input → output |
|---|---|---|
| `TensorUnary` | `exp`, `ln`; `round` | Float/complex → same dtype; round is real-float only |
| `TensorAbs` | `abs` | Real → same dtype; complex → component real dtype |
| `TensorTrig` | `sin`, `asin`, `sinh`, `cos`, `acos`, `cosh`, `tan`, `atan`, `tanh` | Float/complex → same dtype |
| `TensorCast<To>` | `cast` | Any supported concrete dtype → any supported concrete dtype |
| `TensorComplex` | `conj`; `re`, `im`, `angle` | Complex → complex; complex → component real dtype |
| `TensorUnaryBoolean` | `not` | Any supported dtype → u8 |
| `TensorNumeric` | `is_nan`, `is_inf` | Float/complex → u8 |
| `TensorMath`, `TensorMathScalar` | `add`, `sub`, `mul`, `div`, `pow`; `_scalar` variants | Matching supported dtypes → same dtype |
| `TensorMath`, `TensorMathScalar` | `rem`, `rem_scalar`; `log`, `log_scalar` (value, base) | Matching real dtypes; matching float/complex dtypes |
| `TensorCompare`, `TensorCompareScalar` | `eq`, `ne`; `gt`, `ge`, `lt`, `le`; `_scalar` variants | Matching supported dtypes; ordering requires real inputs → u8 |
| `TensorBoolean`, `TensorBooleanScalar` | `and`, `or`, `xor`; `_scalar` variants | Matching supported dtypes → u8 |
| `TensorWhere` | `condition.cond(&then, &or_else)` | u8 condition; matching branches → branch dtype |
| `TensorReduceAll` | `sum_all`, `product_all`; `min_all`, `max_all` | Any supported dtype; extrema require real inputs → same scalar dtype |
| `TensorReduceBoolean` | `all`, `any` | Any supported dtype → bool |
| `TensorReduce` | `sum`, `product`; `min`, `max` | Any supported dtype; extrema require real inputs → same dtype |
| `TensorMatrixUnary` | `mt`, `diag` | Any supported dtype → same dtype |
| `TensorMatrixUnaryComplex` | `mh` | Complex → same dtype |
| `TensorFourier`; `fft::fft2`, `fft::ifft2` | Last-axis and final-two-axis Fourier transforms | Complex → same dtype; bounded axis lengths |
| `TensorMatMul` | `matmul` | Matching supported dtypes → same dtype |

Arithmetic methods borrow operands and are asynchronous. Elementwise tensor
operands must have identical shapes and dtypes; scalar arguments match the source
dtype. Broadcasting and casts are explicit, including identity and narrowing casts.
Select the destination through the result type or `TensorCast::<To>::cast(&view)`;
operations before and after a cast execute in their respective dtypes.
Conversions follow number-general's pipeline, including intermediate widths:
for example, -1i8 → u64 yields 255, while -1f64 → u8 yields zero.
Unary composition nests sources, for example
`UnaryView<UnaryView<Source, Round>, Exp>`. Public operation markers live in
`unary`, `binary`, `scalar`, and `reduce`.

`TensorTransform` provides reshape, broadcast, flip, slice, squeeze, transpose,
and unsqueeze, returning `Self` subject to geometric validation. Elementwise
transforms preserve operation order. Reduction and matrix output transforms map
the output geometry instead of moving through the aggregate. Rank-zero views
cannot be streamed or copied. Empty-dimension tensor storage is unsupported.

Numerical rules belong to [ha-ndarray](../ha-ndarray/NUMERICS.md). In particular,
integer arithmetic wraps at its dtype width and integer division/remainder by zero return zero. Floats
retain backend NaN, infinity, signed-zero, and underflow behavior without domain
clamping. Logical operations return exactly u8 0/1: zero is false, and nonzero
values, including NaN, are true. Comparisons follow IEEE unordered-NaN rules.
Complex operations retain the backend's principal branches; complex predicates
inspect either component, and complex zero requires both components to be zero.
`mt` transposes without conjugation. Backend validation status belongs to
ha-ndarray, not this crate's test results.

## Sparse support

Dense leaves support every coordinate; sparse leaves support stored nonzero
values. Expressions carry support independently of current numerical values.
Final zeros are omitted only from sparse output. Copying into sparse storage
establishes a new support boundary from stored nonzeros.

| Expression | Retained support |
|---|---|
| Unary, cast, scalar operation | Source support |
| Binary arithmetic, comparison, boolean | Union of both sources |
| Conditional | Union of condition and both branches |
| Axis reduction | Output group supported when any input in that group is supported |
| Matrix product | Output supported when either operand is supported at any contraction position |

Unsupported child values contribute zero before their parent operation. Unary
and scalar operations do not populate implicit zeros, even for `exp`, `cos`,
`add_scalar(1)`, or `eq_scalar(0)`. A populated `0.2` under `round().exp()` yields
`1`; an implicit zero stays absent. Copying `round()` first loses that support.
Direct sparse `not` emits no populated output, while `is_nan().not()` retains
ones for finite nonzero values and infinities. Intermediate false results retain
support just as intermediate numeric zeros do.

Binary expressions are sparse only when both operands are sparse. For sparse
`a`, `(a - a).exp()` is one on a's support and absent elsewhere; dividing the
retained zeros by themselves yields NaN only on that support. A conditional is
sparse only when all three inputs are sparse. An absent condition selects the
else branch; an unselected branch still contributes support. Both branches are
evaluated, so corruption in either branch propagates.

Ordered sparse reads support logical row-major order, including transformed
views, and reject other orders with `UnsupportedSparseIterationOrder`. They
scan the selected logical range, sorting and deduplicating explicit selections
for this reader only. Geometric slicing retains selection order and duplicates.
Full-range ordered sparse reads still scale with logical size, not stored support.
Eligible numeric reductions can use occupied-index traversal instead; see
[slice traversal](DESIGN.md#slices-and-reductions).

## Reductions and matrix operations

Stored tensors support terminal reductions directly; axis reductions start from
a view, for example `tensor.view().sum(axes![1], false).await?`. Axes are sorted
and deduplicated, and invalid axes fail at construction. Empty axes reduce
singleton groups. `keepdims` retains reduced dimensions at extent one; removing
every axis produces `[1]`.

Sparse reductions exclude implicit zeros but include supported intermediate zeros.
An empty axis group stays absent for every operation, including product and
extrema. Whole-tensor empty-support results are sum `0`, product `1`, `all=true`,
and `any=false`; `min_all` and `max_all` return `Error::Unsupported`. Consequently,
sparse reductions can differ from reductions over equivalent dense values.

Boolean terminals stop after a decisive consumed batch. Errors in that batch or
earlier propagate; later errors may remain unobserved and prefetched reads may
already have started. Numeric terminals consume all batches, including extrema.

Matrix multiplication requires rank ≥2, equal batch dimensions, and matching
contraction dimensions: `[..., M, K] @ [..., K, N] -> [..., M, N]`. Use explicit
broadcasting or casts to align operands. Products compose with every expression
family, including nested products. A supported row times an absent sparse column
has supported zero outputs; `exp()` can turn those into ones. A wholly absent row
and column remain absent. Supported zero-times-infinity can produce NaN and must
not be skipped. Floating results obey the backend aggregate accuracy contract,
not bitwise equivalence to a multiply/reduce expression.

Matrix-unary operations also require rank ≥2. `mt()` swaps the final two axes,
leaving batch axes and existing geometric write constraints intact. `diag()`
requires square final dimensions and returns a read-only view of shape
`[..., N]` from `[..., N, N]`:

```rust,ignore
let diagonal = tensor.view().mt().await?.diag().await?;
```

Only diagonal source coordinates contribute support, including supported
intermediate zeros; off-diagonal values do not populate the result. Transforms
on a diagonal view address its output, and further operations remain lazy.
Selected reads visit the requested diagonal coordinates; a complete sparse scan
still scales with logical diagonal length. Neither operation persists results.

With the `complex` feature, `mh()` composes matrix transpose and conjugation.
`fft()` and `ifft()` transform each last-axis group independently;
`fensor::fft::fft2(&view)` and `ifft2(&view)` compose transforms over the final
two axes. All are lazy and read-only. Transforms are unnormalized: inverse after
forward scales by the axis length, or the product of both lengths for 2D.

Each transformed axis must fit the execution limit in the
[bound table](DESIGN.md#bound-and-policy-constants); construction rejects longer
axes. Even a point read evaluates a complete axis group. Sparse absent inputs
contribute zero, while any supported input supports every frequency in its group,
including results that cancel to zero. Completely empty groups remain absent.
Nested and 2D transforms may recompute groups; bounded memory does not imply
optimal transform throughput.

## Streams, bounds, and concurrency

| Consumer | Delivery order |
|---|---|
| `read_blocks()` | Logical row-major values |
| `read_sparse_elements_in_order()` | Requested supported row-major sparse order |
| `read_coordinate_blocks()` | Completion-dependent batches of paired coordinates/values |
| Whole-tensor numeric terminals | Accumulate batches in completion order |
| Boolean terminals | Logical order with short-circuiting |

A successful complete coordinate stream visits every logical coordinate exactly
once, including zeros. Its default adapter pairs row-major values with coordinates;
built-in expressions can generate tiled requests. Request traversal is distinct
from batch delivery order. Each stream is independent and live, not a snapshot.
Concurrent source writes are not isolated. Dropping a stream cancels pending work.

**ha-ndarray owns numerical parallelism; fensor owns bounded async concurrency.**
The outer consumer keeps at most `num_cpus::get().max(1)` batch futures in flight,
using `buffered` for ordered consumers and `buffer_unordered` otherwise. Inner
evaluation adds no buffered streams. Synchronous backend calls run on the polling
thread and may use ha-ndarray's workers; `async move` does not create CPU parallelism.

Numeric terminal accumulation can depend on read scheduling, including extreme
overflow/underflow differences permitted by ha-ndarray's aggregate contract.
Wrapping integers, NaN extrema, signed-zero rules, and empty-support identities
are preserved. Unordered consumers promise no input-order error precedence.
Numeric terminals return the first observed error and drop pending evaluation.

Execution batches obey the [execution limit](DESIGN.md#bound-and-policy-constants).
Values, coordinates, masks, and
partial-result collections must not scale with total tensor, output, or reduction
group size. Caller-supplied values and explicit indices, rank-sized metadata,
cache/filesystem metadata, and independent consumers are separate memory costs.
This is not a total-process memory guarantee. The [execution design](DESIGN.md)
defines the bounds and private traversal mechanics. Callers may explicitly collect
streams when they want an in-memory result; fensor offers no whole-result collector.

## Storage, synchronization, and errors

Filesystem adapters implement `freqfs::FileLoad`/`FileSave` and expose `Vec<T>`,
`b_table::Node<u64>`, and `TensorMetadata<T>` through `AsType`. fensor requires
destream but prescribes no byte codec or whole-tensor transfer format. JSON and
TBON adapters are exercised in tests; applications own their format choices.

Metadata encodes geometry only. The adapter must preserve the Rust payload type
across reloads, for example with distinct tagged block/metadata variants per dtype.
Decoding untagged metadata as whichever dtype was requested violates this contract.
Adapters own format versions and migrations; fensor validates geometry and block
lengths and fails closed on malformed input, without recovery or repair paths.

Call `tensor.sync().await?` before dropping and reopening storage. It writes blocks
and publishes the current sparse index root; syncing only the containing directory
does not publish that in-memory root. Exclude concurrent writes during sync.
Callers own subsequent durable directory synchronization and transaction policy.
fensor provides no commit, rollback, or isolation semantics.

Copying groups bounded destination updates and overlaps one update with one source
lookahead using `try_join!`, without spawning tasks. Destination batches remain
sequential. Validation/error handling is non-transactional: failures propagate and
can leave partial output, with no guaranteed write prefix or concurrent-error
precedence. Synchronization and cleanup remain caller responsibilities.

Element types need no destream implementation: adapters encode their payloads,
including any complex component representation. fensor's typed metadata remains
codec-neutral and encodes geometry only.

The freqfs cache accounts for block payload bytes and applies admission backpressure
through eviction/spill. Configure cache capacity, minimum free disk space, and
admission wait in freqfs; oversized files, exhausted admission deadlines, and disk
failures return I/O errors. Bounded execution does not eliminate logical sparse
contraction scans, repeated nested evaluation, or cache-sensitive read amplification.
Completion-order copying can worsen cache locality and increase adapter traffic;
fewer delivery stalls do not guarantee faster copying. See [benchmark methodology
and interpretation](BENCHMARKS.md).

## Compatibility notes

- The default build supports ten real dtypes; enable `complex` for the two native
  complex types. Existing metadata and adapter formats are unchanged; adapters
  must explicitly support new payload types.
- `TensorElement` no longer requires value serialization traits. `TensorAbs`
  reports the backend absolute-value dtype, and real-only operation bounds now
  apply to individual methods/associated outputs. Generic callers must state
  these bounds where needed. Cast destinations may now need an explicit annotation.

- Logical geometry now uses `u64`: import `Shape`, `Strides`, `Range`, and
  `AxisRange` from fensor, and use the re-exported `Axes` for axis identifiers.
  `TensorGeometry::size()` returns `Result<u64>`. Redundant shape/stride aliases
  are removed; `TensorView::flat_offset` now returns `Result<i128>`. Adapter
  coordinate payloads and persisted geometry are unchanged.

- Typed adapter-owned metadata replaces the former version-2 text representation.
  Existing data in that representation requires an explicit adapter migration;
  fensor does not guess formats. There is no fensor dtype-string or whole-tensor codec.
- Use `Tensor<FE, T>` and `NumberType` directly; per-dtype tensor aliases are removed.
- Computed views have no `materialize` method; use `Tensor::copy_from` explicitly.
  Scalar and axis-reduction traits return per-operation associated read outputs.
- `TensorReadBulk`, `read_all`, `read_values`, and `Tensor::compact_sparse` are removed
  without forwarding aliases. Consume streams explicitly. `TensorWriteBulk` still
  accepts caller-owned buffers and validates cardinality before mutation.

See [remaining work](ROADMAP.md), [execution design](DESIGN.md),
[benchmark reproduction](BENCHMARKS.md), and [test ownership](tests/COVERAGE.md).
