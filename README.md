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
`TensorTransform` implementations provide all seven transformations;
`TensorWriteBulk` implementations provide writes from buffers and fill. Each
implementation validates supported geometry and input cardinality.

Each consumed batch uses an explicit stack of evaluation frames. Elementwise
operations realize their bounded numerical result through ha-ndarray before
returning it to the parent; geometric projections forward their source result.
Reductions, matrix products, and Fourier transforms retain bounded batches, tiles,
or accumulators in the same process. They never persist computed intermediates.
Nested expressions can recompute values, and source reads/cache spill can do I/O.

`Tensor::copy_from(dir, &expression, max_capacity).await?` explicitly creates
independent filesystem storage. Copying is never necessary between operations.
The destination adapter can differ from the source adapter and must support the
output dtype. `max_capacity` limits storage-block capacity; it does not select an
exact block shape. Sparse copying uses the expression's reported layout and omits
final numerical zeros. Transformed owned expressions report scalar sparsity when
their original sparse-axis geometry no longer describes the output.

The [native storage contract](DESIGN.md#native-storage-and-sources) describes
logical-block access, source ownership, and replacement guarantees.

Logical shapes, strides, coordinates, and cardinalities use `u64`; axes use `usize`.
Import `Shape`, `Strides`, `Range`, `AxisRange`, and `Axes` from fensor.
`TensorGeometry::size()` returns `Result<u64>`; geometric flat offsets use checked
`i128`. Operation traits expose per-operation associated output types. Generic
callers state the required dtype bounds, and cast destinations may need annotations.

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

Numerical rules belong to ha-ndarray. Its
[numerical contract](https://github.com/TinyChain-Inc/ha-ndarray/blob/main/NUMERICS.md)
provides non-normative integration context for this standalone crate. In particular,
integer arithmetic wraps at its dtype width and integer division/remainder by zero return zero. Floats
retain backend NaN, infinity, signed-zero, and underflow behavior without domain
clamping. Logical operations return exactly u8 0/1: zero is false, and nonzero
values, including NaN, are true. Comparisons follow IEEE unordered-NaN rules.
Complex operations retain the backend's principal branches; complex predicates
inspect either component, and complex zero requires both components to be zero.
`mt` transposes without conjugation. Backend validation status belongs to
ha-ndarray, not this crate's test results.

## Sparse values

An absent sparse value is ordinary numerical zero. Dense and sparse expressions
use the same numerical kernels; there is no separate numerical support mask or
history of intermediate zeros. Each pointwise expression retains its typed
zero-background result so indexed reductions preserve backend signed-zero behavior.
Sparse output omits numerical zeros, including zeros produced by an expression.

Pointwise operations which would change the implicit zero background to a nonzero
value return `Error::WouldDensify { operation }`. For example, sparse `exp()` and
`add_scalar(1)` require explicit conversion first:

```rust
let dense = TensorExpression::new(source)?.into_dense();
let result = dense.exp().await?;
```

`into_dense()` is lazy: it declares dense output without copying storage or
changing existing clones. Evaluation then includes implicit zeros, so
`exp()` returns one at those coordinates. Both conditional branches are evaluated;
corruption in either branch propagates.

Ordered sparse reads emit logical row-major nonzero values, including transformed
views, and reject other orders with `UnsupportedSparseIterationOrder`. They sort
and deduplicate explicit selections for this reader only; geometric slicing retains
selection order and duplicates. Compatible geometry traverses occupied logical blocks
through paginated indexes. Eligible pointwise expressions delegate and merge these
candidates incrementally. Other mappings and aggregate operations retain bounded
logical evaluation; sparse storage alone does not guarantee work proportional to
nonzero values. See [slice traversal](DESIGN.md#slices-and-reductions).

## Reductions and matrix operations

Stored tensors support terminal reductions directly; axis reductions start from
a view, for example `tensor.view().sum(axes![1], false).await?`. Axes are sorted
and deduplicated, and invalid axes fail at construction. Empty axes reduce
singleton groups. `keepdims` retains reduced dimensions at extent one; removing
every axis produces `[1]`.

Terminal and axis reductions include every selected logical value, including
implicit sparse zeros. They have the same numerical meaning as reductions over
equivalent dense values; an all-zero sparse tensor has zero product and extrema.

`TensorStatistics` supplies mean, population standard deviation, and Euclidean norm,
both whole-tensor and by axis. Counts include implicit zeros. Real results use f64;
complex mean uses c64 and complex standard deviation and norm use f64.

Boolean terminals stop after a decisive consumed batch. Errors in that batch or
earlier propagate; later errors may remain unobserved and prefetched reads may
already have started. Numeric terminals consume all batches, including extrema.

Matrix multiplication requires rank ≥2, equal batch dimensions, and matching
contraction dimensions: `[..., M, K] @ [..., K, N] -> [..., M, N]`. Use explicit
broadcasting or casts to align operands. Products compose with every expression
family, including nested products. Dense and sparse operands share bounded tiles
and full logical contraction: zero-times-infinity can produce NaN and must not be
skipped. Floating results obey the backend aggregate accuracy contract, not bitwise
equivalence to a multiply/reduce expression.

Matrix-unary operations also require rank ≥2. `mt()` swaps the final two axes,
leaving batch axes and existing geometric write constraints intact. `diag()`
requires square final dimensions and returns a read-only view of shape
`[..., N]` from `[..., N, N]`:

```rust
let diagonal = tensor.view().mt().await?.diag().await?;
```

Only diagonal source coordinates contribute values. Transforms on a diagonal
view address its output, and further operations remain lazy.
Selected reads visit the requested diagonal coordinates; a complete sparse scan
still scales with logical diagonal length. Neither operation persists results.

With the `complex` feature, `mh()` composes matrix transpose and conjugation.
`fft()` and `ifft()` transform each last-axis group independently;
`fensor::fft::fft2(&view)` and `ifft2(&view)` compose transforms over the final
two axes. All are lazy and read-only. Transforms are unnormalized: inverse after
forward scales by the axis length, or the product of both lengths for 2D.

Each transformed axis must fit the execution limit in the
[bound table](DESIGN.md#bound-and-policy-constants); construction rejects longer
axes. Even a point read evaluates a complete axis group, including implicit sparse
zeros. Nested and 2D transforms may recompute groups; bounded memory does not imply
work proportional to nonzeros or optimal transform throughput.

## Streams, bounds, and concurrency

For `Layout::Sparse { axis: Some(a) }`, coordinates through axis `a` select a
dense trailing region. Physical chunks have extent one on that prefix and split
the full trailing shape into at most 4,096 elements each. `axis: None` selects
scalar sparsity. Chunk capacity does not change the logical dense region.
Loading validates the recorded geometry without retessellating it.

Sparse chunks are dense typed vectors in a native `b-table`, keyed by logical
block ID. Rows share native pages. Missing chunks are zero; all-zero chunks have
no row. Omission changes physical storage only: every logical coordinate still
has its ordinary numerical value.

One native ownership guard coordinates reads, replacement, reclamation, and
synchronization. Interrupted mutation invalidates the owner; callers coordinate
recovery.

Physical block lengths are validated against bounded metadata. Adapters remain
responsible for limiting decoding allocations before fensor receives a payload.

| Consumer | Delivery order |
|---|---|
| `read_blocks()` | Logical row-major values |
| `read_sparse_elements_in_order()` | Requested row-major sparse order |
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
evaluation adds no buffered streams or tasks. The driver suspends a parent until
its requested child completes, preserving operand order without recursive future
polling. Ready evaluation and occupied-candidate traversal yield cooperatively at
bounded work intervals, so cancellation can interrupt cached work. Synchronous
backend calls run on the polling thread and may use ha-ndarray's workers; `async move` does not create CPU parallelism.

Numeric terminal accumulation can depend on read scheduling, including extreme
overflow/underflow differences permitted by ha-ndarray's aggregate contract.
Wrapping integers, NaN extrema, and signed-zero rules are preserved. Unordered consumers promise no input-order error precedence.
Numeric terminals return the first observed error and drop pending evaluation.

Execution batches obey the [execution limit](DESIGN.md#bound-and-policy-constants).
Values, coordinates, and partial-result collections must not scale with total tensor, output, or reduction
group size. Caller-supplied values and explicit indices, rank-sized metadata,
cache/filesystem metadata, and independent consumers are separate memory costs.
This is not a total-process memory guarantee. The [execution design](DESIGN.md)
defines the bounds and private traversal mechanics. Callers may explicitly collect
streams when they want an in-memory result; fensor offers no whole-result collector.

## Storage, synchronization, and errors

Filesystem adapters implement `freqfs::FileLoad`/`FileSave` and expose `Vec<T>`,
`SparseNode<T>`, and `TensorMetadata<T>` through `AsType`. fensor requires
destream but prescribes no byte codec or whole-tensor transfer format. JSON and
TBON adapters are exercised in tests; applications own their format choices.

Metadata encodes shape, layout, sparse axis, and block shape. The adapter must preserve the Rust payload type
across reloads, for example with distinct tagged block/metadata variants per dtype.
Decoding untagged metadata as whichever dtype was requested violates this contract.
The [native storage contract](DESIGN.md#native-storage-and-sources) defines these native structures.
Adapters choose codecs and preserve exact typed scalar bits in sparse nodes; fensor
validates geometry and native structure without recovery or repair paths.

Call `tensor.sync().await?` before dropping and reopening storage. It writes blocks
and publishes the current sparse index root; syncing only the containing directory
does not publish that in-memory root. Exclude concurrent writes during sync.
Use `tensor.sync_all().await?` for delegated durable synchronization. Callers own
transaction and recovery policy.
fensor provides no commit, rollback, or isolation semantics.

Copying groups bounded destination updates and finishes each batch before requesting
the next source batch. It adds no lookahead or background work. Validation and errors
remain non-transactional: failures propagate and can leave partial output without
loadable metadata or a guaranteed write prefix. Synchronization and cleanup remain
caller responsibilities.

Element types need no destream implementation: adapters encode their payloads,
including any complex component representation. fensor's typed metadata remains
codec-neutral and encodes geometry only.

The freqfs cache accounts for typed retained allocation and applies admission backpressure
through eviction/spill. Configure cache capacity, minimum free disk space, and
admission wait in freqfs; oversized files, exhausted admission deadlines, and disk
failures return I/O errors. Bounded execution does not eliminate logical sparse
contraction scans, repeated nested evaluation, or cache-sensitive read amplification.
Completion-order copying can worsen cache locality and increase adapter traffic;
fewer delivery stalls do not guarantee faster copying. See [benchmark methodology
and interpretation](BENCHMARKS.md).

## External storage sources

`TensorSource` connects typed native or caller-provided logical blocks to the
same geometric views and expression evaluator. `StorageGeometry` describes bounded
logical tiles, and occupied-block streams use logical grid IDs. Views own their
source handles; `TensorExpression<T>` owns dynamically composed read-only values.
Borrowed reads and owned consuming streams share one bounded evaluator. Runtime
expression descriptions, evaluation frames, and live batch payloads have separate
checked admission limits, independently of logical tensor size; see the [bound table](DESIGN.md#bound-and-policy-constants).
Request and occupied-candidate traversal use explicit work lists, and final ownership release
detaches operands before draining them. Dropping a stream cancels active evaluation
and releases its retained source handles without following expression depth on the
worker stack. Physical payload and index mutation are private to the native owner.
Storage itself is never erased.
`replace_logical_block` coordinates native sparse
payload/index changes and preserves other logical blocks sharing native table pages.

`Tensor::load` is strict and non-mutating: missing metadata, blocks or indexes,
wrong payload types, invalid lengths, duplicate logical keys, and inconsistent
index coordinates are errors. Each native owner has separate creation and loading
operations; creation requires empty delegated storage. See the
[native storage and sources](DESIGN.md#native-storage-and-sources) for the external ownership boundary.

See [remaining work](ROADMAP.md), [execution design](DESIGN.md),
[benchmark reproduction](BENCHMARKS.md), and [test ownership](tests/COVERAGE.md).

Geometric `TensorView` values can plan bounded `BlockUpdates` from row-major value
batches with `plan_updates`, or consume an owned expression with `updates_from`.
The latter uses completion-order delivery and retains the source until consumption
or drop. These methods describe logical-block offsets; callers own write permission
and publication. Independent native materialization uses `Tensor::copy_from`.

`Tensor::from_values` consumes exactly one typed row-major value per dense element.
`Tensor::from_sparse_elements` consumes ordered unique coordinates for a sparse
layout, validating coordinates and omitting zeros. Both accept borrowed, non-`Unpin`
streams with an input error convertible from `fensor::Error`. They require empty
delegated storage and publish metadata only after successful construction; callers
own cleanup after errors or cancellation. Published replacement requires existing,
valid dense files and never creates or repairs missing payloads.
