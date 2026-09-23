# fensor
A filesystem-backed `Tensor` data structure featuring support for dense and sparse indexing

## Data Integrity Policy

`fensor` is fail-closed on corruption.

- `fensor` does not attempt to repair, recover, or auto-heal corrupted metadata or tensor data.
- If metadata or data is malformed, inconsistent, or unreadable, operations must return a structured error with a clear message.
- Recovery workflows (restore/rebuild/migration) are external operational concerns, not `fensor` runtime behavior.

## Storage and data types

fensor requires `destream` for typed serialization, but selects no byte codec.
Filesystem adapters implement `freqfs::FileLoad` and `FileSave` for their entry
type. freqfs saves and reloads that same entry, then checks the requested payload
through `AsType`. An adapter can use JSON, TBON, or another destream codec.
freqfs supplies no codec adapters or blanket I/O implementations. The test suite
uses explicit caller-owned TBON and JSON adapters.

Adapters support `Vec<T>`, `b_table::Node<u64>`, and `TensorMetadata<T>`.
`TensorMetadata<T>` contains logical shape, layout, and block shape. Its destream
representation contains geometry only; **the adapter must preserve the Rust
payload type across reloads**, for example with distinct tagged entry variants
for f32, f64, and u8 blocks and metadata. Decoding the same untagged metadata as
whichever `T` was requested does not meet this contract. The adapter also owns
format versioning and compatibility. fensor validates geometry and block lengths.

This replaces the former version-2 text metadata. Existing storage needs an
explicit adapter migration; fensor does not guess formats or fall back on errors.

There is no public whole-tensor wire codec. Applications can consume
`TensorRead::read_blocks` or `read_sparse_elements_in_order` to define transfer
formats, and use `Tensor::copy_from` to construct independent filesystem storage.

Before dropping and reopening a tensor, call `tensor.sync().await?` to write its
blocks and publish its current sparse index root. Syncing only the containing
`DirLock` does not publish an in-memory index root. Exclude concurrent tensor
writes during synchronization. This writes to filesystem buffers; callers own
subsequent durable directory synchronization and any transaction policy.

Use `Tensor<FE, u8>`, `Tensor<FE, f32>`, or `Tensor<FE, f64>` directly. Schema and
geometry dtype metadata use `number_general::NumberType` (also re-exported by
fensor), for example `NumberType::Float(FloatType::F32)` or
`NumberType::UInt(UIntType::U8)`. Unsupported and abstract number classes are
rejected when constructing a schema. `TensorElement` uses number-general's
primitive `DType` trait rather than maintaining its own dtype constants.
The associated `TensorGeometry::DType` still names the Rust element type;
`dtype()` returns its `NumberType` class.

There is no fensor dtype string codec. u8 stores all values from 0 to 255.

## Lazy math and bounded reads

`tensor.view().exp().await?`, `ln`, and `round` build reusable expressions over
`ha-ndarray` arrays. `TensorView` contains only coordinate geometry;
`UnaryView<Source, Op>` contains its immediate source and one typed operation.
`fensor::unary::{Exp, Ln, Round}` name the sealed operation types:
`round().exp()` produces `UnaryView<UnaryView<Source, Round>, Exp>`, following
`ha-ndarray`'s nested-access structure. `TensorUnary` uses
`ExpOutput`, `LnOutput`, and `RoundOutput` associated types while preserving the
borrowed `.exp().await?` call syntax. No `FE: Clone` bound is needed to clone
geometric, unary, or binary view descriptions.

`TensorAbs` adds `abs`; `TensorTrig` adds `sin`, `asin`, `sinh`, `cos`, `acos`,
`cosh`, `tan`, `atan`, and `tanh` for both f32 and f64. Import these traits alongside
`TensorUnary` to compose operations, for example
`tensor.view().abs().await?.sin().await?.round().await?`. Their sealed operation
markers are exported through `fensor::unary`, and every operation returns another
nested, read-only `UnaryView`. Absolute value preserves the stored dtype;
trigonometric methods have a distinct associated output type for each operation.

`TensorCast<f64>` adds lazy f32-to-f64 conversion. Import `TensorCast` and call
`tensor.view().cast().await?`, or use `TensorCast::<f64>::cast(&view).await?` to
name the target explicitly. The result is `UnaryView<Source, Cast<f64>>`; operations
before the cast execute in f32 and operations after it execute in f64. Widening
follows ha-ndarray conversion behavior, preserving finite f32 values exactly,
signed zero, infinities, and NaN classification (not a NaN payload guarantee).
Other casts are not yet supported.

Boolean operations return u8 views with values 0 or 1:

| Trait | Operations | Inputs |
| --- | --- | --- |
| `TensorUnaryBoolean` | `not` | u8, f32, f64 |
| `TensorNumeric` | `is_nan`, `is_inf` | f32, f64 |

For example, `tensor.view().is_nan().await?.not().await?` constructs a nested
mask expression. Dense `not` returns 1 for either signed zero and 0 for nonzeros,
including NaN and infinity. Numeric predicates follow ha-ndarray's classification.
All predicate views are read-only and use the same bounded consumers as numeric views.

Sparse predicates preserve original nonzero source support. Implicit zeros stay
absent even for `not`, so direct sparse `not` produces no populated output. A
chain `is_nan().not()` produces ones for finite nonzero values (and infinities),
while NaNs and implicit zeros produce zero. False intermediate results retain
support until the final consumer. Materializing between predicates drops those
zeros: `is_nan()` materialized before `not()` therefore behaves differently from
the unmaterialized sparse chain. No implicit densification is performed.

### Binary arithmetic

`TensorMath<Rhs>` constructs read-only `BinaryView<Left, Right, Op>` descriptions.
Operation markers are exported through `fensor::binary`.

| Methods | Operand and result types |
| --- | --- |
| `add`, `sub`, `mul`, `div`, `pow`, `rem` | matching f32, f64, or u8 |
| `log` (left value, right base) | matching f32 or f64 |

Use `left.view().add(&right.view()).await?`; either operand can also be a
computed view. Shapes must match at construction. Broadcasting is explicit:
`left.view().add(&right.view().broadcast(shape![2, 3])?).await?`.
Use the existing explicit f32-to-f64 cast when needed.

u8 addition, subtraction, multiplication, and exponentiation wrap modulo 256.
u8 division and remainder by zero return zero. Float results follow ha-ndarray,
including NaN, infinity, signed zero, gradual underflow, and cast behavior as
specified in [ha-ndarray's numerical contract](../ha-ndarray/NUMERICS.md).
fensor delegates numerical evaluation to that backend; its source-support rules
below are a separate storage/expression contract.
Backend validation uses certified MPFR/MPC references and exact aggregate
references. Native and CPU-OpenCL results do not replace ha-ndarray's pending
actual-GPU conformance gate.

Binary support is the union of its operands' source support. Dense leaves support
every coordinate; sparse leaves support their original nonzero values. Unary
nodes preserve that support, including intermediate zeros. Unsupported child
coordinates are masked to zero with lazy ndarray selections before the parent
operation. A binary view is sparse (with no axis hint) only when both operands
are sparse; otherwise it is dense.

For sparse `a`, `(a - a).exp()` returns one on a's support and zero elsewhere.
Dividing those retained zero results by themselves yields NaN on that support,
but coordinates absent from both inputs remain absent. These rules apply to
all seven operations, including power and logarithm. Final zeros are omitted
from sparse output; copying to storage establishes a new support boundary.

Chaining does not read data or write intermediate tensors.
Consumers read each batch directly from the geometric leaves, recursively construct
one ndarray expression through nested unary and binary views, and evaluate only its
final result. `UnaryOp<Input>::Output` and its input dtype are independent,
so a cast does not require an intermediate buffer. Intermediate views do not
evaluate buffers or filter sparse support.
Backend execution and fusion remain `ha-ndarray`'s responsibility.
`TensorElement` extends `ha-ndarray::Number`; the supported stored types remain
`u8`, `f32`, and `f64`.

`TensorRead::read_blocks()` returns logical row-major batches of values,
independent of physical storage tiling. Each call creates a fresh stream with
at most 4096 elements per batch and `num_cpus::get().max(1)` concurrent batches,
using `StreamExt::buffered` to preserve ordering. No work is spawned
in the background by a read stream: a stalled consumer stops further polling,
and dropping the stream drops pending reads. Independent streams can be consumed
concurrently, and recompute their own results; they do not share a mutable cursor.
These are live views, not snapshots: concurrent source writes are not isolated.

Direct reads, block streams, and `Tensor::copy_from` evaluate the same expression.
`Tensor::copy_from(dir, &expression, max_capacity).await?` constructs independent
filesystem-backed storage from any `TensorRead`, including base tensors, geometric
views, and computed views. Evaluation happens implicitly as the constructor
consumes bounded batches; views have no separate materialization method.
`max_capacity` controls destination storage blocks independently. The destination
file-entry type must support the output dtype and may differ from the source's.
Encoded and copied schemas use the expression's output dtype, with existing
formats unchanged. Sparse copies reset the axis hint to `None` and omit final zeros.
Source read and destination write errors propagate; cleanup of partial output
remains the caller's responsibility. Readers returning too many or too few values
for their shape return a structured layout error.

`UnaryView` and `BinaryView` do not implement `TensorWrite`, so writes through a computed view
are rejected at compile time. Geometric views retain their existing write-through
constraints. `TensorTransform` still returns `Self`: transforms update the
geometric leaves and retain typed operation order. Slicing and transposition compose with
expressions on both dense and sparse tensors. Scalar (rank-zero) views cannot
be streamed or copied and return a structured schema error.

Sparse unary operations act only on nonzero source values; implicit zeros remain
zero, even for `exp`, `ln`, `cos`, `acos`, and `cosh`. Thus a populated `0.2`
under `round().cos()` yields `1`, while an implicit zero stays absent. Operations
preserve backend NaN and infinity results without domain clamping. A chain retains
its original input support across casts and until
consumption: a populated `0.2` produces `1` under `round().exp()`, while an absent
coordinate stays zero. Materializing `round()` first drops that zero from sparse
support, so a subsequent `exp()` on the stored result leaves it zero. A final
zero is omitted from sparse output. New sparse materializations reset the axis
hint to `None`. Ordered sparse reads support logical row-major order, including
transformed views, and reject other orders with `UnsupportedSparseIterationOrder`.
They scan only the selected logical range in bounded batches, with explicit index
selections sorted and deduplicated. Full-range reads still scale with logical size,
not stored support; very sparse, large shapes can therefore be slow. Index-driven
traversal remains future work.

Filesystem blocks are charged to the `freqfs` cache by their payload byte size.
Point reads borrow cached blocks and point writes modify them in place, avoiding
whole-block copies. With local `freqfs` 0.13, cache admission awaits eviction and
spill; an oversized file, exhausted admission deadline, or disk failure returns
an I/O error. Configure the cache size, minimum free disk space, and admission
wait when constructing `freqfs::Cache`.

The fixed batch size and CPU-based concurrency bound execution batches, not total process RSS. Budget separately
for cache contents, filesystem/index metadata, coordinate buffers (batch size ×
rank), ndarray execution temporaries, and each concurrent consumer. In particular,
`read_all`/`read_values`, collecting a stream, and sparse compaction are allocating
APIs and are not covered by the bounded streaming contract. No blanket OOM
immunity is promised for cache budgets or collecting an entire tensor.
