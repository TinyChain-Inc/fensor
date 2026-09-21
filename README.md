# fensor
A filesystem-backed `Tensor` data structure featuring support for dense and sparse indexing

## Data Integrity Policy

`fensor` is fail-closed on corruption.

- `fensor` does not attempt to repair, recover, or auto-heal corrupted metadata or tensor data.
- If metadata or data is malformed, inconsistent, or unreadable, operations must return a structured error with a clear message.
- Recovery workflows (restore/rebuild/migration) are external operational concerns, not `fensor` runtime behavior.

## Serializing a tensor

`tensor.view_encoder()` (or a view's `view_encoder()`) streams schema information
and nonzero coordinate/value pairs through `destream`. `TensorViewDecoder<FE, T>`
writes them into a fresh filesystem-backed tensor; call `.into_inner()` to obtain
it. Geometric and computed views use the same format. Read and decode errors are
propagated; cleanup of partial destination storage remains the caller's responsibility.
There is no trailer or checksum, so end-to-end completeness belongs to the transport.

`DType` and `Layout` have standalone stream encodings. `Tensor` and `TensorSchema`
do not expose a separate schema-only stream API; persistent schema metadata is
written and loaded by the storage implementation.

Native stored dtypes are `u8`, `f32`, and `f64`, exposed through `TensorU8`,
`TensorF32`, and `TensorF64`. u8 stores all values from 0 to 255, not just boolean
masks. The new `DType::U8` uses the string `"u8"`; metadata version and wire
structure are unchanged, and existing float data requires no migration. Older
readers reject the new dtype. Downstream exhaustive matches on `DType` must add
its `U8` variant. Storage adapters for u8 need `AsType<Vec<u8>>` support.

## Lazy math and bounded reads

`tensor.view().exp().await?`, `ln`, and `round` build reusable expressions over
`ha-ndarray` arrays. `TensorView` contains only coordinate geometry;
`UnaryView<Source, Op>` contains its immediate source and one typed operation.
`fensor::unary::{Exp, Ln, Round}` name the sealed operation types:
`round().exp()` produces `UnaryView<UnaryView<Source, Round>, Exp>`, following
`ha-ndarray`'s nested-access structure. `TensorUnary` uses
`ExpOutput`, `LnOutput`, and `RoundOutput` associated types while preserving the
borrowed `.exp().await?` call syntax. No `FE: Clone` bound is needed to clone
geometric or unary view descriptions.

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

Chaining does not read data or write intermediate tensors.
Consumers read each batch from the root geometric view, recursively construct
one ndarray expression through the nested unary views, and evaluate only its
final result. `UnaryOp<Input>::Output` and the root input dtype are independent,
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

Direct reads, view serialization, and `Tensor::copy_from` evaluate the same expression.
Both view families expose `view_encoder()`, returning `TensorViewEncoder<'_, View>`;
its generic view parameter replaces the previous storage/lifetime parameters.
The wire format and decoder are unchanged.
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

`UnaryView` does not implement `TensorWrite`, so writes through a computed view
are rejected at compile time. Geometric views retain their existing write-through
constraints. `TensorTransform` still returns `Self`: transforms update the
geometric source and retain the typed unary composition. Slicing and transposition compose with unary
operations on both dense and sparse tensors. Scalar (rank-zero) views cannot
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
