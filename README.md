# fensor
A filesystem-backed `Tensor` data structure featuring support for dense and sparse indexing

## Data Integrity Policy

`fensor` is fail-closed on corruption.

- `fensor` does not attempt to repair, recover, or auto-heal corrupted metadata or tensor data.
- If metadata or data is malformed, inconsistent, or unreadable, operations must return a structured error with a clear message.
- Recovery workflows (restore/rebuild/migration) are external operational concerns, not `fensor` runtime behavior.

## Serializing a tensor

`Tensor<FE, T>` has two, independent wire-format surfaces:

- **Schema-only (`Tensor: ToStream`/`FromStream`/`IntoStream`)**: encodes just the `TensorSchema` (dtype/shape/layout). Only a base (identity-view) tensor can be encoded this way; encoding a transformed (sliced/transposed/reshaped) view is rejected, since views are metadata-only and never persisted. Decoding always builds a fresh, empty base tensor at the given directory via `Tensor::create` — no element data is carried.
- **View + data streaming (`Tensor::view_encoder` / `TensorViewDecoder`)**: `tensor.view_encoder()` returns a `TensorViewEncoder<'_, View>` that streams the tensor's *current* view — identity or transformed, dense or sparse — directly to the wire via `destream`'s `ToStream`/`IntoStream` contract. The encoder lazily reads from the tensor's filesystem-backed storage and emits values one at a time, with no full in-memory buffering; only non-default (nonzero) values are transmitted, reducing network traffic for sparse-heavy or mostly-empty tensors. On the receiving end, `TensorViewDecoder<FE, T>` implements `destream`'s `FromStream` and writes each arriving value directly to a fresh, independent, identity base tensor's filesystem storage as it arrives off the wire — also with no full in-memory buffering. There is no trailer or checksum on this wire format: a successful transfer is signaled by natural exhaustion of the pairs sequence, and end-to-end transfer completeness/integrity is left to the transport/caller layer rather than re-implemented here. A read failure on the sending side propagates as a bounded error-code sentinel — a closed classification of which failure shape occurred, deliberately excluding free-text detail (filesystem paths, coordinates, or other sender-machine specifics) since this wire format is meant to cross a machine boundary. On that sentinel, a malformed/truncated stream, or any other decode-time failure after the destination storage is created, the directory is truncated and deleted before a fail-closed error is returned. Call `.into_inner()` on the decoder to extract the reconstructed tensor. Like the design it replaces, this produces a fresh, independent identity base tensor with no link back to the source storage; it remains a distinct, additive wire surface alongside the existing schema-only `Tensor: ToStream/FromStream/IntoStream` contract, which is completely unrelated and still only carries dtype/shape/layout metadata without element data.

## Lazy math and bounded reads

`tensor.view().exp().await?`, `ln`, and `round` build reusable expressions over
`ha-ndarray` arrays. `TensorView` contains only coordinate geometry;
`UnaryView<Source, Op>` contains a geometric source and a typed operation.
`fensor::unary::{Exp, Ln, Round, Then}` name the sealed operation types:
`round().exp()` produces `UnaryView<Source, Then<Round, Exp>>`, retaining one
source rather than wrapping an evaluated intermediate view. `TensorUnary` uses
`ExpOutput`, `LnOutput`, and `RoundOutput` associated types while preserving the
borrowed `.exp().await?` call syntax. No `FE: Clone` bound is needed to clone
geometric or unary view descriptions.

Chaining does not read data or write intermediate tensors.
Each consumed batch constructs an ndarray expression and evaluates its final
result; backend execution and fusion remain `ha-ndarray`'s responsibility.
`TensorElement` extends `ha-ndarray::Number`; the supported stored types remain
`f32` and `f64`.

`TensorRead::read_blocks()` returns logical row-major batches of values,
independent of physical storage tiling. Each call creates a fresh stream with
at most 4096 elements per batch and `num_cpus::get().max(1)` concurrent batches,
using `StreamExt::buffered` to preserve ordering. No work is spawned
in the background by a read stream: a stalled consumer stops further polling,
and dropping the stream drops pending reads. Independent streams can be consumed
concurrently, and recompute their own results; they do not share a mutable cursor.
These are live views, not snapshots: concurrent source writes are not isolated.

Direct reads, view serialization, and `materialize` evaluate the same expression.
Both view families expose `view_encoder()`, returning `TensorViewEncoder<'_, View>`;
its generic view parameter replaces the previous storage/lifetime parameters.
The wire format and decoder are unchanged.
`materialize(dir, max_capacity)` uses the same bounded read stream;
`max_capacity` controls destination storage blocks independently. Materialization
writes only the final tensor, awaits storage writes, and propagates read/write
errors. Partial output after failure remains the caller's lifecycle responsibility.
`UnaryView` does not implement `TensorWrite`, so writes through a computed view
are rejected at compile time. Geometric views retain their existing write-through
constraints. `TensorTransform` still returns `Self`: transforms update the
geometric source and retain the typed unary composition. Slicing and transposition compose with unary
operations on both dense and sparse tensors. Scalar (rank-zero) views cannot
be streamed or materialized and return a structured schema error.

Sparse unary operations act only on nonzero source values; implicit zeros remain
zero, even for `exp` and `ln`. A chain retains its original input support until
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
