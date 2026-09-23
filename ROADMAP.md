# fensor roadmap

> **Non-normative:** this file tracks unimplemented work and cannot override
> this repository's implemented behavior or local contracts.

## Boundary contract (non-transactional core)

`fensor` is a filesystem-backed tensor storage/index primitive, not a transaction manager.

- `fensor` responsibilities:
  - Persist tensor metadata and blocks (dense and sparse).
  - Provide deterministic read/write/merge primitives over blocks and indices.
  - Expose operations that can be composed into canonical+delta merge flows.
- `fensor` non-goals:
  - No `txn_id` ownership or lifecycle management.
  - No commit/rollback/finalize orchestration.
  - No transaction visibility or isolation policy.

Transactional orchestration belongs to `tc-collection`, which composes `fensor` as the storage engine.

## Data integrity contract (fail-closed)

`fensor` must treat corruption as a hard error, not a recoverable condition.

- No in-place auto-repair of metadata or block files.
- No alternate-source read path for corrupted metadata/data.
- Corruption must surface as a structured, user-facing error describing the failed file/field.
- Any recovery path (restore from backup, rebuild index, re-materialize data) is owned by external tooling and/or `tc-collection` orchestration, not by `fensor`.

## Implementation phases

1. **Phase 1: Metadata and coordinate engine.**
   - Define metadata schema fields: `dtype`, `shape`, `layout`, `block_shape`, `strides`, and `axis: Option<usize>` for sparse mode hints.
   - Implement canonical coordinate mapping (`logical <-> physical <-> block-local`).
   - Exit criteria: deterministic coordinate mapping tests pass for permutation and slicing projection.

2. **Phase 2: Dense block backend.**
   - Implement filesystem block addressing and dense read/write paths.
   - Implement dense `slice`/`permute`/`transpose` as view-first operations.
   - Exit criteria: dense round-trip and transform parity tests pass.

3. **Phase 3: Sparse indexing backend.**
   - Define `b-table` index schemas for a single `Sparse` layout, with optional axis-optimized indexing when `axis: Option<usize>` is set.
   - Implement sparse read/write semantics (missing row resolves to zero/default).
   - Exit criteria: sparse correctness tests pass, including zero-overwrite lifecycle behavior (tombstone/delete/retained-row policy is explicit and tested).

4. **Phase 4: Unified transform planner.**
   - Route `slice`/`permute`/`transpose` through one layout-agnostic planner.
   - Add layout-specific execution plans (dense block walk versus index scan).
   - Exit criteria: transform parity across Dense and Sparse layouts, plus explicit support/error boundaries for sparse ordered iteration under incompatible slice+permutation orderings.

5. **Phase 5: Conversion and materialization.**
   - Add conversion/materialization APIs across `Dense <-> Sparse`.
   - Define heuristics and controls for densification/materialization.
   - Exit criteria: conversion fidelity tests pass with no data-loss regressions.

## Implemented lazy elementwise execution

- Schemas and geometry report number-general `NumberType` classes; tensors use
  primitive type parameters directly. Typed `TensorMetadata<T>` replaces
  text metadata and dtype tags. File-entry adapters own byte codecs, type
  discrimination, format versions, and migration from old storage. fensor requires
  destream but does not enable a concrete codec; JSON and TBON storage are tested
  through explicit caller-owned `FileLoad`/`FileSave` implementations.
  Whole-tensor transfer formats belong to callers consuming `TensorRead`.

- Views evaluate implicitly through read consumers. `Tensor::copy_from` is the
  single constructor for copying any reader into independent storage; geometric
  and computed views have no separate materialization method.

- Native u8 storage covers the full byte range. `TensorUnaryBoolean::not` accepts
  all supported dtypes; `TensorNumeric::{is_nan, is_inf}` accepts floats and
  produces u8 views. Nested predicates retain root support through intermediate
  false results, with no implicit sparse densification.

- `TensorCast<f64>` supports f32-to-f64 nested views without intermediate
  evaluation. Root support survives dtype changes, and terminal consumers use
  the output dtype; `Tensor::copy_from` supports a distinct destination adapter.

- `TensorAbs` and `TensorTrig` support absolute value and all nine ndarray
  trigonometric operations for f32/f64 through the same nested unary evaluator.
  Sparse operations retain original source support, including for cosine,
  inverse cosine, and hyperbolic cosine; domain results follow the ndarray backend.

- Geometric `TensorView` and computed `UnaryView<Source, Op>` are separate types.
  `exp`, `ln`, and `round` nest unary views over their immediate sources, following
  ndarray access composition. Consumers read geometric leaves directly and build
  the complete ndarray expression before evaluating once per batch, without
  intermediate filesystem materialization, vector conversion, or sparse filtering.
  Unary traits and `TensorMath<Rhs>` use associated output types; geometric
  transform interfaces retain their existing contracts.
- `BinaryView<Left, Right, Op>` supports add/sub/mul/div/pow/rem for matching
  f32/f64/u8 and log for matching floats. Shape mismatches fail at construction;
  broadcasting and dtype conversions are explicit. Nested unary/binary expressions
  share one recursive batch builder, terminal evaluation, and ordered concurrency
  boundary. Memory scales with expression size and bounded batches.
- Binary support is the union of original leaf support, with absent children
  masked to zero before their parent operation. Intermediate zeros retain support;
  copying creates a new support boundary. Only two sparse operands produce a
  sparse result. Neither computed view family implements TensorWrite or TensorArray.
- Numerical rules and backend conformance belong to
  [ha-ndarray's contract](../ha-ndarray/NUMERICS.md), including wrapping u8
  arithmetic, zero integer divisors, IEEE floating behavior, and cast compatibility.
  fensor retains its distinct sparse source-support semantics and bounded execution.
  Certified reference and capability-rejection tests live in ha-ndarray's shared
  suite. Actual GPU conformance remains a pending ha-ndarray validation gate.
- Bases and views expose fresh, bounded row-major value streams. Fixed-size batches
  and CPU-based concurrency apply backpressure through consumption and awaited I/O.
- Direct reads, block streams, and `Tensor::copy_from` agree;
  computed views exclude `TensorWrite` at compile time, including after transforms.
- Sparse chains retain source support through intermediate zeros. Transformed
  sparse materialization is supported; non-row-major sparse order is rejected.
- Cache admission accounts for block payload bytes; real-cache tests exercise
  spill and reload with tensors larger than the configured cache capacity.
- Range iteration retains interval descriptors instead of expanding axes.
- `Tensor::sync` writes blocks and publishes the current sparse index root before
  reload. Directory-only synchronization is insufficient for an in-memory root;
  durable synchronization and transaction policy remain with the caller.

- Scalar arithmetic, tensor/scalar comparisons, and logical boolean operations
  use typed expressions for f32, f64, and u8 (logarithms remain float-only).
  Scalars preserve source support; binary predicates preserve its union.
- Conditional `WhereView` selection accepts u8 conditions and matching branch
  dtypes/shapes, retaining support from all three sources without intermediate
  evaluation. Both branches are read and propagate errors. All new expressions
  support transforms, bounded reads, and copying, but cannot be written through.

- Whole-tensor numeric/boolean reductions and typed lazy axis reductions support
  f32, f64, and u8. They reduce retained source support, excluding implicit sparse
  zeros but including supported intermediate zeros. Empty axis groups stay absent;
  terminal empty extrema return errors, with identity results for other terminals.
- Reduction output transforms share geometric coordinate mapping with storage
  views. Bounded source batches and one partial accumulator per active group avoid
  whole-group allocation; only the outer consumer starts concurrent batches.
  Boolean terminals short-circuit, so later corruption may remain unobserved.

- Bounded execution is an explicit design rule, enforced at batch boundaries.
  Whole-result collection APIs are removed; callers explicitly collect streams.
  Bulk writes validate range length without collecting coordinates. Existing
  gather slices/reversals share input-sized tables instead of duplicating them.

Remaining execution work:

- Matrix multiplication and additional casts remain separate work.
- Extend casting beyond f32-to-f64 and add complex storage/operations separately.
  Expression input/output dtypes are already independent, and u8 predicate output storage is supported.

- Index-driven bounded sparse traversal. Current ordered reads scan only the selected
  logical range; full-range reads still scale with logical size, not stored support.
- Block-oriented reads/writes to reduce per-coordinate cache/index lookups.
- Bounded sparse compaction and filesystem metadata scaling; execution limits do
  not bound filesystem metadata. Whole-result collection APIs and the previous
  whole-index compaction implementation have been removed.
- Extend the same read-stream contract to additional ndarray operations rather
  than introducing a separate executor or intermediate tensor files.

## Layout support matrix

- `Dense`: filesystem blocks only, no sparse index.
- `Sparse`: sparse coordinate indexing via `b-table` into block payloads, with optional `axis: Option<usize>` hint for axis-focused sparsity planning.

## Execution strategy

1. **Layout accessor correctness first.**
   - Implement and validate coordinate-to-offset, offset-to-block, and sparse-key mapping before higher-level transforms.
   - Gate: dense/sparse point read-write parity tests pass for identical logical coordinates.

2. **One writable base plus tensor views.**
   - Keep one writable base tensor type and represent transformed/sliced tensors as views implementing the same tensor trait surface.
   - Gate: calling code can consume base and view tensors uniformly without adapter-specific branches.

3. **Transform validation in layers.**
   - Validate `transpose` (including arbitrary axis permutations) and `slice` independently first.
   - Then validate composition (`transpose(slice(x))`, `slice(transpose(x))`, chained operations) as a separate gate.
   - Gate: separate and composed transform tests both pass with identical logical results across Dense/Sparse layouts.

## Correctness gates (excluding transactionality)

1. **Sparse ordered iteration contract.**
   - Specify when sparse elements can be streamed in requested logical order without re-materialization.
   - For incompatible order requests, return a structured `UnsupportedSparseIterationOrder` error with actionable guidance.
   - Gate: positive and negative tests cover arbitrary permutation + slicing combinations.

2. **Sparse zero-write lifecycle contract.**
   - Specify behavior for writing zero into sparse coordinates as row removal.
   - Ensure behavior is deterministic and documented; no silent policy drift.
   - Gate: tests assert index/block state transitions for nonzero->zero and zero->nonzero updates.

3. **Persistence and rehydration semantics.**
   - Validate metadata and block persistence across real `freqfs` reload boundaries.
   - Corruption and malformed metadata must fail closed with structured errors.
   - Gate: integration tests cover create->write->reload->read and corruption->error flows.
   - Typed metadata is serialized through destream; file adapters preserve element types and select byte codecs. Applications define whole-tensor transfer formats using `TensorRead`. `Tensor::copy_from` creates independent storage from a reader.

4. **Base/view write-through semantics.**
   - Define which transformed tensors are writable and which are read-only.
   - Preserve one base writable tensor model with trait-compatible views and explicit constraints.
   - Gate: tests assert view writes (when supported) mutate base state correctly and unsupported writes fail with clear errors.

5. **Dense/Sparse operation matrix parity.**
   - Build one parity matrix covering `read`, `write`, `slice`, `transpose`, and chained transforms.
   - Require identical logical results where operations are supported for both layouts.
   - Gate: matrix tests run against both layouts and capture allowed divergence only through explicit structured errors.

6. **Structured error coverage audit.**
   - Eliminate `todo!`, `unimplemented!`, and panic-style placeholders from production paths.
   - Ensure unsupported/invalid states map to typed `Error` variants with user-facing messages.
   - Gate: targeted tests assert error variant and message quality for each unsupported path.

7. **Async trait performance gate.**
   - Keep `BoxFuture` in trait methods during interface stabilization to avoid API churn.
   - Treat `BoxFuture` as an intentional tradeoff (allocation + dynamic dispatch) acceptable for I/O-bound paths.
   - Profile hot compute paths (`read_value` loops, reductions, matmul plumbing); if overhead is measurable, migrate those traits to associated future types (GAT-style) for static dispatch.
   - Gate: benchmark evidence recorded before/after any async-signature migration.

## Next execution checklist

1. **Complete trait-backed base/view implementation.**
   - Finish `TensorArray`, `TensorRead`, `TensorWrite`, `TensorTransform`, `TensorBlockStore`, and `TensorSparseIndex` for one writable base tensor plus trait-compatible view tensors.
   - Keep view construction lazy (metadata/mapping only) until consumed by reads or `Tensor::copy_from`.

2. **Accessor test suite (first).**
   - Add focused tests for coordinate-to-offset, offset-to-block, and sparse-key mapping.
   - Include edge cases (rank-1 tensors, boundary coordinates, degenerate slices, sparse axis hints).

3. **Transform test suite (second).**
   - Add standalone `transpose` tests with arbitrary valid permutations.
   - Add standalone `slice` tests across `At`/`In`/`Of` range forms.
   - Add composition tests (`transpose(slice(x))`, `slice(transpose(x))`, chained views).

4. **Cross-layout parity suite (third).**
   - Validate equal logical results for identical operations across Dense and Sparse layouts.
   - Include read parity and write-through behavior wherever writeable view semantics are enabled.

5. **Phase exit criteria lock-in.**
   - Phase 1 exits only when accessor and standalone transform tests pass.
   - Phase 2 exits only when composed transforms and Dense/Sparse parity tests pass.
   - Non-transactional parity exits only when all gates in `Correctness gates (excluding transactionality)` pass.

## Active deliverables

1. **Core persistence hardening.**
   - Replace placeholder metadata handling with explicit persisted tensor metadata (shape, axes, block size, sparsity mode).
   - Validate coordinate normalization and axis-permutation behavior against canonical Tensor semantics.
   - Remove `TODO` scaffolding in block read/write paths and codify invariants in crate-level docs.

2. **Index model completion.**
   - Finalize sparse index schema contracts over `b-table` (`coord`, `block_offset`, `block_id`) and document extension points for alternate sparse layouts.
   - Add dense/sparse parity tests for identical logical reads/writes.
   - Define compaction/cleanup behavior for sparse blocks when values are overwritten with zero.

## Deferred explorations

- **Adaptive block sizing.** Evaluate configurable or data-driven block sizing after baseline persistence semantics are stable.
- **Typed tensor families.** Expand beyond `f32` once core lifecycle and sparse-index behavior are validated.
- **Cross-host sharding hooks.** Keep routing/sharding orchestration in client libraries while exposing reusable shard-local primitives in `fensor`.
- **Index-driven sparse enumeration.** Sparse reads traverse the selected logical range. Future traversal should use stored support while preserving ordering through geometric transforms.

## ha-ndarray execution dependency acknowledgement

`fensor` will not implement direct CubeCL execution. `fensor` remains a storage
and indexing substrate, while `ha-ndarray` owns execution planning,
accelerator support, runtime fusion strategy, and backend selection.

### How `fensor` benefits from the `ha-ndarray` roadmap

- Streaming execution aligned with block-oriented persistence.
- More efficient block-oriented processing flows for large tensors.
- Out-of-core tensor execution through block iteration and streamed outputs.
- Runtime fusion benefits without embedding backend-specific executors in
  `fensor`.
- WebGPU/browser deployment path through `ha-ndarray` execution layers.
- Backend-independent accelerator support inherited from `ha-ndarray`
  execution contracts.

### Sequencing and dependency notes

1. **Contract alignment first.** `fensor` work should align with the
   `ha-ndarray` execution IR and block-stream contract boundaries before
   optimizing integration pathways.
2. **Storage invariants remain primary.** `fensor` continues prioritizing
   deterministic persistence/index correctness independently of accelerator
   details.
3. **Execution ownership remains external.** Backend choice, fusion policies,
   kernel caching strategy, and placement policy are dependencies from
   `ha-ndarray`, not responsibilities of `fensor`.

### Risks and mitigations

1. **Risk: contract drift between execution and storage layers.**
   - Mitigation: maintain explicit integration gates for block iteration,
     materialization boundaries, and sparse/dense parity expectations.
2. **Risk: accidental accelerator coupling in storage codepaths.**
   - Mitigation: keep `fensor` APIs backend-agnostic and avoid backend-specific
     assumptions in persistence logic.
3. **Risk: sequencing mismatch with browser and out-of-core milestones.**
   - Mitigation: stage `fensor` integration work behind published
     `ha-ndarray` execution milestones and shared parity tests.
