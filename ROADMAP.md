# fensor roadmap

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

Transactional orchestration belongs to `tc-collection`, which composes `fensor` as the storage engine (analogous to `b-tree`/`b-table` usage in v1).

## Data integrity contract (fail-closed)

`fensor` must treat corruption as a hard error, not a recoverable condition.

- No in-place auto-repair of metadata or block files.
- No automatic fallback from a corrupted source to alternate metadata/data copies.
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

## v1 parity gates (excluding LSM transactionality)

1. **Sparse ordered iteration contract.**
   - Specify when sparse elements can be streamed in requested logical order without re-materialization.
   - For incompatible order requests, return a structured `UnsupportedSparseIterationOrder` error with actionable guidance.
   - Gate: positive and negative tests cover arbitrary permutation + slicing combinations.

2. **Sparse zero-write lifecycle contract.**
   - Specify behavior for writing zero into sparse coordinates: row removal, tombstoning, or retained zero rows.
   - Ensure behavior is deterministic and documented; no silent policy drift.
   - Gate: tests assert index/block state transitions for nonzero->zero and zero->nonzero updates.

3. **Persistence and rehydration semantics.**
   - Validate metadata and block persistence across real `freqfs` reload boundaries.
   - Corruption and malformed metadata must fail closed with structured errors.
   - Gate: integration tests cover create->write->reload->read and corruption->error flows.

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
   - Keep view construction lazy (metadata/mapping only) until explicit materialization is requested.

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
   - v1 non-transactional parity exits only when all gates in `v1 parity gates (excluding LSM transactionality)` pass.

## Active deliverables

1. **Core persistence hardening.**
   - Replace placeholder metadata handling with explicit persisted tensor metadata (shape, axes, block size, sparsity mode).
   - Validate coordinate normalization and axis-permutation behavior against canonical Tensor semantics.
   - Remove `TODO` scaffolding in block read/write paths and codify invariants in crate-level docs.

2. **Index model completion.**
   - Finalize sparse index schema contracts over `b-table` (`coord`, `block_offset`, `block_id`) and document extension points for alternate sparse layouts.
   - Add dense/sparse parity tests for identical logical reads/writes.
   - Define compaction/cleanup behavior for sparse blocks when values are overwritten with zero.

3. **TinyChain integration milestones.**
   - Integrate `fensor` as a non-transactional storage primitive in host/state lifecycle so tensor storage participates in install, queue, capability checks, and telemetry emission.
   - Replace `tc-state` transitional in-memory tensor plumbing with `fensor`-backed persistence once lifecycle hooks are wired.
   - Keep URI and serialization behavior aligned with canonical `/state/collection/tensor` and tuple payload contracts.

## Deferred explorations

- **Adaptive block sizing.** Evaluate configurable or data-driven block sizing after baseline persistence semantics are stable.
- **Typed tensor families.** Expand beyond `f32` once core lifecycle and sparse-index behavior are validated.
- **Cross-host sharding hooks.** Keep routing/sharding orchestration in client libraries while exposing reusable shard-local primitives in `fensor`.
