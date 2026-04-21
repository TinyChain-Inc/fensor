# fensor roadmap

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
   - Integrate `fensor` into host/state lifecycle so tensor storage participates in install, queue, capability checks, and telemetry emission.
   - Replace `tc-state` transitional in-memory tensor plumbing with `fensor`-backed persistence once lifecycle hooks are wired.
   - Keep URI and serialization behavior aligned with canonical `/state/collection/tensor` and tuple payload contracts.

## Deferred explorations

- **Adaptive block sizing.** Evaluate configurable or data-driven block sizing after baseline persistence semantics are stable.
- **Typed tensor families.** Expand beyond `f32` once core lifecycle and sparse-index behavior are validated.
- **Cross-host sharding hooks.** Keep routing/sharding orchestration in client libraries while exposing reusable shard-local primitives in `fensor`.
