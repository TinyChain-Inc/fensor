# fensor Agent Notes

`fensor` is a filesystem-backed tensor primitive. Keep it minimal,
non-transactional, and explicit about supported and unsupported behavior.

## Implementation constraints

- Follow [DESIGN.md](DESIGN.md) for bounded requests, mapping, slice traversal,
  matrix execution, and destination writes. Preserve supported geometric paths;
  do not merge distinct data representations merely to reduce line counts.
- Keep one canonical implementation in each owning trait method. Avoid parallel
  `*_impl` forwarding layers, execution registries, and convenience traits.
- Delegate behavior through expression APIs, not strategy enums, equivalent flags,
  or type probes. Request providers return actual iterators; precedence and error
  propagation must be explicit. Enums may represent genuine data alternatives.
- Expression construction is lazy even when asynchronous. Consumers may evaluate
  bounded intermediates in memory but never persist them implicitly. Reads and
  terminal reductions create no result storage. Source-cache spill is separate.
- Only outer consumers buffer batches. Preserve the public stream orders,
  completion-order numeric terminals, boolean short-circuit/error boundaries,
  and sequential destination updates. Add no tasks or nested worker pools.
- Do not add transaction lifecycle, recovery, or codec policy. Malformed metadata
  and blocks fail closed; callers own cleanup, durability, and transactions.

## Bounded execution

Tensor execution must not allocate values, coordinates, support masks, or
partial-result collections proportional to total tensor size, output size, or
reduction-group size. Coordinates must be generated lazily and consumed in
bounded batches. Each collection must have an identifiable bound.

Rank-sized metadata, caller-owned values/explicit selections, filesystem/cache
metadata, and independent consumers are separate costs, not permission to collect
implicit ranges. This is not a total-process memory guarantee. Keep the bounds
and their owners documented in [DESIGN.md](DESIGN.md#bound-and-policy-constants).
No whole-result collectors or whole-index compaction APIs.

## Semantics and validation

- Preserve one writable base and constrained geometric write-through. Computed
  views remain read-only; numerical definitions belong to ha-ndarray.
- Keep support independent of intermediate values; copying establishes a new
  sparse support boundary. Preserve ordered sparse-read rejection and scalar
  zero-write lifecycle behavior.
- Shapes/coordinates use `u64` at schema/wire boundaries; runtime axes use
  `usize`. Centralize conversions. Preserve adapter-owned payload typing.
- Validate batch sizes and support lengths before evaluation/combination and
  after consumption. Keep errors structured; no debug-only safety checks.
- Keep counters test-only and isolated from storage synchronization. Use local
  or task-local observations for concurrent correctness tests.

## Documentation and tests

Follow [CODE_STYLE.md](CODE_STYLE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
Update the owning contract when behavior changes: README for public use, DESIGN
for execution invariants, ROADMAP for unfinished work. Link rather than repeat
implementation history. Keep benchmark code and methodology in Git; write generated
measurements, logs, and provenance under ignored `benchmarks/results/`. Do not
publish result tables in project documentation or delete local evidence as cleanup.
Benchmark runners accept data directories without classifying the underlying storage.
Use [tests/COVERAGE.md](tests/COVERAGE.md) to retain critical parity, persistence,
corruption, cancellation, and structural bounds without duplicating permutations.
