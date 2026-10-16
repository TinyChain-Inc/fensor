# fensor Agent Notes

`fensor` is a filesystem-backed tensor primitive. Keep it minimal,
non-transactional, and explicit about supported and unsupported behavior.

## Implementation constraints

- Follow [DESIGN.md](DESIGN.md) for bounded requests, mapping, slice traversal,
  matrix execution, and destination writes. Preserve supported geometric paths;
  do not merge distinct data representations merely to reduce line counts.
- Keep one canonical implementation in each owning trait method. Avoid parallel
  `*_impl` forwarding layers, execution registries, and convenience traits.
- Keep operations in concrete views and delegate to their operands. Confine owned
  runtime values and stack-safe consumption to shared boundaries; do not build a
  second typed or owned execution engine. Recursive descriptions do not justify
  input-dependent recursion in polling, traversal, or destruction.
- Review new layout coercions, error-ordering state, caches, and fast paths against
  a concrete contract or measurement. Prefer explicit destination choices and
  structured unsupported errors over silently choosing allocation policy.
- Use concrete types and trait methods for operations known at construction.
  Enums may represent runtime data alternatives or explicit traversal states;
  do not add operation/strategy tags that duplicate concrete behavior. Keep
  representation matching near its owner and expose validated traversal, except
  where a specialized consumer needs compact geometry. Request providers return
  actual iterators; precedence and error propagation remain explicit.
- Expression construction is lazy even when asynchronous. Consumers may evaluate
  bounded intermediates in memory but never persist them implicitly. Reads and
  terminal reductions create no result storage. Source-cache spill is separate.
- Consume expression requests one at a time. Dense storage buffers ordinary file
  reads with futures; freqfs owns admission, eviction, and bounded capacity errors.
  Do not add an expression batch queue, speculative warming stage, tasks, or
  nested worker pools. Preserve request order, boolean short-circuit boundaries,
  and sequential destination updates. Optimize against adequately provisioned
  production workloads; constrained-cache fixtures check bounded failure and
  cleanup, not a separate performance policy.
- Do not add transaction lifecycle, recovery, or codec policy. Malformed metadata
  and blocks fail closed; callers own cleanup, durability, and transactions.

## Bounded execution

Tensor execution must not allocate values, coordinates, or
partial-result collections proportional to total tensor size, output size, or
reduction-group size. Coordinates must be generated lazily and consumed in
bounded batches. Each collection must have an identifiable bound; name it locally
for non-obvious allocations. Do not size implicit working buffers from total
logical cardinality. Streamed filesystem materialization and caller-supplied
literal buffers are permitted, not permission to collect tensor streams in RAM.
Keep execution batches, storage blocks, metadata ranks, and index pages independent,
even when their limits have equal values.

Rank-sized metadata, caller-owned values/explicit selections, filesystem/cache
metadata, and independent consumers are separate costs, not permission to collect
implicit ranges. This is not a total-process memory guarantee. Keep the bounds
and their owners documented in [DESIGN.md](DESIGN.md#bound-and-policy-constants).
No whole-result collectors or whole-index compaction APIs.

## Semantics and validation

- Preserve one writable base and constrained geometric write-through. Computed
  views remain read-only; numerical definitions belong to ha-ndarray.
- Sparse absence is numerical zero. Reductions count logical cardinality; sparse
  pointwise construction rejects nonzero implicit backgrounds unless explicitly
  converted to dense. Candidate traversal is an index optimization, never a
  numerical mask. Cache only the typed implicit-zero result needed to preserve
  backend zero signs through expressions and indexed reductions. Preserve ordered
  sparse reads and all-zero storage-block deletion.
- Logical shapes, strides, coordinates, and cardinalities use `u64`; runtime
  axes and bounded buffer indices use `usize`. Follow the collection rules in
  `CODE_STYLE.md`. Preserve adapter-owned payload typing.
- Validate request and batch cardinalities before evaluation/combination and
  after consumption. Keep errors structured; no debug-only safety checks.
- Keep counters test-only and isolated from storage synchronization. Use local
  or task-local observations for concurrent correctness tests.

## Documentation and tests

Follow [CODE_STYLE.md](CODE_STYLE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
Update the owning contract when behavior changes: README for public use, DESIGN
for execution invariants, ROADMAP for unfinished work. Link rather than repeat
implementation history. Keep benchmark code and methodology in Git; retain generated
measurements, logs, and reproducible source/dependency provenance under ignored
`benchmarks/results/`. Build caches and executables are disposable, not evidence:
keep Cargo targets outside results and remove campaign executables after recording
hashes and build inputs. Reuse build targets instead of retaining one per campaign.
Do not publish result tables in project documentation or delete measurements as
cleanup. Benchmark runners accept data directories without classifying storage.
Use [tests/COVERAGE.md](tests/COVERAGE.md) to retain critical parity, persistence,
corruption, cancellation, and structural bounds without duplicating permutations.
