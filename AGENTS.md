# fensor Agent Notes

`fensor` is a filesystem-backed tensor storage primitive. Keep it minimal,
non-transactional, and explicit about supported/unsupported behavior.

## Core constraints

- `fensor` is not a transaction manager. Do not add commit, rollback, or
  finalize semantics here; callers may layer their own lifecycle policy.
- Fail closed on corruption. No fallback recovery paths for malformed metadata
  or data files.
- Keep one obvious code path per feature. Prefer general primitives over
  special-case branches.
- For tensor behavior surfaced by traits (`TensorRead`, `TensorWrite`,
  `TensorTransform`, `TensorBlockStore`, `TensorSparseIndex`), keep the
  canonical implementation in the trait methods themselves. Avoid parallel
  `*_impl` forwarding layers that create a second path to inspect/debug.

## Bounded execution

Tensor execution must not allocate values, coordinates, support masks, or
partial-result collections proportional to total tensor size, output size, or
reduction-group size. Coordinates must be generated lazily and consumed in
bounded batches. Each collection must have an identifiable bound. Only the outer
consumer may introduce concurrent batches.

Rank-sized metadata and caller-supplied values or explicit index selections are
separate, documented memory costs. They must not justify expanding implicit
ranges or collecting execution results. Filesystem/cache metadata remains subject
to its own limits; this contract is not a total-process memory guarantee.

- Execution batches contain at most 4096 elements; storage block capacity is a
  separate bound. Validate batch lengths before evaluation and after consumption.
- Do not add whole-result collection helpers or whole-index compaction. Callers
  may explicitly collect streams; bounded compaction is future work.
- Preserve explicit selection order and duplicates. Slicing or reversing an
  existing gather table must share its storage rather than copy its entries.

## Tensor semantics

- Preserve base/view semantics: one writable base tensor with trait-compatible
  views and explicit write-through constraints.
- Canonical numeric typing rule:
  - shape dimensions and coordinate payloads are `u64` at schema/wire boundaries,
  - axis identifiers and axis indexing are `usize` in runtime structs and APIs.
  Keep conversions centralized at schema/stream boundaries.
- Sparse behavior must be deterministic and documented:
  - ordered iteration support boundaries,
  - structured errors for incompatible order,
  - explicit zero-write lifecycle policy.
- Keep dense/sparse parity where operations are supported; unsupported cases
  must return structured errors.

## Docs and testing

- Update `README.md` and `ROADMAP.md` when behavior contracts change.
- Keep tests focused on critical paths and parity gates:
  - access parity,
  - transform composition,
  - sparse lifecycle,
  - persistence/reload semantics.
- Follow shared code style from [`CODE_STYLE.md`](./CODE_STYLE.md).
