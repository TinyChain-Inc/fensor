# fensor Agent Notes

`fensor` is a filesystem-backed tensor storage primitive. Keep it minimal,
non-transactional, and explicit about supported/unsupported behavior.

## Core constraints

- `fensor` is not a transaction manager. Do not add commit/rollback/finalize
  semantics here; those belong in `tc-collection`.
- Fail closed on corruption. No fallback recovery paths for malformed metadata
  or data files.
- Keep one obvious code path per feature. Prefer general primitives over
  special-case branches.
- For tensor behavior surfaced by traits (`TensorRead`, `TensorWrite`,
  `TensorTransform`, `TensorBlockStore`, `TensorSparseIndex`), keep the
  canonical implementation in the trait methods themselves. Avoid parallel
  `*_impl` forwarding layers that create a second path to inspect/debug.

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
