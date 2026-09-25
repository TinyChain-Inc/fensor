# TinyChain Code Style

These rules apply throughout this repository. Keep local crate notes short and
link back here so the repository does not fork its style guide.

## Imports

Group `use` statements in this order, separated by one blank line:

1. Standard library (`std`, `core`, `alloc`).
2. External crates (alphabetical).
3. Workspace/internal modules (`crate::`, `super::`, repo siblings).

Within each group:

- Sort alphabetically (case-sensitive).
- Prefer explicit paths (`use foo::bar::Baz;`) over glob imports.
- Merge shared prefixes (use `foo::{Bar, Baz}`).

For conditional imports (`#[cfg]`), keep the group ordering and put the cfg on
the line above the statement.

## Linting

- Run `cargo clippy --all-targets --all-features -- -D warnings` locally or via
  `just lint` (whatever script your crate’s `CONTRIBUTING.md` references). Fix
  unused imports, let bindings, etc., instead of adding `allow` attributes.

## Formatting

- Run `cargo fmt` locally

## Crate-specific notes

Each crate’s `CONTRIBUTING.md` should point back to this doc and only mention
extra rules (e.g., feature-flag patterns, generated code) so cross-crate
consistency remains automatic.

## Collections and geometry

- Use `Shape`, `Strides`, and the private `Coord` for small rank-sized `u64`
  metadata. Reuse `ha_ndarray::Axes` for `usize` axis identifiers and permutations.
  Their inline capacity of eight is an allocation optimization, not a rank limit.
- Use `Range` for rank-sized slice bounds; explicit `AxisRange::Of` selections own
  `Vec<u64>` and preserve order and duplicates. Shared gather tables retain `Arc`.
- Use `Vec` for values, support masks, batches, runs, scatter lists, public
  coordinate payloads, and adapter buffers. Prefer slices when borrowing and
  arrays for fixed-cardinality structures such as sparse keys.
- Keep large axis descriptors, signed affine coefficient vectors, and matrix map
  keys heap-backed. Inline capacity would enlarge containing requests, map nodes,
  and async futures; rank-sized alone is not sufficient reason to use `SmallVec`.
- Logical dimensions, unsigned strides, coordinates, cardinalities, and traversal
  positions use `u64`. Ranks, axes, allocation lengths, and buffer offsets use
  `usize`. Narrow only after checking the relevant bounded allocation or index.
  Signed logical mappings use checked `i128`; block-local signed strides retain
  their bounded representation.

## Local operation macros

Private `macro_rules!` macros may generate mechanically identical operation
declarations and constructor members inside explicit trait implementations. Keep
public trait declarations, behavioral control flow, and unusual bounds directly
readable. Preserve each operation's documentation and explicit backend mapping;
do not introduce an operation registry or generate execution implementations.
