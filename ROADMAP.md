# fensor roadmap

This non-normative file tracks unimplemented work. Current behavior is described
in the [README](README.md); execution invariants live in [DESIGN.md](DESIGN.md).
Prioritize additional optimization using the [benchmark methodology](BENCHMARKS.md),
preserving the [logical geometry and collection rules](DESIGN.md#geometry-and-collection-ownership).

## API extensions

- Casts beyond f32-to-f64, additional integer/complex dtypes, and their storage
  adapters and conformance tests. f32, f64, and u8 storage already exist.
- Matrix-specific unary operations and other remaining ha-ndarray operations,
  reusing typed expressions and bounded consumers rather than a separate executor.

## Sparse traversal

- Index-driven ordered sparse reads through geometric transforms. Current public
  sparse streams scan the selected logical range; eligible numeric reductions
  already enumerate occupied storage regions.
- Coordinated occupied-source traversal for binary, conditional, and aggregate
  expression sources, preserving union support and corruption/error contracts.
- Sparse matrix algorithms that skip empty contraction intervals without losing
  retained intermediate support or zero-times-infinity behavior. Current matrix
  products scan logical contraction length.
- Bounded sparse compaction and filesystem/index metadata scaling. Execution
  batch limits do not bound filesystem metadata; whole-index collection is prohibited.

## Execution and storage costs

- Symbolic affine planning for transformed matrix outputs; geometric storage leaves
  already use affine runs. Preserve the bounded irregular/gather paths.
- Operand reuse across matrix tiles and nested-expression scheduling, guided by
  measurements and without implicitly persisting computed intermediates.
- Grouped general bulk writes and possible dense-initialization improvements.
  `copy_from` already groups destination blocks and overlaps reads with sequential
  writes. Investigate completion-order cache locality separately.
- Profile async allocation and backend scheduling before changing future types,
  CPU offloading, concurrency limits, or block-sizing policy. No public tuning
  abstraction is justified by a hypothetical performance benefit alone.

## External dependencies and boundaries

ha-ndarray owns numerical definitions, backend selection, fusion, and hardware
parallelism. Follow its [numerical contract and validation status](../ha-ndarray/NUMERICS.md)
for GPU conformance and future CubeCL support; fensor's storage tests cannot establish
backend conformance. Backend integration must retain bounded, lazy consumption.

Transaction lifecycle, recovery, cross-host orchestration, codec selection, and
durability policy remain caller responsibilities, not pending fensor features.
