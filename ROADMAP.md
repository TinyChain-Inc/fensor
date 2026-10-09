# fensor roadmap

This non-normative file tracks unimplemented work. Current behavior is described
in the [README](README.md); execution invariants live in [DESIGN.md](DESIGN.md).
Prioritize additional optimization using the [benchmark methodology](BENCHMARKS.md),
preserving the [logical geometry and collection rules](DESIGN.md#geometry-and-collection-ownership).

## API extensions

- Fourier transforms beyond the bounded axis-length limit, and frequency shifting
  after auditing backend shift semantics. Bounded last-axis/2D transforms and
  conjugate transpose are supported with the optional complex feature.
- Construction/composition APIs such as constants, ranges, random generation,
  stacking and concatenation, reusing bounded expressions and consumers.

## Sparse traversal

- Extend indexed ordered reads to additional mappings whose order can be provided
  directly by native indexes. Incompatible geometry retains bounded logical evaluation.
- Indexed sparse diagonal traversal; current diagonal reads select bounded
  coordinates and full scans remain proportional to logical diagonal length.
- Indexed aggregate output candidates, preserving implicit-zero values and corruption/error contracts.
- Reuse native traversal for sorted nonadjacent blocks without reading unselected
  storage. Consecutive keys share a bounded range; scattered candidates and
  transaction-visible sources still perform repeated logical-block lookups.
- Sparse matrix algorithms that skip empty contraction intervals without losing
  zero-times-infinity behavior. Current matrix
  products scan logical contraction length.
- Native table metadata scaling and bounded page compaction. Dense chunk rows
  share pages and remove obsolete rows during replacement; execution batch
  limits do not bound total filesystem metadata. Whole-index collection is prohibited.

## Execution and storage costs

- Symbolic affine planning for transformed matrix outputs; geometric storage leaves
  already use affine runs. Preserve the bounded irregular/gather paths.
- Operand reuse across matrix tiles and nested-expression scheduling, guided by
  measurements and without implicitly persisting computed intermediates.
- Dense-initialization improvements and completion-order cache locality, guided
  by measurements. Native fills and value-buffer writes already update bounded
  blocks; `copy_from` consumes and writes each destination batch sequentially.
- Profile async allocation and backend scheduling before changing future types,
  CPU offloading, concurrency limits, or block-sizing policy. No public tuning
  abstraction is justified by a hypothetical performance benefit alone.

## External dependencies and boundaries

ha-ndarray owns numerical definitions, backend selection, fusion, and hardware
parallelism. Its [numerical contract and validation status](https://github.com/TinyChain-Inc/ha-ndarray/blob/main/NUMERICS.md)
provide non-normative integration context for GPU conformance and future CubeCL
support; fensor's storage tests cannot establish
backend conformance. Backend integration must retain bounded, lazy consumption.

Transaction lifecycle, recovery, cross-host orchestration, codec selection, and
durability policy remain caller responsibilities, not pending fensor features.

See the [native storage contract](DESIGN.md#native-storage-and-sources) for the typed
source, logical geometry, occupied traversal, owned value, strict loading, and
native replacement interfaces.
