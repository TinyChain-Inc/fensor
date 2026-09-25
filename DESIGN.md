# Bounded execution design

This is the implementation reference for fensor's execution invariants. Public
usage and sparse semantics are in [README.md](README.md); benchmark methodology
and interpretation are in [BENCHMARKS.md](BENCHMARKS.md). Inline documentation owns
individual field and function invariants.

## Resource and persistence contract

Tensor execution must not allocate values, coordinates, support masks, or
partial-result collections proportional to total tensor size, output size, or
reduction-group size. Coordinates must be generated lazily and consumed in
bounded batches. Each collection must have an identifiable bound. Only the outer
consumer may introduce concurrent batches.

Rank-sized metadata and caller-supplied values or explicit index selections are
separate, documented memory costs. They must not justify expanding implicit
ranges or collecting execution results. Filesystem/cache metadata remains subject
to its own limits; this contract is not a total-process memory guarantee.

Construction creates descriptions only, including asynchronous construction.
Elementwise composition builds a nested backend expression; reductions and matrix
products can evaluate bounded intermediates in memory. None may create temporary
tensor files or persist computed intermediates. Persistence requires an explicit
write or `Tensor::copy_from`. Source-cache spill/reload is permitted. No result
cache, shared cursor, or expression registry is part of execution.

## Requests and evaluation

The private expression contract accepts one `BatchRequest` containing at most
`MAX_BATCH_ELEMENTS` logical elements:

| Representation | Metadata bound |
|---|---|
| Linear span | Start/count; coordinates use rank-sized scratch |
| Cartesian rectangles | Bounded collection of rank-sized axes; intervals or bounded explicit indices |
| Explicit coordinates | At most one rank-sized coordinate per batch element |

Requests preserve order and duplicates. A coordinate cursor writes into reusable
scratch. Unary, binary, and conditional nodes pass the same request recursively;
they do not expand rectangles into coordinate lists. Coordinates are expanded only
when required by irregular mapping or the public coordinate-stream boundary.
Value-only output discards the request without constructing output coordinates.

`evaluate_batch` is the common numerical evaluation boundary. It validates request
cardinality before expression construction, array size and support-mask length
before backend evaluation, and returned values afterward. Union/filter operations
validate lengths so `zip` cannot silently truncate. Violations are structured
`InvalidLayout` errors, not debug-only assertions. Dense support uses no mask;
sparse support survives intermediate zeros independently of numerical values.

## Request providers and concurrency

Expression methods provide actual bounded request iterators, not strategy tags.
`preferred_requests(shape)` returns an iterator or no preference. Matrix products
generate tiles for the consuming expression's current shape, using linear requests
below rank two. Unary/scalar nodes delegate; binary nodes consult left then right;
conditionals consult condition, then, else. The first iterator or error ends the
search. Geometric sources and reductions have no preference, so their consumers
generate linear requests. Reduction output remains a traversal boundary even over
a matrix source. Output mappings are independent of request traversal.

One unbuffered stream of evaluation futures retains each request with its evaluated
batch. Outer consumers apply `buffered(num_cpus::get().max(1))` for row-major value,
ordered sparse, and boolean reads, or `buffer_unordered` with the same limit for
coordinate streams and numeric terminals. No task is spawned. Request generation
order therefore does not guarantee coordinate-batch delivery order.

ha-ndarray owns numerical parallelism; fensor owns asynchronous batch concurrency.
Synchronous backend evaluation occupies the polling thread and may use backend
workers. Inner sources, reduction groups, and matrix contraction chunks do not
buffer recursively. Dropping consumption cancels pending futures. Numeric
terminals accumulate completed partials immediately under the backend aggregate
contract; boolean terminals preserve logical short-circuit/error boundaries.

## Mapping and shared storage reads

`CoordinateMap` shares geometric mapping between storage views and aggregate
outputs. Identity and affine eligibility are derived structurally, without cached
flags. `AxisRange::Of` may allocate metadata proportional to explicitly supplied
indices. Slicing/flipping an existing gather shares its table through offset,
direction/stride, and length; interval slices do not expand to index lists.

Affine leaves fold broadcast constants into the base offset and retain signed
strides. Linear requests become row segments; Cartesian requests become fastest-
axis progressions. Explicit axis selections are scanned for constant-step runs
without sorting or removing duplicates. Checked widened arithmetic validates
endpoints before I/O. Invalid mappings fail; only unsupported optimization geometry
selects bounded coordinate mapping.

Storage leaves split progressions at base-coordinate carries, block boundaries, or
sparse-key changes. Each storage run holds destination start, source-block offset,
signed stride, and count. At most one run per requested element is retained.
Gather mappings and explicit requests emit/coalesce singleton runs using the same
mapping logic and rank-sized scratch, without collecting mapped base coordinates.

Request cursors, affine leaves, and matrix rows share checked logical decoding and
lazy segment iteration. Storage owns block/carry splitting; matrix products own
tile intersection and scattering. Consumers obtain these behaviors through ordinary
expression delegation, without selecting a planning strategy.

The shared reader validates all positions into bounded logical-address groups before
I/O. Each sparse key is resolved once; physical aliases merge before block access. Dense reads bypass
index resolution but use the same borrowing/validation/scatter path. Each needed
block is borrowed and its full length validated once per bounded request; release
the guard before awaiting another block. Unit-stride runs copy slices; negative,
zero, and other strides use the shared scatter loop. No whole block is cloned.
Absent sparse entries yield zero; missing dense or malformed blocks return errors.

Matrix diagonal projection maps each bounded output request to correlated source
coordinates using reusable rank-sized scratch and one explicit request. It calls
its source's expression builder directly, preserving lazy numerical work and
support without another evaluator or concurrency boundary. Output transforms use
the existing coordinate map. Diagonals use logical slice traversal; matrix
transpose delegates to the existing final-axis permutation.

## Slices and reductions

A validated slice retains source rank and compact axis descriptors, checking rank,
bounds, steps, endpoints, and cardinality with checked arithmetic. Iteration
preserves explicit order/duplicates, handles empty selections, and stays exhausted.
Whole reductions use a full slice; axis groups fix unreduced coordinates.

Slices of at most one execution batch retain direct requests. Larger eligible
sparse slices stream existing `(sparse-axis coordinate, grid-block ID)` index
entries, intersect visible regions, and reuse the ordinary reader for values and
support. Index-prefix bounds restrict the sparse axis; other filters can scan
more occupied metadata than they consume. Keys/intersections are validated before
data I/O; unrelated data blocks are not read merely to test intersection.
Index pages are bounded and release guards before value reads. Small intersecting
regions pack into bounded batches, with one overflow rectangle retained.

Base-coordinate slices, forward separable mappings, and supported permutations
can use indexed traversal. Gathers, reversals, broadcasts, incompatible reshapes,
and coordinated binary/conditional/aggregate sources use compact logical requests.
Unary/scalar nodes inherit their source's slice requests and evaluate the complete
expression; they never consume an intermediate zero-filtered reader. Boolean
terminals deliberately use logical requests to preserve error/decision order.

Terminal and axis reductions share an accumulator. Supported values are compacted
in place, reduced by ha-ndarray, then combined through its scalar arithmetic/extrema
rules. Seed from the first nonempty partial to preserve signed-zero behavior.
Complete small groups share one source request and bounded segment boundaries;
long groups keep one active accumulator and chunked inputs, never a partial list.
Requested axis outputs are processed sequentially within each output batch; dense
outputs omit support masks. Output transforms map reduced coordinates and do not
move through source axes. Selected output ranges still require their source groups.

## Matrix planning and accumulation

Output requests are normalized into bounded rectangle plans grouped by matrix
batch and spatial tile. Identity output mappings split Cartesian requests directly
at tile boundaries; linear spans derive row segments arithmetically. Compact
scatter descriptions preserve regular output order. Explicit requests and
nonidentity mappings use reusable coordinate scratch and the irregular planner.

Duplicate outputs share computation but retain scatter positions. A rectangle is
used when its area is at most twice the unique requested outputs; otherwise rows
with identical requested columns form exact rectangles. Plans retain only metadata
proportional to the current request. Process rectangles sequentially.

Each rectangle gathers only referenced rows/columns. Operand slices use the same
validated request construction as reductions and remain direct bounded reads.
Contraction chunks have length `min(remaining_k, MAX_BATCH_ELEMENTS / max(rows, cols))`.
Evaluate operands, reshape their bounded arrays, and call ha-ndarray `matmul`.
Seed one running result from the first partial, then combine validated partials
in place with backend `Number::add`. Do not skip numerical work based on zeros.

For two sparse operands, row/column support summaries retain union support across
contraction positions. When either operand is dense, omit output-support arrays
while still masking unsupported sparse children. Output mapping, tiling, and
contraction order are shared by both layouts. Contraction scans scale with logical
length; nested products and separate output batches may reevaluate operands.
Ordered wide-output streams limit cross-row reuse; tiled coordinate requests
improve reuse without guaranteeing locality after arbitrary transforms.

## Destination writes

`copy_from` validates each incoming coordinate/value batch, its bounds, and checked
cumulative count before its mutation. Exactly-once coverage belongs to the reader;
there is no whole-output duplicate detector. Dense positions group by block;
sparse final zeros are omitted and populated keys resolved once before physical
block grouping. Preserve update order within each group.

Scalar writes and copying share the low-level block-update primitive. Validate the
full stored length and all update offsets before changing a block. New sparse
blocks use one capacity-bounded buffer and publish their index row after creation.
Dense initialization remains eager; sparse scalar zero-write lifecycle is distinct
from copy zero omission. General bulk writes validate caller buffer cardinality
before mutation and zip values with lazy coordinates.

One destination update overlaps fetching one source lookahead via `try_join!`.
Memory includes the current batch and one lookahead beyond the source window, its
grouped positions/values, and one block guard. There is no cross-batch block cache
or additional worker pool. Errors drop pending futures and can leave partial
storage; neither rollback, a write prefix, nor concurrent-error precedence is
promised. Completion order can change cache locality; measure it for the workload.

## Bound and policy constants

These are private implementation owners, not public tuning options. Equal values
do not couple independent bounds.

| Meaning | Owner / constant | Current value |
|---|---|---:|
| Execution elements | `expression::MAX_BATCH_ELEMENTS` | 4096 |
| Sparse-index page entries | `tensor::SPARSE_INDEX_PAGE_ENTRIES` | 4096 |
| Values per storage block | `schema::MAX_BLOCK_CAPACITY` | 4096 |
| Sparse-index leaf bytes | `schema::SPARSE_INDEX_BLOCK_BYTES` | 4096 |
| Sparse-index node order | `schema::SPARSE_INDEX_ORDER` | 16 |
| Matrix spatial tile side | `matmul::TILE_SIDE` | 32 |
| Rectangle area / unique outputs | `matmul::MAX_RECTANGLE_AMPLIFICATION` | 2 |
| Inline rank capacity | `PORTABLE_INLINE_RANK` | 8 |

Result tiles hold at most `TILE_SIDE * TILE_SIDE` values. Contraction chunks use
`min(remaining_k, MAX_BATCH_ELEMENTS / max(selected_rows, selected_columns))`;
full tiles therefore use `MAX_BATCH_ELEMENTS / TILE_SIDE` contraction positions.
These are derived capacities, not extra configuration. Index pagination bounds
retained keys separately from the requests constructed for their visible regions.
Boundary tests use owning constants and adjacent values; numerical and benchmark fixtures retain
explicit inputs. Structural contract tests are indexed in [coverage ownership](tests/COVERAGE.md).

## Geometry and collection ownership

`Shape` and `Strides` hold `u64` metadata inline through `PORTABLE_INLINE_RANK`
and spill above it. `Axes` comes from ha-ndarray and contains machine-sized axis identifiers.
Private coordinate scratch uses `Coord`; public coordinate batches remain
`Vec<Vec<u64>>`. `Range` holds local `AxisRange` bounds with `u64` endpoints and
steps. Explicit selections are caller-sized vectors, never expanded intervals.

Logical cardinalities are checked and `TensorGeometry::size()` is fallible.
Signed mapping calculations use checked `i128`, including shared gather offsets.
Only bounded batch and storage-block dimensions become backend `usize` indices;
the complete logical shape is never narrowed to construct an ndarray.

Payload and batch collections remain heap-backed with their existing execution
bounds. Large axis descriptors, affine coefficients, and matrix map keys also
remain heap-backed to avoid embedding oversized inline arrays in requests and
futures. SmallVec does not alter the execution bound or promise allocation-free
operation. See [collection rules](CODE_STYLE.md#collections-and-geometry).
