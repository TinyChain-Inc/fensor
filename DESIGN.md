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
Consumption evaluates bounded intermediates through one explicit stack of native
continuations. Each elementwise operation realizes its numerical result before
returning it to the parent, so backend operation depth cannot grow with runtime
expression depth. Reductions, matrix products, and Fourier transforms use the
same process. None may create temporary tensor files or persist computed
intermediates. Persistence requires an explicit write or `Tensor::copy_from`.
Source-cache spill/reload is permitted. No result cache, shared cursor, or
expression registry is part of execution.

## Requests and evaluation

Sparse allocation fixes the sparse-axis block extent at one for new storage.
Persisted layouts retain their recorded extents and share the same access path.
Zero writes validate and clear one offset, then inspect at most one physical
block with rank-sized scratch. Only valid edge coordinates on the key's fixed
sparse-axis coordinate determine whether that key remains populated.

Sparse payload ownership and reclamation belong to the native sparse owner described
below. Its guard coordinates payload, descriptor, and occupancy changes; table and
file guards are released before entering another domain.

Fourier views are bounded evaluation boundaries like reductions and matrix
products. Each transformed axis is capped at `MAX_BATCH_ELEMENTS`; complete
axis groups pack into requests within that same limit. Request-sized group keys
and scatter positions retain output order and duplicates. Source packs execute
sequentially inside the outer consumer, using the shared evaluator and backend
buffer FFT. Backend planning and scratch allocations are bounded by the capped
transform size, separately from fensor's input/output batches.

Output transforms map back to original Fourier groups. Sparse group support is
the union of input support, independent of numerical results. Two-dimensional
transforms compose these boundaries; nested evaluation can recompute earlier
groups and has no result cache or intermediate files.

The private expression contract accepts one `BatchRequest` containing at most
`MAX_BATCH_ELEMENTS` logical elements:

| Representation | Metadata bound |
|---|---|
| Linear span | Start/count; coordinates use rank-sized scratch |
| Cartesian rectangles | Bounded collection of rank-sized axes; intervals or bounded explicit indices |
| Flat arithmetic runs | At most one signed start/step/count run per batch element |
| Explicit coordinates | At most one rank-sized coordinate per batch element |

Requests preserve order and duplicates. A coordinate cursor writes into reusable
scratch. Unary, binary, and conditional nodes share the same immutable request
through `Arc`; they do not copy its descriptors or expand rectangles into coordinate
lists when submitting child work. Coordinates are expanded only at consumers requiring explicit coordinates or the public coordinate-stream boundary.
Owned-expression mappings share affine progression planning with storage leaves.
They retain signed flat runs through child evaluation; irregular and nested
mappings use reusable coordinate scratch and coalesce singleton runs.
Value-only output discards the request without constructing output coordinates.
Sparse output reuses the request cursor and allocates coordinates only for emitted
nonzero values; explicit selections transfer their existing coordinate allocations.

`evaluate_batch` is the common numerical evaluation boundary. It validates request
cardinality before expression construction, array size and support-mask length
before backend evaluation, and returned values afterward. Union/filter operations
validate lengths so `zip` cannot silently truncate. Violations are structured
`InvalidLayout` errors, not debug-only assertions. Dense support uses no mask;
sparse support survives intermediate zeros independently of numerical values.

The evaluation driver owns suspended parent futures and polls only its active
frame. A child request suspends its parent and appends one frame; storage I/O can
then suspend the driver normally. Operand requests remain sequential inside a
batch. Elementwise kernels use ha-ndarray's `into_read` to retain a native bounded
buffer, including its platform, rather than building another unbounded backend
chain. Owned mappings and diagonal projection forward completed batches without
another numerical realization. Only value consumers convert results to host values.

Frames retain bounded requests and any completed sibling batches until their
parent finishes. Their allocation is checked independently of batch cardinality,
with a private frame limit. The driver separately admits batch payload bytes,
charging dtype width and support before submitting each child. Outputs are
reserved conservatively while their inputs remain live, and the charge follows
a returned batch until it is consumed or dropped. This bounds retained siblings
independently of expression depth. Numerical kernels and conversion scratch are
bounded separately by a fixed multiple of the execution batch; this admission
is not a total-process memory guarantee. Runtime expression construction also checks retained
description size; repeated shared operands count at each occurrence. These are
resource limits, not assumptions about a safe machine-stack depth. Metadata at
owned value boundaries is retained directly, so inspecting dtype, layout, or
shape does not descend through runtime composition.

Cancellation detaches any queued child and drains active frames from child to
parent. Owned expressions detach their shared operands before draining final
owners iteratively; other references remain live. This same ownership path handles
completed, failed, cancelled, and never-polled consumption. There is no spawned
child task, nested executor, or recursive evaluation fallback.

## Request providers and concurrency

Expression methods provide actual bounded request iterators, not strategy tags.
Concrete expressions implement shallow `preferred_step`, `support_step`, and
`selection_step` hooks. Shared traversal consumes these hooks in one direction;
default hooks return a leaf result without reentering traversal. Preferred traversal
returns an iterator or no preference. Matrix products
generate tiles for the consuming expression's current shape, using linear requests
below rank two. Unary/scalar nodes delegate; binary nodes consult left then right;
conditionals consult condition, then, else. The first iterator or error ends the
search. Geometric sources and reductions have no preference, so their consumers
generate linear requests. Reduction output remains a traversal boundary even over
a matrix source. Output mappings are independent of request traversal. Provider
steps use explicit work lists with fallible allocation and the expression
description limit, preserving first-provider and first-error precedence.

Sparse support merging retains one bounded request and reusable coordinate cursor
per leaf, plus one output batch. Implicit requests retain compact metadata rather
than expanding every leaf batch into coordinate lists; explicit requests retain
their caller-provided bounded coordinates. A merge step advances all heads equal
to the preceding output before selecting the next unique coordinate. Request
metadata and rank-sized cursor scratch scale with the admitted leaf count,
separately from the numerical payload admission limit.

One unbuffered stream of evaluation futures retains each request with its evaluated
batch. Outer consumers apply `buffered(num_cpus::get().max(1))` for row-major value,
ordered sparse, and boolean reads, or `buffer_unordered` with the same limit for
coordinate streams and numeric terminals. No task is spawned. Request generation
order therefore does not guarantee coordinate-batch delivery order.

ha-ndarray owns numerical parallelism; fensor owns asynchronous batch concurrency.
The evaluation driver and support merger share a cooperative budget of 128 ready
steps. Frame polls and merger input/head advances consume this budget; exhausting
it wakes the consumer and yields before continuing. This makes ready traversal
cancellable without spawning tasks or changing request buffering. It does not
preempt a synchronous numerical kernel or impose a wall-clock latency bound.

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

The shared reader validates positions into bounded logical-address groups before
I/O. Dense reads retain native borrowed-block validation and scatter. Sparse reads
resolve each needed logical descriptor once and reconstruct at most one bounded
block from typed rows or its dense block file before scattering. Each table or
file guard is released before entering another domain. Unit-stride runs copy
slices; negative, zero, and other strides use the shared scatter loop. Absent
sparse descriptors yield zero; missing or malformed required payloads fail closed.
Sparse payload validation checks padding gaps between valid offsets without a
validity mask. Full blocks need no padding traversal; numerical validation scans
the payload directly. Only construction, replacement, and strict occupancy validation
collect occupied regions; ordinary reads do not choose an encoding or build occupancy.

Native shape helpers describe broadcasting and normalize reduction axes without
narrowing logical dimensions. Callers choose automatic versus explicit broadcasting
and scalar versus tensor results.

`StorageGeometry` owns one rank-sized block cursor. `block_offsets` exposes valid
offsets, skipping padding and preserving logical order.

Matrix diagonal projection maps each bounded output request to correlated source
coordinates using reusable rank-sized scratch and one explicit request. It calls
the shared driver for its source batch, preserving support without another
numerical realization or concurrency boundary. Output transforms use
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

`copy_from` consumes its source once. Dense copying validates bounded coordinate/value
batches and their cumulative count, then groups updates by destination block.
`from_values` and dense copying share first-use file allocation in an unpublished
construction owner. A bounded batch updates existing construction files when
coordinates revisit them; no incomplete-block map is retained. Published dense
replacement instead requires one existing, correctly typed and sized file. Shared
required-payload helpers also serve dense-encoded sparse blocks. Missing required
files return `InvalidLayout`; replacement never creates or repairs them.
`from_sparse_elements` and sparse copying share ordered unique ingestion and omit
final numerical zeros. Input errors retain their original type, and borrowed
streams are pinned locally. No consumer polls its source again after EOF.
A private construction owner withholds metadata until successful completion.
When geometry proves row-major block contiguity, construction retains one block
until complete. Otherwise it stages rows in the destination's native
value table and visits one complete logical block at a time. Both forms publish
occupancy and the final descriptor once, using the ordinary encoding heuristic.
Construction never looks up an old descriptor or deletes absent occupancy; all-zero
completed blocks need no publication. Staged-row completion and fresh-block
construction remain explicit processes, sharing payload and index helpers.
Completed `(block, encoding)` descriptors are appended in ascending bounded batches
using native ordered insertion. Their buffer holds at most `MAX_BATCH_ELEMENTS`
pairs (64 KiB at the current limit), independently of payload buffers. No table
guard spans another table or file operation. This avoids repeated descriptor-page
eviction without speculative reads or cache-policy changes.
Dense conversion removes staged rows through the shared bounded deletion helper.
No incomplete-block map, extra staging table, sorting, or source reevaluation is used.
Cancellation or error never exposes a completed Tensor, and missing metadata makes
strict reopening fail. Callers continue owning cleanup and durability.

Dense copying and native value-buffer writes share request-to-block planning.
Published writes retain their read-modify-write corruption checks. Value-buffer writes validate cardinality before mutation and group at
most one execution batch by logical block, preserving repeated-offset input order.
Native fills validate and replace one block at a time, retaining zero edge padding.
Sparse published replacement and construction share one bounded block analysis
for padding, occupancy, and nonzero count. Each chooses its encoding once after
analysis; strict reopening analyzes stored contents without choosing an encoding. Its geometry visitor
reuses rank-sized coordinate scratch; it allocates no per-element coordinates.
Geometric views expose bounded `BlockUpdates` through `plan_updates` and
`updates_from`. Both use the storage-run planner; affine updates do not expand
coordinate lists. Owned expression updates share completion-order evaluation with
coordinate consumption, retaining source handles until consumed or dropped.
Planning describes addresses only; callers enforce write permissions.
Creation alone initializes dense blocks; mutation fails on missing storage.
Full stored lengths and offsets are validated before mutation.

Each destination update completes before the next source batch is requested.
Memory includes the current batch, its grouped positions/values, and one block
guard in addition to the source's own bounded window. There is no copy lookahead,
cross-batch block cache, or additional worker pool. Errors or cancellation drop
the active operation and source stream. Failed construction remains unpublished;
callers own cleanup, and no write prefix or rollback is promised.

## Bound and policy constants

These are private implementation owners, not public tuning options. Equal values
do not couple independent bounds.

| Meaning | Owner / constant | Current value |
|---|---|---:|
| Execution elements / Fourier axis length | `expression::MAX_BATCH_ELEMENTS` | 4096 |
| Active evaluation frames | `expression::driver::MAX_FRAMES` | 16384 |
| Admitted batch payload bytes per driver | `expression::driver::MAX_LIVE_BATCH_BYTES` | 16 MiB |
| Runtime description admission / planning steps | `expression::traversal::MAX_EXPRESSION_NODES` | 65536 |
| Ready traversal steps before yielding | `expression::driver::READY_WORK_STEPS` | 128 |
| Sparse-index page entries | `tensor::SPARSE_INDEX_PAGE_ENTRIES` | 4096 |
| Values per storage block | `schema::MAX_BLOCK_CAPACITY` | 4096 |
| Sparse-node allocation hint | `schema::SPARSE_INDEX_BLOCK_BYTES` | 4096 |
| Sparse-node retained-memory target | `schema::SPARSE_NODE_MEMORY` | 16384 bytes; even node order derived from row width, cell size, row containers, and child IDs |
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

## Element types and payload ownership

Native metadata owns one geometry grammar. Its decoding context is the caller's
maximum metadata rank; dtype tags, byte codecs, and admission policy remain in
file adapters. This rank limit is independent of tensor block capacity.

Expressions use native backend element types and delegate numerical conversions
and complex operations to ha-ndarray. Real-only capabilities are method bounds,
not runtime dtype dispatch. Complex projections and casts retain source support.
Element traits impose no payload codec: adapters own complex representations and
preserve concrete types through reload. The optional complex feature only enables
additional types/operations; it does not select another execution path.

Batch and storage capacities count elements. Their bounded payload bytes scale
with element width, including both complex components; cache accounting uses the
actual Rust payload size. Typed geometry metadata remains unchanged.

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

## Native storage and sources

Native creation and loading select dense or sparse storage directly. The sparse
owner creates or strictly loads its three native tables from validated geometry;
loading never enters creation code. Both operations use the same geometry metadata
and descriptor, occupancy, and value tables. Creation requires empty delegated
storage and keeps metadata unpublished until construction succeeds. Metadata publication
is creation-only: an existing metadata file is an error, never an overwrite.

`TensorSource` is the statically typed storage boundary for expression leaves.
Native tensors and caller-defined sources use `TensorView<Source>`, the
same geometric mapper, logical-block reader, support rules, occupied traversal,
and expression evaluator. `StorageGeometry` validates and exposes logical block
positions, bounds, and valid block offsets. Owned source scans deliver sorted
unique `(sparse-axis coordinate, logical grid block)` keys with bounded lookahead.
Native page sizes remain implementation details.
Sources release index guards before delivering keys.

Views own their source handles. `TensorExpression<T>` owns a dynamically composed
read-only expression and erases only that value description; concrete storage
remains statically typed. External source handles may retain read leases. Native
clones share live storage; cloning does not create a snapshot.
Execution batches remain distinct from storage blocks. `copy_from` explicitly
creates independent storage using bounded input batches. Borrowed and owned source
handles feed the same evaluation pipeline; each outer consumer owns its sole
buffering boundary. Sparse output flattens bounded batch iterators directly.

`replace_logical_block` owns adaptive sparse publication. Three native tables hold
block descriptors `(block, encoding)`, occupied-region keys, and typed
nonzero rows `(block, offset, value)`. `SparseCell<T>` distinguishes
integer keys from numerical payloads; its storage equality/order preserve scalar
bits, including NaN payloads and complex components. Adapters serialize typed
`SparseNode<T>` nodes with explicit tags and bounded allocations. Table pages
use the existing native index page size and order.

Absent descriptors denote all-zero blocks. Payloads smaller than 1 KiB use rows;
otherwise rows win when `nnz * (24 + size_of::<T>()) < block_len * size_of::<T>()`.
The heuristic chooses representation deterministically on each replacement and
is not an exact serialized-size model. Dense files are named by logical block ID.
Each block exclusively owns its payload even when rows share native pages.
Replacement validates the incoming block, then updates its payload, occupancy,
and descriptor under one ownership guard. One concrete sparse owner
retains geometry, payload files, all three tables, and its invalidation state. Its
shared ownership guard is acquired before native table/file guards; each inner
guard is released before
entering another domain. Guarded entry points check owner health; internal helpers
never reacquire the guard or repeat that check. Each staging call, construction
completion, or published mutation has one failure guard. Replacement validates
its ID and length before storage access, then analyzes padding before persistent
mutation. Sparse read-modify-write holds the ownership guard throughout.
Cancellation or failure during mutation invalidates the owner and requires caller
recovery. Native storage retains only the current payload; it provides no history
or recovery protocol.

`create_with_geometry` accepts validated logical geometry; ordinary creation
chooses geometry and delegates. Physical packing never enlarges logical blocks.
Metadata records shape, layout, sparse axis, and block shape.
Native node orders remain storage invariants because splitting and deletion depend on them.
Adapters retain explicit typed payload tags.

Creation requires empty delegated storage. Loading requires typed metadata,
directories, every dense block, and all three native sparse tables. It validates native
structure, logical key uniqueness, coordinates, offsets, occupancy, typed payloads, and block
lengths without creating or rewriting files. `sync` writes native blocks and the
sparse index root; `sync_all` additionally synchronizes delegated storage durably.
The sparse owner holds its guard through native synchronization and publishes all three
table roots before durable directory synchronization. Neither method establishes visibility
or records a commit decision. Failed native
mutation or materialization may leave partial state; callers coordinate recovery.

`tests/storage_source.rs` checks owned external-source composition, consumer parity,
and strict reopen. Native `tensor::physical_tests` and `tensor::dtype_storage` own
corruption, logical replacements, and shared-page isolation regressions. General
public sparse streams use indexed traversal for compatible geometry; indexed numeric reductions
do not imply indexed execution for every operation or geometric view.


## Ordered support and statistics

Ordered sparse consumers share native support request providers. Storage uses
paginated occupied keys when sparse-axis ordering and contiguous storage regions
match logical row-major order. Forward separable slices preserving axis order
reuse that traversal. Unary operations delegate support, while binary and
conditional operations combine candidate sources through explicit traversal steps.
The resulting union consumes a flat list of leaf streams, retaining one bounded
request and lookahead per leaf plus one output batch. Polling and dropping a union
do not recurse through the expression. Provider order and ordered delivery remain
explicit; candidate support is independent of evaluated numerical zeros, and
filtering occurs at the public sparse output boundary.

Geometries whose index order is incompatible (including general reordered,
negative-stride, broadcast, and gather mappings), transformed owned expressions,
and aggregate outputs use bounded logical evaluation. Their cost may be
proportional to selected logical cardinality. There is no external sort, result
cache, or implicit persisted intermediate. Eligible native sparse copies consume
ordered entries and reuse the existing bounded destination batch writer.

Mean, population standard deviation, and Euclidean norm reuse numeric terminal
batching and axis-group traversal. Counts include supported intermediate zeros
and exclude implicit sparse zeros. Partial statistics retain constant-sized
state; standard deviation combines centered moments and uses squared complex
magnitude. Real results are f64; complex mean is c64 and complex standard
deviation and norm are f64. Empty support yields NaN for mean/std and zero for
norm. Nonfinite arithmetic and completion-order reproducibility limits remain
part of numerical evaluation, not transaction visibility.

`TensorView::logical_block_range` derives a conservative half-open grid interval
from mapping metadata. Points cover one block. Explicit gathers are inspected as
caller-owned metadata; implicit logical selections are never expanded. Gaps may
be included. Callers own reservations and publication; fensor owns neither.

### Native ordered ingestion and adapter writes

Contiguous sparse construction streams typed rows through `b-table::upsert_sorted`
and `b-tree::insert_sorted`. The tree retains only its rightmost leaf identifier
and previous key; ordinary insertion owns splitting and ancestor updates.
Interleaved construction continues staging in the destination value table.
Neither path sorts or collects the source. Row order and sparse block geometry
are independent of native page capacity. The node-memory target estimates a full
index node; vector spare capacity and storage cache residency remain separate costs.

File adapters report retained allocation through `GetSize` and inspect encoded
containers with bounded scratch in `FileLoad::load_size` before decoding payloads.
Admission counts retained typed capacities, independently of serialized bytes;
adapters bound temporary decoder allocations separately. Native writes reserve
replacement capacity before mutation and retain their existing admission for
updates which do not grow the payload. The caller's cache must also fit
simultaneously pinned native split/merge pages; the node memory target is not the
minimum usable whole-cache capacity.

File adapters should coalesce small codec chunks with bounded buffered writes and
flush before returning from `FileSave`. This does not change encoded bytes or make
`sync` durable; delegated `sync_all` retains that responsibility.

Sparse reopening shares payload and occupancy analysis with replacement, without selecting an encoding. Ordinary reads validate payloads without allocating occupancy. Descriptor replacement delegates primary-key replacement to native Table; zero blocks delete their descriptor.

Occupied scans retain their cursors and bounded native page lookahead until consumed or dropped. Expression consumers keep the scan across output batches. Native pages validate complete index rows, including marker, ordering, and block membership, before returning keys. Consumers may trust this provenance without a second membership lookup.

Dense validation borrows each required payload and checks its length without cloning it.
Sparse orphan validation checks each ordered block group once, retaining only
the current group across bounded pages.

Sparse partial updates retain the descriptor returned by their validated payload read
through replacement under the same ownership guard; they do not resolve it again.
Native fills traverse valid block offsets directly while retaining old-payload
validation and zero padding.
