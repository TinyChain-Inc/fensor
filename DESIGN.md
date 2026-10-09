# Bounded execution design

This is the implementation reference for fensor's execution invariants. Public
usage and sparse semantics are in [README.md](README.md); benchmark methodology
and interpretation are in [BENCHMARKS.md](BENCHMARKS.md). Inline documentation owns
individual field and function invariants.

## Resource and persistence contract

Tensor execution must not allocate values, coordinates, or
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
The copy caller selects destination layout; reader layout never selects allocation
implicitly. Geometry validates block IDs, lengths, and padding for native storage
and caller-owned logical-block replacements; storage owners retain payload and
mutation validation.

The private evaluation driver is a trampoline using an explicit continuation
stack: concrete operations request children without recursively polling them.
Request discovery has its own iterative traversal; owned expressions detach their
operands for iterative destruction. Boxing futures does not flatten polling or
destruction, and Tokio cooperation controls fairness rather than stack depth.

Source-cache spill/reload is permitted. No result cache, shared cursor, or
expression registry is part of execution.

## Requests and evaluation

Sparse geometry gives each prefix coordinate a full dense trailing region,
independent of the physical chunk capacity. Zero writes validate their chunk and
remove its row when all values are zero. The native sparse
owner coordinates payload rows, reclamation, and synchronization under one guard;
inner table guards are released before entering another ownership domain.

Fourier views are bounded evaluation boundaries like reductions and matrix
products. Each transformed axis is capped at `MAX_BATCH_ELEMENTS`; complete
axis groups pack into requests within that same limit. Request-sized group keys
and scatter positions retain output order and duplicates. Source packs execute
sequentially inside the outer consumer, using the shared evaluator and backend
buffer FFT. Backend planning and scratch allocations are bounded by the capped
transform size, separately from fensor's input/output batches.

Output transforms map back to original Fourier groups. Sparse absent values are
zero inputs to the same kernel. Two-dimensional transforms compose these
boundaries; nested evaluation can recompute earlier groups and has no result cache
or intermediate files.

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
cardinality before expression construction, array size before backend evaluation,
and returned values afterward. Violations are structured `InvalidLayout` errors,
not debug-only assertions. Both layouts carry the same numerical batches; no
numerical support mask is retained.

The evaluation driver owns suspended parent futures and polls only its active
frame. A child request suspends its parent and appends one frame; storage I/O can
then suspend the driver normally. Operand requests remain sequential inside a
batch. Batch construction completes numerical operations with ha-ndarray's
`into_read`, retaining a native bounded buffer and its platform rather than an
unbounded backend chain. Host-value construction uses its already completed
buffer directly. The driver attaches a private batch reservation; concrete
operations do not manage admission bookkeeping. Owned mappings and diagonal
projection forward completed batches without another numerical realization.
Only value consumers convert results to host values.

Frames retain bounded requests and any completed sibling batches until their
parent finishes. Their allocation is checked independently of batch cardinality,
with a private frame limit. The driver separately admits batch payload bytes,
charging dtype width before submitting each child. Outputs are
reserved conservatively while their inputs remain live, and the charge follows
a returned batch until it is consumed or dropped. This bounds retained siblings
independently of expression depth. Tokio owned semaphore permits account for
admitted bytes and return capacity when batches are dropped, including cancellation
and unwinding. Acquisition is
nonblocking: waiting for capacity while retaining intermediate results could
prevent those results from being released. The semaphore is independent of
pending evaluation frames because completed batches can outlive the driver.
Numerical kernels and conversion scratch are bounded separately by a fixed
multiple of the execution batch; this admission
is not a total-process memory guarantee. Runtime expression construction also checks retained
description size; repeated shared operands count at each occurrence. These are
resource limits, not assumptions about a safe machine-stack depth. Metadata at
owned value boundaries is retained directly, so inspecting dtype, layout, or
shape does not descend through runtime composition.

Cancellation detaches any queued child and drains active frames from child to
parent. `TensorExpression` owns its destructor directly. Its private optional
source is a detachable slot, emptied before draining shared operands; detached
values drop harmlessly. Consuming geometric transforms clone rank-sized mapping
metadata because Rust prevents moving fields out of a Drop owner. Gather payloads
remain shared, and tensor values are never cloned by these transforms.
Final owners are drained iteratively; other references remain live. This path handles
completed, failed, cancelled, and never-polled consumption. There is no spawned
child task, nested executor, or recursive evaluation fallback.

## Request providers and concurrency

Expression methods provide actual bounded request iterators, not strategy tags.
Concrete expressions implement shallow `preferred_step`, `ordered_step`, and
`selection_step` hooks. A shared bounded walker consumes preferred and ordered
hooks left to right;
default hooks return a leaf result without reentering traversal. Preferred traversal
returns an iterator or no preference. Matrix products
generate tiles for the consuming expression's current shape, using linear requests
below rank two. Unary/scalar nodes delegate; binary nodes consult left then right;
conditionals consult condition, then, else. The first iterator or error ends the
search. Geometric sources and reductions have no preference, so their consumers
generate linear requests. Reduction output remains a traversal boundary even over
a matrix source. Output mappings are independent of request traversal. The walker
owns deferred-child ordering, fallible stack allocation, and the
expression traversal limit. Its consumers retain first-provider selection and
ordered union respectively; single-child selection remains an explicit loop.
Concrete operations own mathematics, native storage owns address discovery,
and shared consumers own bounded stack-safe execution.

Ordered providers use the same operand order as evaluation: left before right,
and condition before then and else. One work list flattens their leaf streams;
there is no separate error-ordering tree.

Sparse occupied-candidate traversal retains one bounded request and reusable
coordinate cursor per leaf, plus one output batch. Each cursor supplies row-major
positions to collate's flat ordered union; only distinct output positions become
coordinates. Implicit requests retain compact metadata, and explicit requests
retain their caller-provided bounded coordinates. Request metadata and rank-sized
scratch scale with admitted leaf count, separately from numerical payload admission.
Both union polling and empty-request traversal yield cooperatively.

Borrowed and owned block streams share request selection and ordered consumption;
the owned stream retains its expression through an `Arc`.

One unbuffered stream of evaluation futures retains each request with its evaluated
batch. Outer consumers apply `buffered(num_cpus::get().max(1))` for row-major value,
ordered sparse, and boolean reads, or `buffer_unordered` with the same limit for
coordinate streams and numeric terminals. No task is spawned. Request generation
order therefore does not guarantee coordinate-batch delivery order.

ha-ndarray owns numerical parallelism; fensor owns asynchronous batch concurrency.
The evaluation driver and occupied-candidate adapters cooperate with Tokio's
per-task scheduler budget. Completed frames, queued children, and candidate
advances record progress; pending operations refund their cooperation attempt.
Tokio owns accounting and wakeups, with no fensor scheduling counter. This keeps
ready traversal cancellable on Tokio without spawning tasks or changing request
buffering. Outside Tokio (or in an unconstrained task), these hooks do not enforce
fairness. They do not preempt a synchronous numerical kernel or impose a
wall-clock latency bound.

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
read a bounded dense vector from its typed native table row before scattering.
Each table or file guard is released before entering another domain. Unit-stride
runs copy slices; negative, zero, and other strides use the shared scatter loop.
Missing sparse chunks yield zero; malformed stored chunks fail closed.
Payload validation checks padding gaps between valid offsets without a validity
mask. Full blocks need no padding traversal.

Native shape helpers describe broadcasting and normalize reduction axes without
narrowing logical dimensions. Callers choose automatic versus explicit broadcasting
and scalar versus tensor results.

`StorageGeometry` owns one rank-sized block cursor. `block_offsets` exposes valid
offsets, skipping padding and preserving logical order.

Matrix diagonal projection maps each bounded output request to correlated source
coordinates using reusable rank-sized scratch and one explicit request. It calls
the shared driver for its source batch, preserving values without another
numerical realization or concurrency boundary. Output transforms use
the existing coordinate map. Diagonals use logical slice traversal; matrix
transpose delegates to the existing final-axis permutation.

## Slices and reductions

A validated slice retains source rank and compact axis descriptors, checking rank,
bounds, steps, endpoints, and cardinality with checked arithmetic. Iteration
preserves explicit order/duplicates, handles empty selections, and stays exhausted.
Whole reductions use a full slice; axis groups fix unreduced coordinates.

Slices of at most one execution batch retain direct requests. Larger eligible
sparse slices stream existing logical block IDs, intersect their bounds, and reuse
the ordinary reader for values. Conservative block ranges can scan more occupied
metadata than they consume. Keys/intersections are validated before
data I/O; unrelated data blocks are not read merely to test intersection.
Index pages are bounded and release guards before value reads. Small intersecting
regions pack into bounded batches, with one overflow rectangle retained.

Base-coordinate slices, forward separable mappings, and supported permutations
can use indexed traversal. Gathers, reversals, broadcasts, incompatible reshapes,
and aggregate sources use compact logical requests. Eligible binary and conditional
sources merge their ordered candidates.
Unary/scalar nodes inherit their source's slice requests and evaluate the complete
expression; they never consume an intermediate zero-filtered reader. Boolean
terminals deliberately use logical requests to preserve error/decision order.

Reductions consume every selected logical value, including implicit sparse zeros.
Indexed traversal evaluates candidate coordinates once and accounts for the
remaining cardinality through zero partials, without expanding absent coordinates.
Each pointwise node caches the typed result
of its operation on its operands' implicit zeros. This constant metadata preserves
backend zero signs when aggregating gaps; it is not an element mask or occupancy
history. ha-ndarray reduces each bounded batch; its scalar arithmetic/extrema rules
combine partials. Seed from the first partial to preserve signed-zero behavior. Complete small groups share one source
request and bounded segment boundaries, then delegate each nonempty group directly
to the operation's partial and finish methods. Terminals and long groups share one
active accumulator and chunked inputs, never a partial list. Requested axis outputs
are processed sequentially within each output batch. Output transforms map reduced coordinates
and do not move through source axes. Selected output ranges still require their
source groups.

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

Output mapping, tiling, kernels, and contraction order are shared by both layouts.
Implicit zeros participate in numerical arithmetic, including zero times infinity
or NaN. Contraction scans scale with logical length; nested products and separate
output batches may reevaluate operands.
Ordered wide-output streams limit cross-row reuse; tiled coordinate requests
improve reuse without guaranteeing locality after arbitrary transforms.

## Destination writes

`copy_from` consumes its source once. Dense copying validates bounded coordinate/value
batches and their cumulative count, then groups updates by destination block.
`from_values` and dense copying share first-use file allocation in an unpublished
construction owner. A bounded batch updates existing construction files when
coordinates revisit them; no incomplete-block map is retained. Published dense
replacement instead requires one existing, correctly typed and sized file. Missing required
files return `InvalidLayout`; replacement never creates or repairs them.
`from_sparse_elements` and sparse copying share ordered unique ingestion and omit
final numerical zeros. Input errors retain their original type, and borrowed
streams are pinned locally. No consumer polls its source again after EOF.
A private construction owner withholds metadata until successful completion.
When geometry proves row-major block contiguity, sparse construction retains one
dense block until complete and streams completed rows into native ordered ingestion.
Only the current block and the bounded input batch remain live. Interleaved input
groups each bounded batch by block
and updates the destination's existing typed rows. All-zero chunks are omitted;
successful completion publishes metadata. There is no
incomplete-block map, extra staging table, sorting, or source reevaluation.
Cancellation or error never exposes a completed Tensor, and missing metadata makes
strict reopening fail. Callers continue owning cleanup and durability.

Dense copying and native value-buffer writes share request-to-block planning.
Published writes retain their read-modify-write corruption checks. Value-buffer writes validate cardinality before mutation and group at
most one execution batch by logical block, preserving repeated-offset input order.
Native fills validate and replace one block at a time, retaining zero edge padding.
Sparse replacement, construction, and strict reopening share bounded payload
validation. The geometry visitor reuses rank-sized coordinate scratch without
per-element allocations.
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
| Sparse-index page entries | `tensor::SPARSE_INDEX_PAGE_ENTRIES` | 4096 |
| Values per storage block | `schema::MAX_BLOCK_CAPACITY` | 4096 |
| Sparse-node allocation hint | `schema::SPARSE_INDEX_BLOCK_BYTES` | 4096 |
| Sparse-node retained-memory target | `schema::SPARSE_NODE_MEMORY` | 16384 bytes; even node order derived from dense payload bytes, row containers, cells, and child IDs |
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
not runtime dtype dispatch. Complex projections and casts share numerical kernels
across layouts.
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
owner creates or strictly loads one typed native table from validated geometry;
loading never enters creation code. Creation requires empty delegated storage and
withholds metadata until construction succeeds. Metadata publication is
creation-only: existing metadata is an error, never an overwrite.

`TensorSource` is the statically typed storage boundary. Native tensors and caller
sources share `TensorView<Source>`, geometry, logical reads, occupied traversal,
and numerical evaluation. `StorageGeometry` exposes logical block positions,
bounds, and valid offsets. `occupied_blocks(range)` delivers sorted unique logical
block IDs within a half-open range, with bounded lookahead, and releases index guards
before delivery. Resume after an ID with `id + 1..range.end`. Page sizes remain private.

Views own source handles. `TensorExpression<T>` erases only the owned read-only
value description; storage remains statically typed. Its lazy `into_dense()`
conversion includes implicit zeros without copying storage or changing shared
clones. Nonidentity output mappings conservatively report scalar sparse layout
instead of retaining a stale dense-suffix axis. External handles may retain read
leases. Native clones share live storage and are not snapshots. Borrowed and owned
consumption use one evaluator and one outer buffering boundary.

For `Layout::Sparse { axis: Some(a) }`, axes through `a` select a full dense
trailing region. Physical chunks have prefix extent one and bounded trailing
extents. `axis: None` denotes scalar sparsity. `create_with_geometry` accepts
validated geometry; normal creation chooses geometry and delegates. Metadata
records shape, layout, sparse axis, and physical block shape. Physical chunking
does not change logical values or the dense trailing region.

Each sparse table row stores a logical block ID and dense typed vector.
`SparseCell<T>` separates integer keys from typed payloads;
its equality and collation preserve exact scalar representations, including NaN
bits and complex components. Adapters own `SparseNode<T>` serialization, explicit
payload tags, and bounded decoding allocations. Rows share native table pages;
there is no separate descriptor, occupancy table, or payload-file representation.

An absent physical chunk is all-zero. Removing a chunk's last nonzero value
removes its row. Invalid padding, malformed lengths, empty stored chunks, and
incorrect block IDs are errors. Batch reads share a native table range for
consecutive requested block IDs. Gaps retain separate reads so unrelated payload
corruption remains outside the selection. Replacement and read-modify-write
hold one native ownership guard; helpers never reacquire it. Table guards are
released before entering another domain. Failed or cancelled mutation may leave partial changes. Callers must discard
all handles to the affected storage and coordinate recovery before reuse. Native
storage retains no history or recovery protocol.

The sparse ownership lock coordinates native access without tracking recovery
state. Payload validation still reports corruption. Transaction abort-only state
belongs to callers such as tc-collection; fensor neither marks clones invalid nor
repairs partial writes.

Strict loading requires typed metadata, directories, every dense file, and a valid
sparse table. It validates structure, ordering, geometry, payloads, and block IDs
without creating or repairing storage. `sync` writes native payloads and the sparse table root; `sync_all` additionally performs delegated
durable synchronization. Neither method creates transaction visibility or records
a commit decision. Callers own recovery from failed materialization.

`tests/storage_source.rs` owns external-source composition and consumer parity.
`tests/dense_sparse_rows.rs` exercises sparse-axis geometry, omitted zero chunks,
nonfinite values, reopening, and explicit dense conversion. Independent corruption,
cancellation, pagination, and lifecycle fixtures retain their owning assertions;
see [test ownership](tests/COVERAGE.md).

## Ordered sparse traversal and statistics

Ordered sparse consumers share occupied-candidate request providers. Storage uses
paginated occupied block IDs when their bounds match logical row-major order.
Forward separable slices preserving axis order
reuse that traversal. Eligible unary operations delegate candidates, while binary and
conditional operations combine candidate sources through explicit traversal steps.
The resulting union consumes a flat list of leaf streams, retaining one bounded
request and lookahead per leaf plus one output batch. Polling and dropping a union
do not recurse through the expression. Provider order and ordered delivery remain
explicit; candidate occupancy is not numerical support provenance. Numerical zeros
are filtered at the public sparse output boundary. Pointwise construction rejects
operations which change the implicit zero background with `Error::WouldDensify`;
explicit `TensorExpression::new(source)?.into_dense()` permits full evaluation.

Geometries whose index order is incompatible (including general reordered,
negative-stride, broadcast, and gather mappings), transformed owned expressions,
and aggregate outputs use bounded logical evaluation. Their cost may be
proportional to selected logical cardinality. There is no external sort, result
cache, or implicit persisted intermediate. Eligible native sparse copies consume
ordered entries and reuse the existing bounded destination batch writer.

Mean, population standard deviation, and Euclidean norm reuse numeric terminal
batching and axis-group traversal. Counts include all logical values, including
implicit sparse zeros. Partial statistics retain constant-sized state; standard
deviation combines centered moments and uses squared complex magnitude. Real
results are f64; complex mean is c64 and complex standard deviation and norm are
f64. Nonfinite arithmetic and completion-order reproducibility limits remain
part of numerical evaluation, not transaction visibility.

`TensorView::logical_block_range` derives a conservative half-open grid interval
from mapping metadata. Points cover one block. Explicit gathers are inspected as
caller-owned metadata; implicit logical selections are never expanded. Gaps may
be included. Callers own reservations and publication; fensor owns neither.

### Adapter writes

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

Occupied scans retain cursors and bounded native page lookahead until consumed or
dropped. Expression consumers retain the scan across output batches. Native pages
validate ordered block IDs before returning keys; consumers need no second
membership lookup. Payload reads validate the selected dense chunks. Dense
validation borrows required payloads without cloning them. Native fills visit valid offsets while retaining old-value
validation and zero padding.
