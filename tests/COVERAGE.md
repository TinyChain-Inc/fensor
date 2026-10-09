# Regression ownership

Keep broad numerical cases separate from expensive full-consumer parity. Shared
filesystem scaffolding lives in `common/mod.rs`, imported directly by unit tests.
Special codec/error/independent-adapter fixtures remain local to their contracts.

| Contract | Primary coverage |
|---|---|
| Public downstream composition, geometric mutation, logical-block access, copy/sync/reload | `handoff` (f32 dense/trailing-axis sparse); no transaction-readiness claim |
| Logical u64 geometry, inline rank spilling, overflow, large-coordinate persistence | `geometry`; request/mapping/storage-run unit bounds |
| Matrix transpose/diagonal dtype and consumer parity, output transforms, huge selected reads, corruption, cache pressure, cancellation and write constraints | `matrix_unary`; matrix unit request counters; diagonal doctests |
| Matrix shapes, dtypes, operand layouts and backend agreement | `matmul` numerical parity matrix, block streams |
| Matrix point/sparse/copy/sync/reload parity | Every dtype/layout pair on `(2,3,2)`; f32 dense/dense and sparse/sparse on `(33,129,35)`; all large pairs retain boundary point checks |
| Reduction numerical lengths, operations, axes and keepdims | `reduce` parameterized block/terminal parity |
| Reduction full consumer/persistence parity | Sum at length 7 and axis 1 of `[2,3,4]` with both keepdims values for every dtype/layout; sum at length 4097 for f32 dense/sparse; all operators retain numerical and terminal assertions |
| Matrix transforms, batches, nesting, exceptional values and exact bounds | Dedicated `matmul` cases, retained independently |
| Sparse slice dtype semantics and persistence | `slice_reductions` small f32/f64/u8 fixtures |
| Multi-node sparse index topology across axes/capacities/shapes | `slice_reductions::indexed_topology` (f32) |
| Exclusive block-ID continuation at the index-page limit | Tensor unit `occupied_block_pages_resume_after_delivered_ids` |
| Matrix, diagonal, and Fourier sparse consumer delegation | `sparse_consumers`: transformed outputs, cross-batch selections, zeros, nonfinite values, empty sparse storage |
| Sparse output order, duplicates, final zeros, NaN and lengths | Expression unit `sparse_output_*` |
| Request/affine/irregular planning bounds and block grouping | Request, storage-read, matrix and tensor unit cases; mapping compact-run/reference parity and view owned-consumption counters |
| Valid block offset order, shared payload validation, edge padding, high-rank scratch reuse | Storage geometry traversal tests; sparse replacement lock/health precedence fixture; collection canonical-delta padding fixture |
| Ordinary implicit zeros, dense-equivalent statistics, explicit dense conversion and bounded groups | `dense_sparse_rows`; `sparse_semantics`; reduction zero-gap integration/unit cases |
| Expression zero signs during indexed gap aggregation | Reduction unit `zero_gap_products_preserve_expression_signed_zeros` |
| Corruption, boolean decision boundaries, cache pressure, cancellation and live reads | Native `tensor::physical_tests` and `tensor::dtype_storage`; dedicated copy cases |
| No implicit result storage and transformed leaf parity | `storage_runs` |
| Recursive operand order, first-error propagation, and final-argument ownership | Mapping unit `recursive_transforms_stop_at_first_error_and_move_the_final_argument` |
| Transform input rejection and structured sparse-order errors | Public access integration cases; view unit tests retain mapping/write-through contracts |
| Sparse axis bounds on creation and reload | `foundations::public_create_rejects_invalid_sparse_axis_hint`, `storage_codec::reload_rejects_out_of_bounds_sparse_axis` |
| Shared preferred/ordered walker: precedence, errors and single invocation | Expression units `request_providers_delegate_once_in_order_and_propagate_errors`, `ordered_providers_visit_operands_left_to_right` |
| Tiled propagation, reduction boundaries and transformed current-shape coverage | Matrix `coordinate_traversal_propagates_and_preserves_coverage` |
| Full-shape linear batches and invalid shapes | Request unit `linear_batches_preserve_boundaries_and_validate_shape`; reduction tests retain boolean error boundaries |
| Request-to-update parity, affine allocation bounds, owned source consumption and release | View unit `update_planning_matches_coordinates_without_affine_expansion`, `update_stream_evaluates_once_and_owns_its_source` |
| Bulk block publication counts, edge padding, duplicate writes, NaN, and missing-block mutation rejection | `tensor::physical_tests::bulk_mutation` |
| Sequential copy backpressure, drop/error cancellation | `copy_pipeline` with a held filesystem block guard |
| Ordered batch completion and concurrency bound | Expression unit `buffered_batches_are_bounded_ordered_and_cancelled_by_drop` for borrowed and owned handles |
| Completion-order progress, slot replenishment, one-slot window, pairing and cancellation | Expression unit `completion_order_replenishes_slots_and_preserves_pairs` |
| Dynamic expression depth on normal worker stacks, shared operands, sparse values, cancellation/error and unpolled-drop source release | `expression_depth` isolated subprocesses in default and complex suites; also run the target in release |
| Wide sparse candidate merging, high-rank compact request retention, duplicates, empty requests, EOF/error/cancellation cleanup | Expression traversal `wide_sparse_candidates_retain_compact_requests_and_release_sources`, `ready_sparse_candidates_can_be_cancelled_and_errors_release_sources` |
| Permit refunds, closed admission, and cleanup during unwinding | Expression driver units `batch_admission_follows_retained_values_and_refunds_on_drop`, `admission_refunds_during_unwind_and_rejects_closed_budget`, and `poisoned_driver_rejects_polling_and_releases_pending_work` |
| Ready expression cancellation and peer-task progress | `expression_depth` ready-cancel subprocess case; expression traversal cancellation test |
| Owned/borrowed consumer parity, transformed values and nonfinite values | `storage_source::owned_and_borrowed_consumers_preserve_geometry_zeros_and_nonfinite_values` |
| Owned source release after errors | Expression unit `owned_stream_error_releases_pending_evaluation_on_drop` |
| First observed errors and pending evaluation cancellation | Expression unit `unordered_errors_cancel_pending_evaluation` |
| Numeric terminal completion schedules, wrapping and aggregate accuracy | Expression unit `numeric_terminals_accept_completion_schedules` |
| Fixture directory uniqueness and scoped cleanup | `foundations::temporary_directory_names_are_unique_across_threads`, `fixture_directories_are_removed_on_completion_and_unwind` |
| Public type restrictions and no adapter Clone requirement | Doctests |
| Composition and coordinate-stream consumption example | Crate-level compile-checked doctest in `src/lib.rs` |

Do not replace topology/pagination tests with dtype permutations, or structural
assertions with timing thresholds. Ignored benchmarks share fixtures but are not
ordinary regression tests. [BENCHMARKS.md](../BENCHMARKS.md#measurement-and-reproduction)
describes the shared workloads, two entrypoints, smoke instructions and record schema. Smoke runs are not performance evidence.

Routine checks use `cargo test`; benchmark harnesses compile only with the
`benchmarks` feature. See [CONTRIBUTING.md](../CONTRIBUTING.md) for targeted
checks and [BENCHMARKS.md](../BENCHMARKS.md) for opt-in smoke/profile runs.

## Concrete dtype coverage

- The `tensor::sparse_lifecycle_tests` module in the Tensor test directory owns same-key zero writes, persisted block geometry,
  edge padding, shared pages, interrupted publication, and capacity/axis
  allocation. `tensor::dtype_storage` checks geometric and bulk zero writes for every
  supported dtype. Existing storage/planner tests retain logical block grouping coverage
  with explicitly persisted layouts.

- `fourier` owns complex conjugate transpose, independent forward/inverse Fourier
  expectations, propagated round-trip bounds, consumer parity, sparse values,
  output transforms, corruption and axis-limit rejection. Fourier unit tests own
  complete-group packing and duplicate scatter.

- `tensor::dtype_storage`: all concrete dtypes in dense/sparse storage, spill/reload,
  write-through, zero lifecycle, copy persistence, malformed blocks, and typed
  metadata mismatch, concrete tensor/view/schema classes, typed metadata shape,
  and complex feature gating. `number_type` owns abstract-class rejection and
  computed-output classes.
- `integer_types`: every integer width's operation families, axis/terminal
  reductions, matrix consumers, wrapping boundaries, and signed negative powers.
- `all_casts`: every enabled source/destination pair against backend buffers and
  the number-general scalar pipeline, reusing one source per dtype/layout.
  Literal conversion fixtures independently check fensor point and block reads.
- `complex_types`: complex operation families, component projections, signed zeros,
  branch cuts, exceptional predicates and concurrent/cancelled
  consumption across batch boundaries. Existing float suites retain their accuracy
  and exceptional-value ownership. Public doctests own capability restrictions.

The `handoff` smoke test reuses public APIs only. Broader transform coverage stays
with `access_matrix` and mapping tests; independent codecs with `storage_codec`;
sparse lifecycle and corruption with `tensor::dtype_storage` and tensor unit tests;
bounded requests and execution with the structural suites listed above.

- `mapping::traversal_tests` covers mapped request order, duplicates, empty requests,
  prevalidation, early errors, reusable scratch beyond inline rank capacity, and
  singleton-axis identity preservation for indexed request planning.

## Test consolidation policy

- Elementwise numerical matrices retain every operation, dtype, operand-layout,
  and scalar case through ordered block comparisons. Full point/sparse/copy/sync/
  reopen checks use representable pointwise operations and conditional selection.
  Densifying scalar operations instead require structured rejection and explicit
  dense-conversion checks. Dedicated sparse, exceptional
  value, cache-pressure, cancellation, and persistence regressions remain separate.
- `common::Directory` owns temporary storage for default-cache, custom-cache, and
  independent-codec fixtures. `common::fixture` owns native stream construction and
  consumer assertions. Helpers return the directory guard alongside each tensor;
  numerical setup uses construction while mutation tests retain explicit writes.
  `common::numbers::same` owns exact values, NaN classification, and signed zeros.
  Tolerance-based numerical assertions remain operation-specific. Predicate tests
  use the same consumer assertions while retaining their truth tables and densification errors.
- `math_trig` compares every abs/trig operation, float dtype, and layout with the
  backend through ordered blocks, including nonfinite values and signed zeros.
  Sine supplies shared point/sparse/copy/sync/reopen parity per dtype/layout;
  the separate composed-chain case retains values, transform, and drop coverage.
- `access_matrix` parameterizes dense/sparse slice, transpose, chained transpose/
  slice, squeeze, and unsqueeze while retaining original geometry, sparse axes,
  capacities, and values. Dense reload and sparse index-update reload stay distinct.
- `tensor::dtype_storage::f64_storage` includes PI/E values and full schema checks
  after reopening; it subsumes the former standalone f64 roundtrip smoke test.

## Native storage ownership

| Contract | Primary coverage |
|---|---|
| Absent-zero, insertion, replacement, deletion, reinsertion, block presence and stable logical addressing | `sparse_point_mutation_lifecycle`; shared-page, edge and interrupted-publication fixtures remain separate |
| Every dtype/layout, borrowed non-Unpin input, schema/type checks, write-through, original mutation/reopen, transformed copy/reopen and malformed payloads | `tensor::dtype_storage`; full block comparisons, exhaustive sparse point/entry checks, dense point probes at special values, every block boundary and the modified final element |
| Dense trailing regions, scalar sparsity, omitted zero chunks, edge padding, exact nonfinite bits, capacity-independent values and reopening | `dense_sparse_rows` parameterized cases |
| Lazy dense conversion, unchanged operand clones, transformed layout and implicit zeros across batch boundaries | `dense_sparse_rows::lazy_dense_conversion_includes_implicit_zeros_across_batches` and companion geometry case |
| Copy batch boundaries, explicit destination layout, pre-consumption rejection, independent geometry and exceptional values | `tensor_copy`; collection copy policy fixture retains axis hints; specialized errors and cancellation remain separate |
| Typed source errors, cardinality, coordinate bounds/order/duplicates, cancellation, incomplete metadata and strict rejection | `tensor::construction::tests`; interrupted staging, publication and completion release guards |
| Replacement ID-before-length validation, missing/wrong-type/malformed payloads remain unrepaired | Native replacement fixtures, with borrowed dense validation |
| Creation-only metadata publication preserves existing contents | Native metadata tests |
| Occupied-key validation and full cursor continuation | Native page tests |
| Shared metadata fields and extra-field rejection | `storage_codec` |
| Ordered insertion errors, duplicates, cancellation, reopening and subsequent mutation | b-tree example tests; b-table examples own ordered auxiliary-index ingestion and replacement |
| Transaction overlay suppression and prefetched-workspace cleanup | Collection storage tests |

`common::fixture::reads` checks blocks, every point and sparse entries; `copied`
checks independent copy/sync/reopen; `consumers` delegates to both. Large dense dtype
fixtures select their own point probes without changing other callers. Numerical
and storage matrices retain all types; repeated persistence is restricted to the
representative cases above. Future integration coverage does not replace these
contracts until equivalent assertions are running.

Sparse-output conversion uses named explicit, linear, rectangular, transformed,
high-rank, empty and zero-only cases. All request forms retain ordering and final
zero filtering; explicit coordinates retain ownership and compact requests allocate
only emitted coordinates. Invalid cardinality/request tests remain separate.

Benchmark runner tests own paired summaries, zero baselines, thread grouping and
admission filtering. The existing production/profiling entrypoints share
`common/adaptive_cases.rs`; `common::benchmark::storage` distinguishes encoded
bytes from retained node memory. Construction and final synchronization remain
separate measurements. Results stay outside tracked documentation.

Generic ordered-union duplicates, polling precedence, cooperative progress, and input
release belong to collate stream tests. Fensor retains compact-request adaptation,
batch error boundaries, high-rank scratch, and expression integration coverage.

## Call-site cleanup assertion ownership

- `access_matrix::section_g_sparse_iteration::in_order_iteration_matches_base_order`
  retains the original full-range assertions and the former
  `in_order_iteration_with_partial_range` assertions in one directory fixture.
  The partial read precedes the third insertion; both original inputs are retained.
- `incompatible_order_returns_structured_error` owns structured order errors and
  named invalid-range cases formerly repeated by `sparse_consumers::parity`.
  Numerical parity retains every full, selected, and empty-range comparison,
  including transformed matrix, diagonal, and Fourier outputs.
- `validate::tests::range_cardinality_handles_empty_selections_and_overflow`
  owns shared range-validation boundaries; `slice` unit tests retain independent
  descriptor validation. Iterator construction and sparse conversion share the
  validator without cloning explicit selections for validation alone.
- Cancellation phases and expression-depth cases remain separate. Local pinned
  futures leave their owning scope before release assertions; already boxed
  construction futures are cancelled directly without an additional box.
