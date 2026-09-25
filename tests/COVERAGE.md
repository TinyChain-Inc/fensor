# Regression ownership

Keep broad numerical cases separate from expensive full-consumer parity. Shared
filesystem scaffolding lives in `common/mod.rs`, imported directly by unit tests.
Special codec/error/independent-adapter fixtures remain local to their contracts.

| Contract | Primary coverage |
|---|---|
| Logical u64 geometry, inline rank spilling, overflow, large-coordinate persistence | `geometry`; request/mapping/storage-run unit bounds |
| Matrix transpose/diagonal dtype and consumer parity, output transforms, support, huge selected reads, corruption, cache pressure, cancellation and write constraints | `matrix_unary`; matrix unit request/support counters; diagonal doctests |
| Matrix shapes, dtypes, operand layouts and backend agreement | `matmul` numerical parity matrix, block streams |
| Matrix point/sparse/copy/sync/reload parity | Every dtype/layout pair on `(2,3,2)` and `(33,129,35)` |
| Reduction numerical lengths, operations, axes and keepdims | `reduce` parameterized block/terminal parity |
| Reduction full consumer/persistence parity | Lengths 7/4097; axis 1 of `[2,3,4]` with both keepdims values; each dtype/layout; dedicated sparse support and slice fixtures |
| Matrix transforms, batches, nesting, exceptional values and exact bounds | Dedicated `matmul` cases, retained independently |
| Sparse slice dtype semantics and persistence | `slice_reductions` small f32/f64/u8 fixtures |
| Multi-node sparse index topology across axes/capacities/shapes | `slice_reductions::indexed_topology` (f32) |
| Index continuation within/between coordinates at the index-page limit | Tensor unit `slice_index_pages_resume_within_and_between_coordinates` |
| Sparse output order, duplicates, final zeros, NaN and lengths | Expression unit `sparse_output_*` |
| Request/affine/irregular planning bounds and block grouping | Request, storage-read, matrix and tensor unit cases |
| Support masks, intermediate zeros and bounded group packing | Expression and slice unit cases; reduction integration cases |
| Corruption, boolean decision boundaries, cache pressure, cancellation and live reads | Dedicated reduction, matrix, storage and copy cases |
| No implicit result storage and transformed leaf parity | `storage_runs` |
| Transform input rejection and structured sparse-order errors | Public access integration cases; view unit tests retain mapping/write-through contracts |
| Sparse axis bounds on creation and reload | `foundations::public_create_rejects_invalid_sparse_axis_hint`, `storage_codec::reload_rejects_out_of_bounds_sparse_axis` |
| Request-provider precedence, errors and single invocation | Expression unit `request_providers_delegate_once_in_order_and_propagate_errors` |
| Tiled propagation, reduction boundaries and transformed current-shape coverage | Matrix `coordinate_traversal_propagates_and_preserves_coverage` |
| Full-shape linear batches and invalid shapes | Request unit `linear_batches_preserve_boundaries_and_validate_shape`; reduction tests retain boolean error boundaries |
| Copy lookahead, sequential writes, drop/error cancellation | `copy_pipeline` with a held filesystem block guard |
| Ordered batch completion and concurrency bound | Expression unit `buffered_batches_are_bounded_ordered_and_cancelled_by_drop` |
| Completion-order progress, slot replenishment, one-slot window, pairing and cancellation | Expression unit `completion_order_replenishes_slots_and_preserves_pairs` |
| First observed errors and pending evaluation cancellation | Expression unit `unordered_errors_cancel_pending_evaluation` |
| Numeric terminal completion schedules, wrapping and aggregate accuracy | Expression unit `numeric_terminals_accept_completion_schedules` |
| Parallel fixture directory uniqueness | `foundations::temporary_directory_names_are_unique_across_threads` |
| Public type restrictions and no adapter Clone requirement | Doctests |
| Composition and coordinate-stream consumption example | Crate-level compile-checked doctest in `src/lib.rs` |

Do not replace topology/pagination tests with dtype permutations, or structural
assertions with timing thresholds. Ignored benchmarks share fixtures but are not
ordinary regression tests. [BENCHMARKS.md](../BENCHMARKS.md#measurement-and-reproduction)
describes the shared workloads, two entrypoints, smoke instructions and record schema. Smoke runs are not performance evidence.

Routine checks use `cargo test`; benchmark harnesses compile only with the
`benchmarks` feature. See [CONTRIBUTING.md](../CONTRIBUTING.md) for targeted
checks and [BENCHMARKS.md](../BENCHMARKS.md) for opt-in smoke/profile runs.
