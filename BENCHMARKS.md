# Tensor execution benchmarks

Track harnesses and runners, not generated results. All default output paths are
under ignored `benchmarks/results/`, relative to the runner's location.
`git add benchmarks` includes code without staging local measurements. Retain raw
CSVs, logs, commands, source/dependency identities, and the patches or snapshots
needed to reproduce dirty inputs. Each campaign writes a new output file.

Cargo targets, incremental caches, and compiled executables are temporary build
artifacts. Keep targets outside `benchmarks/results/`, reuse them across campaigns,
and delete frozen executables when comparisons finish, retaining their hashes and
build inputs. The runner already removes each run's temporary tensor directory.
Do not copy build caches into an evidence archive or retain them to preserve timing
results; offline summaries read only the CSV.

## Measurement and reproduction

The opt-in `benchmarks` feature compiles two ignored tests which use the same workloads in `tests/common/workloads.rs`:

- Integration `benchmark` measures production execution without internal metrics.
- Unit `profiling::profile` additionally observes task-local read/copy counters.

`FENSOR_BENCH_SUITE` selects `matrix`, `reduction`, `completion`, `pipeline`,
`copy`, `sparse`, `adaptive`, `mutation`, `traversal`, `reopen`, or `all` (default). Matrix cases include storage reads, affine/gather transforms,
regular/irregular selections, batches, nested products and extreme shapes.
Reduction cases include short/long groups, axes, strided and nested sources.
Completion cases compare streams and numeric terminals. Pipeline/copy cases vary
destination capacity, layout and cache pressure. Workload generators remain the
source of truth; capacities include 1, 7, 31, 128 and 4096.

Source construction, writes/sync and fixture assertions precede measurement.
Matrix, completion, and pipeline sources use the existing large fixture cache for construction, then
sync, drop, and reopen under the originally requested measured read budget.
This keeps construction's split-page requirements separate from consumption.
Row-major streams, coordinate streams, numeric terminals and copy-plus-sync are
separate operations. Copy timing includes destination tensor initialization and
final synchronization. Data-directory creation is outside timing.

The `copy`, `completion`, and `pipeline` pressure fixtures retain their original small cache budgets. A native
split can need several pinned pages whose combined admitted bounds exceed those
budgets. Their `copy-attempt` records report `admitted` and `capacity_rejected`
for every case; only native cache-capacity errors are accepted outcomes there.
Any other error fails the workload; every large-cache control must succeed. Rejections
report zero completed elements, not copy throughput. Compare attempt wall time
as successful-copy latency only for cases admitted in both runs; inspect
admission outcomes separately. Other copy workloads require success.

Freeze before/after sources, lockfiles, and dependency identities. Compile identical
workloads, copying each executable aside before another build can replace it. A
reused target is sufficient for sequential builds; use separate targets only when
build isolation requires them, and release those caches afterward. If the harness
changes, apply only its test-only changes to the frozen baseline; retain its
production source hash. Never substitute another revision for a missing baseline.
Finish builds before timing, and run comparisons without concurrent tests.
When switching source roots in a reused target, run
`cargo clean --release -p ha-ndarray` before rebuilding: its un-hashed rlib/cdylib
filenames otherwise permit stale artifacts from the other root. This removes generated output, not evidence.

The `sparse` suite isolates sparse-axis allocation and final-value clearing across
the standard capacities, leading/trailing axes, and two cache budgets. It records
payload-file bytes, all native table bytes, page occupancy, and retained decoded
node memory alongside separate stream, copy-plus-sync, and clear timings. Encoded file sizes and retained memory are recorded separately. These cached
filesystem measurements do not establish physical-device throughput.

```sh
export CC=/usr/bin/cc CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc
export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1
cargo test --features benchmarks --release --test benchmark --no-run --target-dir /tmp/fensor-after
cargo test --features benchmarks --release --lib --no-run --target-dir /tmp/fensor-after

# Arguments are paths to compiled test executables, not target directories.
python3 benchmarks/compare.py BEFORE_EXECUTABLE AFTER_EXECUTABLE \
  --before-revision BEFORE_ID --after-revision AFTER_ID --suite matrix --pairs 2
# Use the library test executables for profiles:
python3 benchmarks/compare.py BEFORE_LIB AFTER_LIB --entry profiling::profile \
  --before-revision BEFORE_ID --after-revision AFTER_ID --suite reduction \
  --output benchmarks/results/reduction-profile.csv
```

The runner alternates before/after order, defaults to backend thread counts one
and four, and validates identical record keys across runs. Optional `--root`
selects an existing parent for temporary tensor directories, removed after each
run; it assumes nothing about the underlying filesystem. `--output` overrides the
CSV path and places raw logs beside it. `RAYON_NUM_THREADS` configures backend
parallelism, not fensor's outer async concurrency window.

Keep source/dirty-tree and dependency hashes, lockfiles, commands, compiler/runtime,
cache settings and environment details alongside ignored outputs. The CSV adds
revision labels, executable SHA-256, before/after identity, thread count and pair.
Temporary paths alone do not establish reproducible provenance.

Completed comparisons print a descriptive summary. Summarize existing evidence
without rebuilding or running workloads:

```sh
python3 benchmarks/compare.py --summarize benchmarks/results/comparison.csv
```

Summaries validate paired runs, measurement identities, and executable provenance.
Each thread count and measurement stays separate, with sample counts, medians,
absolute changes, and percentage changes; zero baselines have no percentage.
For operations carrying admission outcomes, timing comparisons include only pairs
admitted on both sides and report excluded-pair counts. Admission counts remain
visible. Rejected attempts are not successful-copy timings. Raw CSV records and
logs remain authoritative; summaries neither establish statistical significance
nor impose timing-based correctness failures.

## Records and checks

Harness records use `FENSOR,workload,operation,temperature,metric,unit,value`.
Durations use `ns`; element counts, adapter traffic and structural observations
use `count`. Profiles report existing request/run/index/borrow/backend counters,
structural copy counters. `sync` is reported
separately within copy-plus-sync. All workloads emit this record shape; the runner performs no schema conversion.

```sh
FENSOR_BENCH_SMOKE=1 cargo test --features benchmarks --test benchmark -- --ignored --nocapture --test-threads=1
FENSOR_BENCH_SMOKE=1 cargo test --features benchmarks --lib profiling::profile -- --ignored --nocapture --test-threads=1
python3 -B -m unittest discover -s benchmarks -p 'test_*.py'
```

Smoke fixtures check wiring, values and schemas, not performance. The runner
rejects the smoke setting. Adapter traffic totals require an isolated process;
concurrent correctness tests use local/task-local observations instead.

## Interpretation

Compare structural work separately from wall time. Profiling collects task-local
counts of requests, reads, and publications. Its wall times include observation
and reporting overhead and must not be compared with uninstrumented runs.
Synchronous numerical work remains on the polling thread and uses ha-ndarray's
backend parallelism.

First/warm reads do not guarantee physical-device cache state. Cached timings and
codec traffic do not establish disk throughput. Repeat suspected regressions,
using ordered streams as controls. Tiny blocks and sparse keys can limit run
formation; sparse contractions scan logical positions, and nested expressions or
tiles may repeat work. Bounded memory is not a universal performance guarantee.

Keep correctness and structural gates in [tests/COVERAGE.md](tests/COVERAGE.md),
execution contracts in [DESIGN.md](DESIGN.md), and machine-specific measurements
out of project documentation. Timing thresholds do not belong in ordinary CI.

## Storage and transaction processes

The same harness selects `adaptive` for sparse vectors, narrow and trailing-axis
regions, low occupancy, and nearly full blocks; `mutation` for scalar/fill and
dense replacement; `traversal` for point/stream/reduction/transformed reads; and
`reopen` for strict loading. Storage accounting includes native dense-chunk row pages and dense payload files. It reads one synchronized page at a time outside timing.

```sh
FENSOR_BENCH_SUITE=adaptive cargo test --release --features benchmarks --test benchmark -- --ignored --nocapture --test-threads=1
```

Set `FENSOR_WORKLOAD_REPEATS` to lengthen steady-state measurements: defaults are
4 for traversal, 64 for dense replacement, and 1 for sparse point updates. Use
identical workloads, repetitions, geometry and cache budgets in paired builds.

The collection production benchmark uses the same comparison runner:

```sh
python3 benchmarks/compare.py BEFORE_COLLECTION AFTER_COLLECTION \
  --record-prefix COLLECTION --suite-env COLLECTION_BENCH_SUITE \
  --before-revision BEFORE_ID --after-revision AFTER_ID --suite overlay
```

Its opt-in integration target owns transaction mutation, assignment, decoding,
restoration, overlay-depth scans, streaming and finalization. Wall-clock claims
come from production-linked consumers; unit-test counters establish structural
work independently. Smoke runs check both native production and profiling wiring
with the normal thread stack. Generated measurements remain under ignored
`benchmarks/results/`.
