# Tensor execution benchmarks

Track harnesses and runners, not generated results. All default output paths are
under ignored `benchmarks/results/`, relative to the runner's location.
`git add benchmarks` includes code without staging local measurements. Historical
ignored datasets retain their original schemas; do not append new records to them.

## Measurement and reproduction

The opt-in `benchmarks` feature compiles two ignored tests which use the same workloads in `tests/common/workloads.rs`:

- Integration `benchmark` measures production execution without internal metrics.
- Unit `profiling::profile` additionally observes task-local read/copy counters.

`FENSOR_BENCH_SUITE` selects `matrix`, `reduction`, `completion`, `pipeline`, `copy`,
or `all` (default). Matrix cases include storage reads, affine/gather transforms,
regular/irregular selections, batches, nested products and extreme shapes.
Reduction cases include short/long groups, axes, strided and nested sources.
Completion cases compare streams and numeric terminals. Pipeline/copy cases vary
destination capacity, layout and cache pressure. Workload generators remain the
source of truth; capacities include 1, 7, 31, 128 and 4096.

Source construction, writes/sync and fixture assertions precede measurement.
Row-major streams, coordinate streams, numeric terminals and copy-plus-sync are
separate operations. Copy timing includes destination tensor initialization and
final synchronization. Data-directory creation is outside timing.

Freeze before/after sources and dependency identities. Compile identical workload
code against both versions with separate target directories and lockfiles. If the
harness changes, apply only its test-only changes to the frozen baseline; retain
its production source hash. Never substitute another revision for a missing
baseline. Finish builds before timing, and run comparisons without concurrent tests.

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

## Records and checks

Harness records use `FENSOR,workload,operation,temperature,metric,unit,value`.
Durations use `ns`; element counts, adapter traffic and structural observations
use `count`. Profiles report existing request/run/index/borrow/backend counters,
read phases and copy initialization/consumption/update/overlap. `sync` is reported
separately within copy-plus-sync. No old-schema compatibility conversion is provided.

```sh
FENSOR_BENCH_SMOKE=1 cargo test --features benchmarks --test benchmark -- --ignored --nocapture --test-threads=1
FENSOR_BENCH_SMOKE=1 cargo test --features benchmarks --lib profiling::profile -- --ignored --nocapture --test-threads=1
python3 -B -m unittest discover -s benchmarks -p 'test_*.py'
```

Smoke fixtures check wiring, values and schemas, not performance. The runner
rejects the smoke setting. Adapter traffic totals require an isolated process;
concurrent correctness tests use local/task-local observations instead.

## Interpretation

Compare structural work separately from wall time. Inclusive nested durations
must not be summed; copy consumption and updates overlap. Copy overlap measures
intersecting awaited intervals, not simultaneous CPU execution. Profile timings
include observation/reporting overhead and must not be compared with uninstrumented
wall times. Synchronous numerical work remains on the polling thread and uses
ha-ndarray's backend parallelism.

First/warm reads do not guarantee physical-device cache state. Cached timings and
codec traffic do not establish disk throughput. Repeat suspected regressions,
using ordered streams as controls. Tiny blocks and sparse keys can limit run
formation; sparse contractions scan logical positions, and nested expressions or
tiles may repeat work. Bounded memory is not a universal performance guarantee.

Keep correctness and structural gates in [tests/COVERAGE.md](tests/COVERAGE.md),
execution contracts in [DESIGN.md](DESIGN.md), and machine-specific measurements
out of project documentation. Timing thresholds do not belong in ordinary CI.
