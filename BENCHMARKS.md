# Tensor execution benchmarks

Benchmark harnesses, shared fixtures, instrumentation, and comparison runners
belong in Git. Generated timings, profiles, logs, and machine-specific provenance
do not. Keep all outputs under the ignored `benchmarks/results/` directory,
separate from the Python runners. `git add benchmarks` includes the runners
without staging local results.

The slice runner defaults to `benchmarks/results/slices/` for its CSV and logs;
the concurrency runner defaults to `benchmarks/results/concurrency.csv`. Defaults
are relative to the script location, independent of the working directory.
`--output` overrides them; relative overrides are resolved from the working
directory. Choose a separate output file for completion comparisons as shown below.

Execution contracts live in [DESIGN.md](DESIGN.md). Deterministic correctness and
structural assertions are indexed in [tests/COVERAGE.md](tests/COVERAGE.md).

## Measurement and reproduction

The harnesses use filesystem-backed f32 tensors. Keep source construction,
initial writes/sync, and fixture assertions outside consumption timing. Measure
copy-plus-sync separately, including destination creation, updates, and final sync.

| Entrypoint | Measurements and fixtures |
|---|---|
| `matrix_benchmark` | First/warm coordinate streams and separate copying; square, wide, narrow, batched, transformed, gathered, and nested products |
| `read_benchmark` | Row-major and coordinate streams; affine storage mapping without a destination |
| `slice_benchmark` | Terminal sums and both streams; whole/axis, strided, unary, nested, short/long groups |
| Unit `read_profile`, `slice_profile` | Corresponding cases with test-only structural counters and phase observations |
| Unit `copy_phase_benchmark` | Initialization, consumption, updates, sync; destination capacities 1/7/31/128/4096 |
| `concurrency_benchmark` | Coordinate streams and copy-plus-sync; square/wide/narrow products, warm/constrained caches, differing destination capacities |
| Unit `copy_pipeline_profile` | Copy phases and read/write overlap over a transposed source |
| `completion_benchmark` | Ordered stream controls, coordinate streams, numeric terminals, and separate copying over geometric/unary/reduction/matrix sources |

Fixtures cover dense/sparse layouts, block sizes, cache pressure, and geometric
transforms. The case definitions in `tests/common/` and each harness are the
source of truth for inputs. Inspect them before comparing different revisions.

Build the same harness against frozen before/after sources with identical
lockfiles and dependency sources. Use separate target directories, or clean the
fensor package between builds and retain separately named executables. Never
substitute a different revision for an unavailable baseline.

Run these commands in each respective checkout, changing the target path:

```sh
export CC=/usr/bin/cc CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc
export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1
cargo test --release --test matrix_benchmark --test read_benchmark --test slice_benchmark \
  --test concurrency_benchmark --test completion_benchmark \
  --target-dir /tmp/fensor-after --no-run
```

Finish compilation before timing. Run comparisons sequentially in alternating
before/after order, without concurrent builds or tests. Use isolated data
directories and record backend thread counts. The runners exercise one/four
backend threads; `RAYON_NUM_THREADS` does not change fensor's outer CPU-limited
async window.

```sh
# BEFORE_TARGET and AFTER_TARGET contain the separate release builds.
python3 benchmarks/run_slice_comparison.py BEFORE_TARGET AFTER_TARGET \
  --root /path/to/benchmark-data --output benchmarks/results/slices --pairs 2

# These arguments are executable paths, not target directories.
# concurrency_benchmark emits 84 records per full run.
python3 benchmarks/run_concurrency_comparison.py BEFORE_EXECUTABLE AFTER_EXECUTABLE \
  --output benchmarks/results/concurrency.csv --pairs 2
# completion_benchmark emits 208 records per full run.
python3 benchmarks/run_concurrency_comparison.py BEFORE_EXECUTABLE AFTER_EXECUTABLE \
  --output benchmarks/results/completion.csv --pairs 2 --expected-records 208
```

Both runners accept an optional `--root` naming an existing parent directory for
temporary benchmark data; the default is the system temporary directory. They
make no assumptions about the storage behind that directory. Temporary data
directories are removed after each run. The slice runner's `--start-pair` appends
additional pairs; the concurrency runner's `--focus-wide` selects its focused
wide/dense fixture.

Keep run provenance alongside the ignored output: source revision and dirty-tree
hashes, dependency/lockfile identities, executable hashes, exact commands, compiler,
device/runtime, data directory, cache settings, and thread counts. Temporary paths
alone do not make a baseline reproducible. Share outputs separately when needed.

## Profiles and smoke checks

Adapter traffic counters live in `tests/common/counters.rs`, absent from production
builds. Reset/snapshot requires an isolated benchmark process and one test thread;
the ordinary shared adapter is instrumented too. These counters observe codec
calls, not storage synchronization. Concurrent correctness tests use local or
task-local observations.

Run profiles independently of uninstrumented timings, for example:

```sh
mkdir -p benchmarks/results
RAYON_NUM_THREADS=4 cargo test --release --lib read_profile -- --ignored --nocapture --test-threads=1 > benchmarks/results/read-profile.log
RAYON_NUM_THREADS=4 cargo test --release --lib slice_profile -- --ignored --nocapture --test-threads=1 > benchmarks/results/slice-profile.log
RAYON_NUM_THREADS=4 cargo test --release --lib copy_phase_benchmark -- --ignored --nocapture --test-threads=1 > benchmarks/results/copy-phase.log
RAYON_NUM_THREADS=4 cargo test --release --lib copy_pipeline_profile -- --ignored --nocapture --test-threads=1 > benchmarks/results/copy-pipeline.log
```

`FENSOR_BENCH_SMOKE=1` selects small fixtures in matrix/read/slice/concurrency/
completion benchmarks and read/slice profiles. Copy profiles use fixed fixtures
even with that variable set. Smoke runs validate wiring and schemas, not performance:

```sh
FENSOR_BENCH_SMOKE=1 cargo test --test matrix_benchmark --test read_benchmark \
  --test slice_benchmark --test concurrency_benchmark --test completion_benchmark \
  -- --ignored --nocapture --test-threads=1
```

Unset the smoke variable before paired measurements; the comparison runners reject
it. Do not mix smoke records with measurements.

## Records and interpretation

READ durations are microseconds; SLICE and PIPELINE durations are nanoseconds.
The slice runner normalizes READ/SLICE to nanoseconds. Copy profiles emit
`COPY_PHASE` records; the pipeline profile also emits `COPY_OVERLAP` nanoseconds.
CSV headers and harness format strings define the remaining columns. Numeric
terminals emit one result, while stream records count consumed tensor elements.

Separate wall time from structural work: requests, coordinate resolutions, index
entries examined, physical-block borrows, backend calls, and adapter loads/saves.
Inclusive nested phase timings must not be summed. Copy consumption and update
durations overlap; overlap is their summed duration minus the enclosing join
duration, clamped to zero. It measures intersecting awaited intervals, not
simultaneous CPU execution.

First/warm consumption does not guarantee cold/warm physical-device caches.
Cached execution timings and adapter codec counts do not establish disk throughput.
Ordered controls help identify general run-to-run noise. Repeat regressions and
inspect structural counts before attributing a change to scheduling.

Completion-order consumption can remove delivery stalls yet worsen cache locality
or copy traffic. Tiny blocks and sparse-key boundaries can prevent useful run
formation. Sparse matrix contractions still scan logical positions; nested
expressions and separate tiles may repeat work. Bounded memory does not imply
sparsity-proportional execution or a universal speedup.

Timing comparisons are observations, not CI thresholds. Keep deterministic
progress, boundedness, corruption, cancellation, and numerical assertions in tests;
do not replace them with elapsed-time gates or publish generated result tables in
project documentation.
