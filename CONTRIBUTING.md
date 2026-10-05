# Contributing to `fensor`

Thanks for helping improve `fensor`.

## How to contribute

If you are not sure where to start, open an issue or discussion in the TinyChain
repository. If you already have a change, open a pull request with a focused
scope and a short rationale.

## Validation workflow

During edits, run the affected target or named regression, for example:

```sh
cargo test --lib request::tests
cargo test --test matmul
```

Reuse the same Cargo target directory and normal incremental debug builds. Separate
clean targets, disabled incremental compilation, and serialized builds are resource
workarounds or benchmark controls, not the routine validation workflow. Most of the
cost is compiling generic expressions and filesystem adapters; filtering a test
still compiles its test target.
Unary, trig, and cast tests retain separate integration targets so a local edit
rebuilds only its affected target. Use `cargo test --test math_unary`,
`--test math_trig`, or `--test math_cast`; no aggregate target is required.


Before review, run the correctness suite and lint gate once on the final changes:

```sh
cargo fmt --check
cargo test
cargo test --features complex
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
git diff --check
```

Feature configurations sharing one target directory must finish tests and doctests
before the next build. Overlapping builds can replace backend artifacts while
rustdoc still uses them.

`cargo test` includes unit tests, integration tests, and doctests. Benchmark and
profiling harnesses require the opt-in `benchmarks` feature and remain ignored
unless explicitly selected. Correctness tests do not require that feature.
The separate complex-enabled run covers native complex storage, projections,
and the full cast matrix; the default run checks that feature boundaries hold.

Changes to shared test/benchmark infrastructure also need the two small smoke
fixtures and Python runner tests in [BENCHMARKS.md](BENCHMARKS.md). Run paired release
measurements only for execution changes requiring performance evidence; do not
repeat the full campaign for documentation or fixture-only edits. When changing
feature wiring, check both default and all-feature targets.

Documentation-only changes need applicable doctests and link/diff checks. Follow
[`CODE_STYLE.md`](CODE_STYLE.md), and use [test ownership](tests/COVERAGE.md) to
select relevant targets without dropping independent regression coverage.


Keep documentation with its owner: README describes public use, [DESIGN.md](DESIGN.md)
owns execution invariants, and ROADMAP tracks unfinished work. Version benchmark
code and the [methodology](BENCHMARKS.md), but keep generated outputs in ignored
`benchmarks/results/`; use [test ownership](tests/COVERAGE.md)
to avoid duplicating regression permutations. Numerical fixtures use native
stream construction and retain their shared `common::Directory` guards. Keep the
guard alive through sources, views, and reopening; helpers return it with storage
and never discard it through tuple projection. Mutation fixtures continue to
exercise the mutation under test. Benchmark directory ownership stays outside
timed operations, with cleanup after storage accounting. Keep timing workloads behind the
`benchmarks` feature and use the shared record schema and comparison runner.

## Licensing

By contributing code to this project, you represent that you own the copyright
on your contributions, or that you have followed the licensing requirements of
the copyright holder, and that your code may be used without any further
restrictions than those specified in the Apache 2.0 open-source license. A copy
of the license can be found in the `LICENSE` file in the root directory of the
project.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct.
