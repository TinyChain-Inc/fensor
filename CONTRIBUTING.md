# Contributing to `fensor`

Thanks for helping improve `fensor`.

## How to contribute

If you are not sure where to start, open an issue or discussion in the TinyChain
repository. If you already have a change, open a pull request with a focused
scope and a short rationale.

Before submitting:

1. Run `cargo fmt`.
2. Run `cargo clippy --all-targets --all-features -- -D warnings`.
3. From this crate's root, run `cargo test --all-targets --all-features` and
   `cargo test --doc --all-features`. Documentation-only changes need doctests,
   formatting, and link/diff checks; numerical benchmarks are not required.
4. Follow style guidance in [`CODE_STYLE.md`](./CODE_STYLE.md).

Keep documentation with its owner: README describes public use, [DESIGN.md](DESIGN.md)
owns execution invariants, and ROADMAP tracks unfinished work. Version benchmark
code and the [methodology](BENCHMARKS.md), but keep generated outputs in ignored
`benchmarks/results/`; use [test ownership](tests/COVERAGE.md)
to avoid duplicating regression permutations.

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
