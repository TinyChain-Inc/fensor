# TinyChain Code Style

## Imports

Group `use` statements in this order, separated by one blank line:

1. Standard library (`std`, `core`, `alloc`).
2. External crates (alphabetical).
3. Workspace/internal modules (`crate::`, `super::`, repo siblings).

Within each group:

- Sort alphabetically (case-sensitive).
- Prefer explicit paths (`use foo::bar::Baz;`) over glob imports.
- Merge shared prefixes (use `foo::{Bar, Baz}`).

For conditional imports (`#[cfg]`), keep the group ordering and put the cfg on
the line above the statement.

## Linting

- Run `cargo clippy --all-targets --all-features -- -D warnings` locally or via
  `just lint` (whatever script your crate’s `CONTRIBUTING.md` references). Fix
  unused imports, let bindings, etc., instead of adding `allow` attributes.

## Formatting

- Run `cargo fmt` locally

## Crate-specific notes

Each crate’s `CONTRIBUTING.md` should point back to this doc and only mention
extra rules (e.g., feature-flag patterns, generated code) so cross-crate
consistency remains automatic.
