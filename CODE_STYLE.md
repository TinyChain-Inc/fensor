# TinyChain Code Style

These rules apply across every crate in this workspace. Keep local crate notes
short and link back here so we don’t fork the style guide.

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

## Formatting & linting

- Run `cargo fmt` before sending patches. The workspace uses a shared
  `rustfmt.toml` so every crate stays consistent (see below).
- Run `cargo clippy --all-targets --all-features -D warnings` locally or via
  `just lint` (whatever script your crate’s `CONTRIBUTING.md` references). Fix
  unused imports, let bindings, etc., instead of adding `allow` attributes.

## Rustfmt configuration

`rustfmt.toml` at the repo root configures the default behavior. Key settings:

- `group_imports = "StdExternalCrate"` – enforces the import grouping above.
- `imports_granularity = "Module"` – merges nested imports automatically.
- `wrap_comments = true`, `normalize_doc_attributes = true` – keep docs tidy.
-
We stick to stable rustfmt options where possible; if a feature is nightly-only,
gate it behind `unstable_features = true` and note it in `README.md`.

## Crate-specific notes

Each crate’s `CONTRIBUTING.md` should point back to this doc and only mention
extra rules (e.g., feature-flag patterns, generated code) so cross-crate
consistency remains automatic.
