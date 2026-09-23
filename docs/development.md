# Development

Install the pinned tools and Conventional Commits hook after cloning:

```bash
mise install
cog install-hook commit-msg
```

CI checks tests, Clippy, Miri, generated bindings, documentation, and commit
messages. `release-plz` creates release pull requests, updates the changelog,
tags releases, and publishes the crate.

The convenience feature `cpu-all` enables every CPU-testable backend. `miri`
enables the same Rust integrations without PyO3, whose tests call the Python C
API.

## Regenerating FFI bindings

Bindings in `src/ffi.rs` are generated from the pinned DLPack header under
`vendor/dlpack/`:

```bash
mise run bindgen
# equivalent to: cargo xtask bindgen
```

CI verifies the checked-in output with `cargo xtask bindgen --check`.

To update the vendored header and its license from the pinned signed upstream
release, update the version and full commit SHA in `xtask`, then run:

```bash
mise run vendor:dlpack
```

Review the vendored header and generated Rust diff together. The task verifies
that the release tag resolves to the configured commit and that GitHub reports
a valid commit signature.

## Benchmarks

Criterion benchmarks live under `benches/`:

```bash
cargo bench --bench builder
```
