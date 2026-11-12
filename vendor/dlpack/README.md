# DLPack header

This directory vendors the single upstream header needed to regenerate dlpark's FFI bindings.

- Upstream: https://github.com/dmlc/dlpack
- Version: `v1.4`
- Commit: `6eca9e643fcf7e870b7a1c99ee7fb166c5c2c459`
- Verification: GitHub reports a valid commit signature

To refresh these files from the pinned release, run `mise run vendor:dlpack`.
To upgrade DLPack, update the version and full commit SHA in `xtask/src/cmd/vendor_dlpack.rs`, then run the same task and review both the vendored and generated binding diffs.
