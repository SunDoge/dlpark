# DLPack header

This directory vendors the single upstream header needed to regenerate dlpark's FFI bindings.

- Upstream: https://github.com/dmlc/dlpack
- Version: `v1.3`
- Commit: `84d107bf416c6bab9ae68ad285876600d230490d`
- Verification: GitHub reports a valid commit signature

To refresh these files from the pinned release, run `mise run vendor:dlpack`.
To upgrade DLPack, update the version and full commit SHA in `xtask/src/cmd/vendor_dlpack.rs`, then run the same task and review both the vendored and generated binding diffs.
