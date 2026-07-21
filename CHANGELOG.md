# Changelog

All notable changes to this project will be documented in this file.

---

## 0.8.0 - 2026-07-21

### 🌀 Miscellaneous

- Chore: update changelog
- Docs: configure docs.rs features ([#66](https://github.com/SunDoge/dlpark/pull/66))
- Docs: configure docs.rs features
- Prepare release ([#65](https://github.com/SunDoge/dlpark/pull/65))
- Docs: configure docs.rs features
- Prepare to release 0.8.0
- Chore: add more inline
- Fix chatgpt ([#64](https://github.com/SunDoge/dlpark/pull/64))
- Refactor: split metadata and builder code
- Chore: bump version
- Chore: add clippy action
- Fix: fix cargo doc for ffi.rs

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.8.0-alpha.3...v0.8.0>

---

## 0.8.0-alpha.3 - 2026-07-20

### 🌀 Miscellaneous

- Update docs ([#62](https://github.com/SunDoge/dlpark/pull/62))
- Doc: update readme
- Chore: fix docs
- Interop builder refactor ([#60](https://github.com/SunDoge/dlpark/pull/60))
- Fix: return builder instead of dlpack
- Refactor: derive metadata from builder context
- Fix: return Dlpack instead of Builder to avoid lifetime problem
- Doc: fix cargo doc
- Chore: check before release
- Feat!: add safe  `array_view_from_dlpack_mut` and `cpu_data_slice_mut` ([#59](https://github.com/SunDoge/dlpark/pull/59))
- Feat!: add safe  `array_view_from_dlpack_mut` and `cpu_data_slice_mut`
- Chore: merge copy_generic_metadata_unchecked
- Feat: support fallible generic metadata conversion
- Feat: add GenericArray/Slice metadata to avoid allocing Vec<i64> ([#57](https://github.com/SunDoge/dlpark/pull/57))
- Feat: add GenericArray/Slice metadata to avoid allocing Vec<i64>

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
* [@flying-sheep](https://github.com/flying-sheep)
**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.8.0-alpha.2...v0.8.0-alpha.3>

---

## 0.8.0-alpha.2 - 2026-07-19

### 🌀 Miscellaneous

- Add stream-aware Python and cudarc DLPack interop ([#56](https://github.com/SunDoge/dlpark/pull/56))
- Chore: disable default features for image to faster compilation
- Fix: enforce thread-safe contexts and accept empty tensor strides
- Fix: validate cudarc layouts and versioned Python extraction
- Chore: add is_known for DeviceType and DataTypeCode
- Update examples to use versioned DLPack tensors

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.8.0-alpha.1...v0.8.0-alpha.2>

---

## 0.8.0-alpha.1 - 2026-07-18

### 🧪 Dependencies

- Update pyo3 requirement from 0.28 to 0.29 ([#46](https://github.com/SunDoge/dlpark/pull/46))
- Update snafu requirement from 0.8 to 0.9 ([#45](https://github.com/SunDoge/dlpark/pull/45))

### 🌀 Miscellaneous

- Split test and Miri CI badges
- Refactor DLPack ownership, metadata, builders, interop, and CI
- Update FUNDING.yml
- Add Ko-fi funding link
- Add compact stride view helper
- Refine DLPack interop API
- Use versioned legacy api
- Update slice layout
- Update naming
- Add bench
- Use setup-rust-toolchain@v1 ([#52](https://github.com/SunDoge/dlpark/pull/52))
- Use setup-rust-toolchain@v1
- Designed by gemini ([#51](https://github.com/SunDoge/dlpark/pull/51))
- Bump version and adjust dev profile
- Add ndarray-candle example
- Add candle support
- Fix interop
- Update naming
- Add cudarc
- Make more api unsafe
- Save codex
- Update image
- Feat: impl OpaqueContext for Vec<T> with overhead doc; remove ImageContext wrapper
- Fix: add T: Sized bound to Box, Arc, NonNull OpaqueContext impls to prevent DST fat pointer UB
- Feat: add element_size() to DLDataType, num_elements() and num_bytes() to DLTensor and Dlpack
- Feat: make strides() return Result<Option<&[i64]>, Error> with ensure! guard
- Refact: use ensure! in DLTensor::shape()
- Feat: make DLTensor::shape() return Result with Snafu errors for null ptr and negative ndim
- Feat: add shape() and strides() to DLTensor and Dlpack
- Refact: remove unused OpaqueContext::as_ref and Target associated type
- Feat: implement image and python interop, resolve examples build
- Feat: add into_raw and as_ptr methods to Dlpack
- Refact: implement standard Builder pattern with DlpackBuilder, DlpackBox and build() method
- Refact: rename builder module to boxed to match Rust naming conventions
- Refact: rename smart pointer to DlpackBox and FFI representation to DlpackBoxInner
- Refact(miri): restore contiguous layout using DlpackBoxPtr and raw pointer initialization to pass Stacked Borrows
- Fix(miri): resolve stacked borrows UB by heap-allocating shape/strides and using full allocation provenance in tests
- Test: add unit tests for all layout constructors and RAII drop validation
- Update snafu
- Refact: consolidate Error definitions locally in each module, delete error.rs
- Refact: rename to Dlpack/DlpackFlags, add RAII Drop, and resolve clippy safety docs
- Flags -> DLPackFlags
- Add error
- Save
- Fix clippy lint ([#48](https://github.com/SunDoge/dlpark/pull/48))
- Fix clippy lint
- Update pyo3 requirement from 0.28 to 0.29
- Update snafu requirement from 0.8 to 0.9

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
* [@weiji14](https://github.com/weiji14)
* [@kylebarron](https://github.com/kylebarron)

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.7.0...v0.8.0-alpha.1>

---

## 0.7.0 - 2026-05-11

### 🚀 Features

- Create impl TensorLike and TryFrom for CudaView ([#29](https://github.com/SunDoge/dlpark/pull/29))

### 📝 Documentation

- Add CUDA GPU usage example to main README.md ([#39](https://github.com/SunDoge/dlpark/pull/39))

### 🧪 Dependencies

- Update cudarc requirement from 0.18.2 to 0.19.2 ([#38](https://github.com/SunDoge/dlpark/pull/38))
- Update pyo3 requirement from 0.27 to 0.28 ([#36](https://github.com/SunDoge/dlpark/pull/36))
- Update cudarc requirement from 0.17.7 to 0.18.2 ([#34](https://github.com/SunDoge/dlpark/pull/34))
- Update ndarray requirement from 0.16 to 0.17 ([#32](https://github.com/SunDoge/dlpark/pull/32))
- Update pyo3 requirement from 0.26 to 0.27 ([#30](https://github.com/SunDoge/dlpark/pull/30))
- Update pyo3 requirement from 0.25 to 0.26 ([#27](https://github.com/SunDoge/dlpark/pull/27))
- Update cudarc requirement from 0.16.4 to 0.17.0 ([#25](https://github.com/SunDoge/dlpark/pull/25))

### 🌀 Miscellaneous

- Changelog entry for v0.7.0 ([#44](https://github.com/SunDoge/dlpark/pull/44))
- Bump dlpark from 0.6.0 to 0.7.0
- Manually update changelog to group dep updates and add more entries
- Changelog entry for v0.7.0
- Setup trusted publishing to crates.io ([#43](https://github.com/SunDoge/dlpark/pull/43))
- Setup trusted publishing to crates.io
- Add CUDA GPU usage example to main README.md
- Update cudarc requirement from 0.18.2 to 0.19.2
- Update pyo3 requirement from 0.27 to 0.28
- Update cudarc requirement from 0.17.7 to 0.18.2
- Update ndarray requirement from 0.16 to 0.17
- Bump cudarc from 0.17.0 to 0.17.7, enable fallback-latest feature flag
- Merge branch 'main' into cudaview_impl
- Refactor FromPyObject for pyo3 0.27.0
- Update pyo3 requirement from 0.26 to 0.27
- Get cuda device ordinal properly from CudaContext
- Create impl TensorLike and TryFrom for CudaView
- Replace pyo3 0.26.0 deprecated functions ([#28](https://github.com/SunDoge/dlpark/pull/28))
- Replace pyo3 0.26.0 deprecated functions
- Update pyo3 requirement from 0.25 to 0.26
- Fix: remove cuda feature from cargo test
- Remove extra impl TryFrom<SafeManagedTensorVersioned> ([#26](https://github.com/SunDoge/dlpark/pull/26))
- Remove extra impl TryFrom
- Revert "Re-add Deref impl for SafeManagedTensorVersioned"
- Re-add Deref impl for SafeManagedTensorVersioned
- Update cudarc requirement from 0.16.4 to 0.17.0
- Create FUNDING.yml
- Add cuda

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
* [@weiji14](https://github.com/weiji14)

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.6.0...v0.7.0>

---

## 0.6.0 - 2025-06-04

### 🚀 Features

- Feat dlpack v1.1 ([#22](https://github.com/SunDoge/dlpark/pull/22))

### 🧪 Dependencies

- Update pyo3 requirement from 0.23 to 0.24 ([#19](https://github.com/SunDoge/dlpark/pull/19))

### 🌀 Miscellaneous

- Update publish.yml
- Update readme
- Update
- Add result
- Add Error for TensorLike
- Update std_container
- Update github action
- Use flags instead of u64
- Use uv in dlparkimg example
- Prepare for new version
- Add comments
- Add utils
- Bump version to v0.6
- Update ndarray
- Add docs
- Impl infer data type
- Update
- Add image convertor
- Add TensorView
- Add std container
- Add first convertor
- Add error handling
- Add versioned
- Save
- Update
- Add versioned
- Add workspace features
- Add more methods for tensor
- Update managed tensor
- Add device
- Add data type
- Update pyo3 requirement from 0.23 to 0.24

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.5.0...v0.6.0>

---

## 0.5.0 - 2025-04-30

### 🧪 Dependencies

- Upgrade pyo3 to 0.23 ([#18](https://github.com/SunDoge/dlpark/pull/18))
- Update ndarray requirement from 0.15.6 to 0.16.1 ([#16](https://github.com/SunDoge/dlpark/pull/16))

### 🌀 Miscellaneous

- Update Cargo.toml
- Upgrade pyo3 to 0.23
- Update ndarray requirement from 0.15.6 to 0.16.1

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
* [@kthui](https://github.com/kthui)

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.4.1...v0.5.0>

---

## 0.4.1 - 2024-03-26

### 🐛 Fixes

- Fix pyo3 ([#10](https://github.com/SunDoge/dlpark/pull/10))

### 🧪 Dependencies

- Update PyO3 to 0.21 ([#12](https://github.com/SunDoge/dlpark/pull/12))
- Update image requirement from 0.24.7 to 0.25.0 ([#11](https://github.com/SunDoge/dlpark/pull/11))
- Update pyo3 requirement from 0.19 to 0.20 ([#8](https://github.com/SunDoge/dlpark/pull/8))

### 🌀 Miscellaneous

- Remove unnecessary unbinding
- Update pyo3
- Update image requirement from 0.24.7 to 0.25.0
- Bump version
- Remove unused code
- Remove IntoPyPointer trait
- Update pyo3 requirement from 0.19 to 0.20
- Exclude github workflow

### 🧑‍🤝‍🧑 Contributors

* [@SunDoge](https://github.com/SunDoge)
* [@ChieloNewctle](https://github.com/ChieloNewctle)

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.4.0...v0.4.1>

---

## 0.4.0 - 2023-08-21

### 🚀 Features

- Optimize memory usage ([#7](https://github.com/SunDoge/dlpark/pull/7))
- Reduce memory usage ([#6](https://github.com/SunDoge/dlpark/pull/6))

### 🌀 Miscellaneous

- Cargo +nightly fmt
- Fix compile profile
- Fix clippy
- Override build opt-level
- Add dlparkimg example
- Update readme
- Update rust.yml
- Update api
- Optimize memory usage
- Reduce memory usage
- Add docs link

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.3.0...v0.4.0>

---

## 0.3.0 - 2023-07-18

### 🚀 Features

- Use Nonnull instead of *mut DLManagedTensor ([#5](https://github.com/SunDoge/dlpark/pull/5))

### 🌀 Miscellaneous

- Update docs
- Export more
- Update naming
- Export type
- Update readme
- Update default features
- Update
- Update strides
- Update prelude
- Update comments
- Clean up code
- Prepare to add pin
- Update ManagedTensor
- Add f16, bf16 supports
- Update Cargo.toml
- Clippy fix
- Save
- Introduce CowIntArray
- Sad
- Save
- Fix stack overflow
- Should reorganize the code structure
- Save
- Save
- Make it safe
- Rethinking
- Make clippy happy
- Remove redundant deleter
- Update license name
- Update version

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.2.2...v0.3.0>

---

## 0.2.2 - 2023-06-05

### 🧪 Dependencies

- Update pyo3 requirement from 0.18.3 to 0.19.0 ([#4](https://github.com/SunDoge/dlpark/pull/4))

### 🌀 Miscellaneous

- Update pyo3 requirement from 0.18.3 to 0.19.0
- Action -> actions

### 🧑‍🤝‍🧑 Contributors


**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.2.1...v0.2.2>

---

## 0.2.1 - 2023-05-23

### 🌀 Miscellaneous

- Add publish action
- Create dependabot.yml
- Merge branch 'main' of https://github.com/SunDoge/dlpark into main
- Update README.md
- Update readme and examples

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.2.0...v0.2.1>

---

## 0.2.0 - 2023-05-12

### 🌀 Miscellaneous

- Update traits
- Update readme ([#3](https://github.com/SunDoge/dlpark/pull/3))
- Add changelog
- Update readme and add badge
- Update readme
- Merge branch 'dev'
- Make clippy happy
- Dev ([#2](https://github.com/SunDoge/dlpark/pull/2))
- Cargo clippy
- From python is ok
- Add from python
- Clippy fix
- Use if let
- Add AsTensor trait
- Add drop
- Add managed
- Rename to ffi

**Full Changelog**: <https://github.com/SunDoge/dlpark/compare/v0.1.0...v0.2.0>

---

## 0.1.0 - 2023-05-09

### 🌀 Miscellaneous

- Update readme
- Prepare to release
- Remove link
- Add inferdtype
- Add more conversion
- Create rust.yml
- Make tensorwrapper a pyobject
- Test dict
- Fix memory problem ([#1](https://github.com/SunDoge/dlpark/pull/1))
- Update readme
- Save
- Remove more
- Add more dtypes
- Clippy fix
- Pytorch works
- Save
- Add more traits
- Add features
- Less box
- No safety garentee
- Pass
- Save
- Update
- Add more defaults
- Make it safe
- Save
- Save
- Update raw
- Save
- Add license
- Prepare to add docs
- Fix Cargo.toml
- Choose new name dlpark
- Add indicator
- Init


<!-- generated by git-cliff -->
