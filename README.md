# dlpark

[![Tests](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge&label=test)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
[![Clippy](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/clippy.yml?branch=main&style=for-the-badge&label=clippy)](https://github.com/SunDoge/dlpark/actions/workflows/clippy.yml)
[![Miri](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/miri.yml?branch=main&style=for-the-badge&label=miri)](https://github.com/SunDoge/dlpark/actions/workflows/miri.yml)
[![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)](https://crates.io/crates/dlpark)
[![docs.rs](https://img.shields.io/docsrs/dlpark/latest?style=for-the-badge)](https://docs.rs/dlpark)

A pure Rust implementation of [dmlc/dlpack](https://github.com/dmlc/dlpack).

This crate focuses on transferring tensors between Rust and Python, and between Rust tensor/array libraries, without copying. It targets DLPack 1.3 and Rust edition 2024.

## Installation

`dlpark` ships **no default features** — enable the interop backends you need:

```bash
cargo add dlpark --features "ndarray half"          # Rust-only
cargo add dlpark --features "pyo3 image"             # Python extension
cargo add dlpark --features "cudarc"                 # CUDA (needs a CUDA toolchain)
```

Feature groups for testing:

- `cpu-all` — every CPU-testable backend (`candle`, `half`, `image`, `ndarray`, `pyo3`) in one go. Used by the `cargo test` and `cargo clippy` CI jobs.
- `miri` — `candle`, `half`, `image`, `ndarray` (no `pyo3`, whose tests call the Python C API). Used by the Miri job.

## Mental model

A **producer** wraps its data into an `allocation::Initialized` value, sets the scalar tensor fields, and finishes it into a `Managed<M>` — an RAII handle over a raw DLPack managed tensor pointer that calls the DLPack deleter on drop. `M` selects the ABI:

- `legacy::Dlpack` = `Managed<DLManagedTensor>` — the pre-v0.8 `"dltensor"` capsule.
- `versioned::Dlpack` = `Managed<DLManagedTensorVersioned>` — the current `"dltensor_versioned"` capsule, carrying version and flags.

The high-level producer path converts a boxed container directly:

```rust
use dlpark::{allocation::dynamic, ffi::DLManagedTensorVersioned, versioned};
use ndarray::arr2;

let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(arr2(&[[1_i32, 2, 3], [4, 5, 6]])).try_into()?;
let tensor: versioned::Dlpack = unsafe { initialized.finish() };
```

A **consumer** receives a `Managed` and calls `validate()` to check its descriptor metadata. The resulting `TensorRef` exposes shape, strides, dtype, device, and size through safe accessors. Dereferencing the data pointer remains unsafe because DLPack does not report allocation bounds. Backend conversions go through the `TryFromDlpack` trait.

## What is DLPack?

`DLPack` is a common in-memory tensor structure that enables sharing tensor data between different deep learning frameworks. It provides a standardized way to exchange tensor data without copying, making it efficient for framework interoperability.

Key features of `DLPack`:

- Zero-copy tensor sharing between frameworks
- Support for various data types and devices (CPU, GPU, etc.)
- Memory management through deleter functions
- Versioned ABI for compatibility

## Versioning

The library implements both legacy and versioned `DLPack` structures:

- `legacy::Dlpack`: legacy managed tensor capsule support.
- `versioned::Dlpack`: versioned managed tensor capsule support with:
  - Major version `1`, minor version `3` (the version provided by the bundled headers).
  - Additional flags for tensor properties (read-only, copied, sub-byte type padding).

`DLPackVersion` exposes semantic comparisons: `is_compatible_with` checks ABI (major-version) compatibility, and `supports` checks whether a version includes a requested feature level. An incompatible major version is rejected on ingestion and the tensor is released.

## Safe abstractions

The library provides a Rust ownership wrapper over the C-style `DLPack` structures, `Managed<M>`:

- RAII wrapper around a raw managed `DLPack` tensor pointer.
- Automatic cleanup through the DLPack deleter function on drop. If the managed tensor carries a NULL deleter (per the DLPack spec: the producer retains ownership and the consumer must not free it), `Drop` is a no-op and the allocation plus `manager_ctx` are *not* released — the caller that built such a tensor reclaims them through their original owner.
- A single ownership model for locally produced and externally received tensors, with `legacy::Dlpack` and `versioned::Dlpack` aliases selecting the ABI.
- `TensorRef` as the validated, borrowed descriptor view; `TensorMut` as the exclusively borrowed, writable-descriptor view.

The `ManagedTensorBase` trait abstracts the common operations of the legacy and versioned ABIs so allocation and ownership APIs can operate generically over `M` while preserving the concrete C layout selected by the caller.

Other key features:

- Memory safety through Rust's ownership system.
- Support for [image] buffers, [ndarray], and [candle] tensors, plus raw DLPack tensor layouts.
- Python interoperability through PyO3.
- Optional DLPack 1.3 C Exchange API fast path when a producer type exposes `__dlpack_c_exchange_api__`.

## Producing a tensor

A producer pipeline has three stages: **metadata** → **initialization** → **finish**.

### 1. Metadata: shape and strides

`metadata::Fixed` and `metadata::Dynamic` compose shape and strides with the managed tensor allocation. Each axis value is stored as `i64`. The storage wrappers select how the values get there:

- `metadata::Copied<T>` — values are **copied** into the managed tensor allocation. Safe `prepare`.
- `metadata::Borrowed<T>` — values are **borrowed** from caller-owned `i64` storage. Zero-copy metadata, but requires `unsafe prepare_unchecked` because the arrays must outlive the managed tensor.

| Rank | Copied (safe) | Borrowed (unsafe) |
| ---- | ------------- | ----------------- |
| Fixed, known at compile time | `Fixed::new(Copied(shape), Copied(strides))` → `prepare` | `Fixed::new(Borrowed(&shape), Borrowed(&strides))` → `prepare_unchecked` |
| Dynamic, runtime | `Dynamic::new(Copied(shape_vec), Copied(strides_vec))` → `prepare` | `Dynamic::new(Borrowed(shape_slice), Borrowed(strides_slice))` → `prepare_unchecked` |

`Copied` accepts any integer element type that implements `TryInto<i64>` (for example `u32`, `i16`, `usize`, `isize`), not just `i64`. When the source is already `i64`, a `TypeId` fast path uses `ptr::copy_nonoverlapping` directly; otherwise each value is converted in place. Either way no temporary `Vec<i64>` is allocated.

### 2. Initialization

`prepare::<M>()` returns a `PreparedFixed` / `PreparedDynamic`. Calling `.initialize(ctx)` installs the owning context and deleter and returns an `allocation::Initialized<M, _>`. The context owns the backing data and any allocation metadata; it must implement `OpaqueContext`, which is provided for `Box<T: Send>` and `Arc<T: Send + Sync>` (the deleter may fire on a different thread).

The boxed container itself is the canonical context: interop producers pass `Box<Tensor>` / `Box<CudaSlice<T>>` / `Box<ImageBuffer<...>>` directly, so the data owner stays alive for the DLPack tensor's lifetime.

### 3. Scalar fields and finish

`Initialized` exposes chainable setters before the final `finish`:

```rust
use dlpark::{
    allocation::fixed,
    ffi::{DLDevice, DLDataType, DLManagedTensorVersioned},
    metadata::{Copied, Fixed},
    versioned,
};

let shape = [2_u32, 3];
let strides = [3_i16, 1];
let mut data = vec![0_f32; 6];
let data_ptr = data.as_mut_ptr().cast();

let prepared = Fixed::new(Copied(shape), Copied(strides))
    .prepare::<DLManagedTensorVersioned>()?;
let mut initialized = prepared.initialize(Box::new(data));
initialized
    .set_data(data_ptr)
    .set_dtype(DLDataType::F32)
    .set_device(DLDevice::CPU);
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };
```

`finish` is unsafe because the caller asserts the descriptor now satisfies the DLPack contract: data and metadata pointers remain valid until drop, and flags accurately describe aliasing and mutability. `set_flags` rejects newly asserting `IS_COPIED`; use `set_flags_unchecked` after producing a copy. Versioned tensors also expose `set_version` (rejects an incompatible major version).

### Direct conversion

For the common case where a boxed container already determines shape, strides, dtype, and device, skip the manual metadata stage and convert directly:

```rust
use dlpark::{allocation::fixed, ffi::DLManagedTensorVersioned, versioned};

let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> =
    Box::new(rgb_image).try_into()?; // image: fixed rank 3, HWC
let tensor: versioned::Dlpack = unsafe { initialized.finish() };
```

The element type's DLPack descriptor is provided by the `DlpackElement` trait, implemented for the Rust integer and float primitives (and `half::f16` / `half::bf16` under the `half` feature).

### Performance of metadata copy

In the included length-64 microbenchmark, the `i64` fast path takes approximately 11.7 ns, while allocating temporary `Vec<i64>` storage and then calling `copy_nonoverlapping` takes approximately 54.9 ns on the development machine. Reproduce it with:

```bash
cargo bench --bench builder -- generic_metadata_copy
```

The [ndarray] exporter uses `Copied` for its `usize` shape and `isize` strides, so exporting an owned array does not allocate temporary `Vec<i64>` metadata.

## Reading tensor data

Once you hold a `Managed`, validate its descriptor into a `TensorRef` before reading metadata:

```rust
let tensor = dlpack.validate()?;
let shape = tensor.shape();                      // &[i64]
let strides = tensor.strides();                  // Option<&[i64]> (None = compact)
let n = tensor.num_elements();
let bytes = tensor.num_bytes();                   // sub-byte-packing aware
let data = unsafe { tensor.cpu_slice::<f32>()? }; // compact CPU data, dtype-checked
```

`TensorRef::cpu_slice` validates device, dtype, alignment, and compact layout, but remains unsafe because a descriptor cannot prove the data allocation's bounds. `cpu_bytes` is the dtype-agnostic variant and also supports packed sub-byte dtypes. Low-level consumers may use `TensorRef::offset_data_ptr` / `offset_bytes_ptr` to obtain a device-agnostic pointer with `byte_offset` applied.

**Mutable access.** Call `validate_mut()` to validate metadata and reject `READ_ONLY`, then use the unsafe mutable data accessor:

```rust
let mut tensor = dlpack.validate_mut()?;
let data = unsafe { tensor.cpu_slice_mut::<f32>()? };
```

Mutable data access is always unsafe because Rust cannot prove the bounds, aliasing, or concurrent use of an external allocation. `IS_COPIED` does not affect validation; it only reports that a producer made a copy for an exchange operation. Rust zero-copy adapters leave it unset.

`Managed::flags()` / `version()` read the versioned fields; `flags_mut` is `unsafe` because setting `IS_COPIED` or clearing `READ_ONLY` asserts the corresponding ownership/mutability guarantee.

## Python exchange paths

The `pyo3` feature supports the standard Python DLPack capsule protocol:

- `legacy::Dlpack` consumes or produces legacy `"dltensor"` capsules.
- `versioned::Dlpack` consumes or produces `"dltensor_versioned"` capsules.
- `python::dlpack_device(obj)` calls and validates `obj.__dlpack_device__()`, returning a Rust `DLDevice`.
- When extracting a versioned tensor from a Python object, dlpark first checks the object's type for a `__dlpack_c_exchange_api__` PyCapsule named `"dlpack_exchange_api"`. If present, it walks the `prev_api` chain for a compatible major version and uses the DLPack C Exchange API no-sync function table (`managed_tensor_from_py_object_no_sync`). Otherwise it calls `obj.__dlpack__(max_version=(1, 3))` and consumes the returned capsule. Producers that only implement the legacy no-argument protocol must be extracted as `legacy::Dlpack`, because they return the incompatible `"dltensor"` capsule ABI.
- Capsule consumption is single-use: extracting renames the capsule to `"..._used"`; a second extraction raises `PyValueError("DLPack capsule has already been consumed")`.
- Consumers can call `versioned::Dlpack::extract_with_options(obj, stream, copy)` to pass an optional stream and tri-state copy request to `__dlpack__`; `extract_with_stream(obj, stream, copy)` is the typed convenience path for GPU consumers. The `cudarc` feature implements `python::DlpackStream` for `cudarc::driver::CudaStream` (and `Arc<CudaStream>`); other backends can implement the unsafe `DlpackStream` trait.

The C Exchange API is intended for extension/library use where the consumer borrows tensors and coordinates work on the producer's current stream. It is not a replacement for the normal `__dlpack__` ingestion path.

## Interop backends

| Feature | Producer | Consumer | Data movement |
| --- | --- | --- | --- |
| `image` | boxed `ImageBuffer` | borrowed or owning `ImageBuffer` | zero-copy |
| `ndarray` | boxed owned array | `ArrayViewD` / `ArrayViewMutD` | zero-copy |
| `candle` | boxed CPU `Tensor` | owned CPU `Tensor` | export is zero-copy; import copies |
| `cudarc` | boxed `CudaSlice` | owning CUDA slice view | zero-copy |

Producer conversions require a `Box` because the container itself becomes the stable, type-erased DLPack `manager_ctx`; the library does not implicitly allocate that box. The `half` feature adds `DlpackElement` impls for the [half] crate's 16-bit floating-point types, independent of these adapters.

### image

Zero-copy both ways. Producing wraps an `ImageBuffer<P, Vec<P::Subpixel>>` as a rank-3 HWC tensor (`[height, width, channels]`). Consuming validates an HWC compact layout and exposes the data either as a borrowed `ImageBuffer<P, &[P::Subpixel]>` or as an owning `ImageBuffer<P, DlpackContainer<M, P::Subpixel>>` that keeps the managed tensor alive.

### ndarray

Zero-copy both ways for CPU arrays. Producing converts a boxed owned array into a `dynamic::Initialized` (runtime rank). Consuming yields `ArrayViewD<T>` (read) or `ArrayViewMutD<T>` (mut, which rejects `READ_ONLY` and requires the caller to prove exclusivity — DLPack flags alone cannot establish Rust aliasing).

### candle

Zero-copy from `candle::Tensor` to DLPack (the boxed tensor's `Arc`-refcounted storage stays alive as the context); the reverse direction always copies, since candle has no borrowed CPU tensor type. Compact-stride sources take a bulk-copy path; arbitrary strides are gathered. CPU only — candle's CUDA backend needs separate integration work. Candle storage sits behind an `RwLock`, so the exported pointer aliases memory candle itself can still mutate; set `DlpackFlags::READ_ONLY` on the returned `Initialized` before finishing if you need to signal read-only intent.

### cudarc

Zero-copy in both directions between a [cudarc] `CudaSlice<T>` and a DLPack tensor. The 1-D `TryFrom` producer returns a contiguous default layout (`shape = [len]`, `strides = [1]`) and leaves `IS_COPIED` unset; use `interop::cudarc::from_cuda_slice` for higher-rank tensors. The reverse direction consumes the managed tensor through `TryFrom<Managed<M>> for BorrowedCudaSlice<M, T>`, keeping the tensor alive for as long as the CUDA view exists — the view's destructor calls `CudaSlice::leak` before the managed tensor drops, so `cudaFree` is not called on a DLPack-owned allocation.

## Features

No features are enabled by default — enable the backends you need (see [Installation](#installation)).

| Feature | Description | Status |
| --- | --- | --- |
| `pyo3` | Python interop via [pyo3] (capsule protocol + DLPack C Exchange API fast path) | ✅ |
| `image` | Zero-copy conversion with [image] buffers | ✅ |
| `ndarray` | Zero-copy conversion with [ndarray] arrays/views | ✅ |
| `half` | `f16`/`bf16` element type support (via [half]) | ✅ |
| `candle` | Conversion with [candle] `Tensor` — CPU only; candle's CUDA backend needs separate integration work | ✅ |
| `cudarc` | Zero-copy conversion with [cudarc] `CudaSlice<T>` — no automated tests here, needs a CUDA-capable device to exercise | ✅ |

## Quick start

Two runnable examples:

- [`examples/dlparkimg`](./examples/dlparkimg/) — a Python extension module (via `pyo3`) transferring `image::RgbImage` to/from Python (e.g. `torch.Tensor`). Run with `uv run main.py`.
- [`examples/ndarray-candle`](./examples/ndarray-candle/) — a plain binary round-tripping data through DLPack: `ndarray::Array2` → `versioned::Dlpack` → `candle::Tensor` → `versioned::Dlpack` → `ndarray` view, run with `cargo run -p ndarray-candle`.

`examples/profile_builder.rs` profiles the `metadata::Fixed` / `metadata::Dynamic` allocation paths (`cargo run --release --example profile_builder`); `benches/builder.rs` benchmarks them (`cargo bench --bench builder`).

## Usage examples

### Converting between Rust and Python

```rust
use dlpark::{allocation::fixed, ffi::DLManagedTensorVersioned, versioned, TryFromDlpack};
use image::ImageBuffer;
use pyo3::prelude::*;

#[pyfunction]
fn read_image(filename: &str) -> PyResult<versioned::Dlpack> {
    let img = image::open(filename)?.to_rgb8();
    let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> =
        Box::new(img).try_into()?;
    Ok(unsafe { initialized.finish() })
}

#[pyfunction]
fn write_image(filename: &str, tensor: versioned::Dlpack) -> PyResult<()> {
    // SAFETY: this extension accepts tensors through the Python DLPack
    // protocol and relies on the producer to provide a valid descriptor.
    let img: ImageBuffer<image::Rgb<u8>, _> =
        unsafe { ImageBuffer::try_from_dlpack(&tensor) }?;
    img.save(filename)?;
    Ok(())
}
```

### Image processing

```rust
use dlpark::{allocation::fixed, ffi::DLManagedTensorVersioned, versioned, TryFromDlpack};
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> =
    Box::new(img).try_into()?;
let tensor: versioned::Dlpack = unsafe { initialized.finish() };
let img2 = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&tensor)? };
```

### ndarray

```rust
use dlpark::{allocation::dynamic, ffi::DLManagedTensorVersioned, versioned, TryFromDlpack};
use ndarray::{ArrayD, ArrayViewD, arr2};

let array = arr2(&[[1_i32, 2, 3], [4, 5, 6]]);
let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(array).try_into()?;
let tensor: versioned::Dlpack = unsafe { initialized.finish() };
let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&tensor)? };

assert_eq!(view[[1, 2]], 6);

let dynamic: ArrayD<i32> = arr2(&[[1, 2], [3, 4]]).into_dyn();
let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(dynamic).try_into()?;
let dynamic_tensor: versioned::Dlpack = unsafe { initialized.finish() };
```

### candle

Zero-copy from `candle::Tensor` to DLPack; the reverse direction (DLPack to `candle::Tensor`) always copies, since candle has no borrowed CPU tensor type.

```rust
use candle_core::Tensor;
use dlpark::{
    DlpackFlags, TryFromDlpack,
    allocation::dynamic, ffi::DLManagedTensorVersioned, versioned,
};

let tensor = Tensor::new(&[1_f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(tensor).try_into()?;
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };

let tensor2 = unsafe { Tensor::try_from_dlpack(&dlpack)? };
assert_eq!(tensor2.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);

// Signal read-only intent before finishing (candle storage is shared via RwLock):
let tensor = Tensor::new(&[1_f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let mut initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(tensor).try_into()?;
initialized.set_flags(DlpackFlags::READ_ONLY)?;
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };
```

### cudarc

Zero-copy in both directions between a [cudarc] `CudaSlice<T>` and a DLPack tensor.

```rust
use dlpark::{
    TryFromDlpack,
    allocation::{dynamic, fixed},
    ffi::DLManagedTensorVersioned,
    interop::cudarc::{BorrowedCudaSlice, from_cuda_slice},
    versioned,
};

// 1-D default layout (shape = [len], strides = [1]):
let initialized: fixed::Initialized<DLManagedTensorVersioned, 1> =
    Box::new(cuda_slice).try_into()?;
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };

// Higher-rank:
let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    from_cuda_slice::<_, DLManagedTensorVersioned>(Box::new(cuda_slice), &[2, 3], &[3, 1])?;
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };

// Reverse direction keeps the managed tensor alive for the CUDA view's lifetime:
let borrowed =
    unsafe { BorrowedCudaSlice::<DLManagedTensorVersioned, f32>::try_from_dlpack(dlpack)? };
```

## Regenerating FFI bindings

The C bindings in `src/ffi.rs` are generated from the `dlpack` C header (a git submodule at `dlpack/`) by the workspace maintenance tool:

```bash
mise run bindgen    # = cargo xtask bindgen
```

It is a manual regeneration step, not a build dependency of the crate.

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image
[ndarray]: https://github.com/rust-ndarray/ndarray
[half]: https://crates.io/crates/half
[candle]: https://github.com/huggingface/candle
[cudarc]: https://crates.io/crates/cudarc
