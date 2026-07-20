# dlpark

[![Tests](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge&label=test)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
[![Miri](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/miri.yml?branch=main&style=for-the-badge&label=miri)](https://github.com/SunDoge/dlpark/actions/workflows/miri.yml)
[![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)](https://crates.io/crates/dlpark)
[![docs.rs](https://img.shields.io/docsrs/dlpark/latest?style=for-the-badge)](https://docs.rs/dlpark)

A pure Rust implementation of [dmlc/dlpack](https://github.com/dmlc/dlpack).

This implementation focuses on transferring tensors between Rust and Python, and between Rust tensor/array libraries, without copying.

## What is `DLPack`?

`DLPack` is a common in-memory tensor structure that enables sharing tensor data between different deep learning frameworks. It provides a standardized way to exchange tensor data without copying, making it efficient for framework interoperability.

Key features of `DLPack`:

- Zero-copy tensor sharing between frameworks
- Support for various data types and devices (CPU, GPU, etc.)
- Memory management through deleter functions
- Versioned ABI for compatibility

## Implementation Details

### Versioning

The library implements both legacy and versioned `DLPack` structures:

- `legacy::Dlpack`: Legacy managed tensor capsule support
- `versioned::Dlpack`: Versioned managed tensor capsule support with:
  - Major version: 1
  - Minor version: 3
  - Additional flags for tensor properties (read-only, copied, sub-byte type padding)

### Safe Abstractions

The library provides a Rust ownership wrapper over the C-style `DLPack` structures, `ManagedBox<M>`:

- RAII wrapper around a raw managed `DLPack` tensor pointer
- Automatic cleanup through the DLPack deleter function on drop
- `legacy::Dlpack` and `versioned::Dlpack` are convenience aliases for `ManagedBox<DLManagedTensor>` and `ManagedBox<DLManagedTensorVersioned>` — the two concrete forms you'll actually use

Other key features:

- Memory safety through Rust's ownership system
- Support for `image` buffers, [ndarray], and [candle] tensors, plus raw DLPack tensor layouts
- Python interoperability through PyO3
- Optional DLPack 1.3 C Exchange API fast path when a producer type exposes `__dlpack_c_exchange_api__`

### Choosing Builder Metadata

`Builder<C, L>` uses the metadata type `L` to select its allocation strategy:

- **`CopiedArray<S, T, N>`** — fixed rank known at compile time. Shape and strides are copied into the managed tensor allocation. `build` and `build_raw` are infallible.
- **`CopiedSlice<S, T>`** — dynamic rank. The containers may be borrowed slices or owned values such as `Vec<i64>`. Use `try_build` or `try_build_raw`.
- **`GenericArray<S, T, A, B, N>`** — fixed-rank metadata with elements that implement `TryInto<i64>`. Values are converted directly into the managed tensor allocation; use `try_build` or `try_build_raw`.
- **`GenericSlice<S, T, A, B>`** — dynamic-rank metadata with elements that implement `TryInto<i64>`. It avoids allocating temporary `Vec<i64>` values.
- **`BorrowedArray<N>`** — fixed-rank, zero-copy metadata. Its `build` methods are unsafe because the arrays must outlive the managed tensor.
- **`BorrowedSlice`** — dynamic-rank, zero-copy metadata. Its `try_build` methods are unsafe for the same lifetime reason.

Copied metadata and the managed tensor share one allocation. Borrowed metadata only allocates the managed tensor header.

Use `GenericArray` or `GenericSlice` when a framework exposes dimensions or
strides as an integer type other than `i64`:

```rust
use dlpark::{Builder, legacy};
use dlpark::metadata::GenericArray;

let shape = [2u32, 3];
let strides = [3i16, 1];
let mut data = vec![0f32; 6];
let data_ptr = data.as_mut_ptr().cast();
let dlpack: legacy::Dlpack =
    Builder::new(Box::new(data), GenericArray::new(&shape, &strides))
        .data(data_ptr)
        .build();
```

The generic path converts each value directly into the trailing `i64`
metadata storage. In the included length-64 microbenchmark, direct conversion
takes approximately 11.7 ns, while allocating temporary `Vec<i64>` storage
and then calling `copy_nonoverlapping` takes approximately 54.9 ns on the
development machine. Reproduce it with:

```bash
cargo bench --bench builder -- generic_metadata_copy
```

The ndarray exporter uses the same direct path for its `usize` shape and
`isize` strides, so exporting an owned array does not allocate temporary
`Vec<i64>` metadata.

### Python Exchange Paths

The `pyo3` feature supports the standard Python DLPack capsule protocol:

- `legacy::Dlpack` consumes or produces legacy `"dltensor"` capsules.
- `versioned::Dlpack` consumes or produces `"dltensor_versioned"` capsules.
- `python::dlpack_device(obj)` calls and validates `obj.__dlpack_device__()`, returning a Rust `DLDevice`.
- When extracting a versioned tensor from a Python object, dlpark first checks the object's type for a `__dlpack_c_exchange_api__` PyCapsule named `"dlpack_exchange_api"`. If present, it uses the DLPack C Exchange API no-sync function table. Otherwise it calls `obj.__dlpack__(max_version=(1, 3))`. Producers that only implement the legacy no-argument protocol must be extracted as `legacy::Dlpack`, because they return the incompatible `"dltensor"` capsule ABI.
- Consumers can call `versioned::Dlpack::extract_with_options(obj, stream, copy)` to pass optional stream and tri-state copy requests to `__dlpack__`; `extract_with_stream` is the typed convenience path for GPU consumers. The `cudarc` feature implements stream mapping for `CudaStream`; other backends can implement the unsafe `python::DlpackStream` trait.

The C Exchange API is intended for extension/library use where the consumer can borrow tensors and coordinate work on the producer's current stream. It is not a replacement for the normal `__dlpack__` ingestion path.

## Features

| Feature   | Description                                                                                                          | Status |
| --------- | -------------------------------------------------------------------------------------------------------------------- | ------ |
| `pyo3`    | Python interop via [pyo3] (capsule protocol + DLPack C Exchange API fast path)                                       | ✅     |
| `image`   | Zero-copy conversion with [image] buffers                                                                            | ✅     |
| `ndarray` | Zero-copy conversion with [ndarray] arrays/views                                                                     | ✅     |
| `half`    | `f16`/`bf16` element type support (via [half])                                                                       | ✅     |
| `candle`  | Conversion with [candle] `Tensor` — CPU only; candle's CUDA backend needs separate integration work                  | ✅     |
| `cudarc`  | Zero-copy conversion with [cudarc] `CudaSlice<T>` — no automated tests here, needs a CUDA-capable device to exercise | ✅     |

## Quick Start

Two runnable examples:

- [`examples/dlparkimg`](./examples/dlparkimg/) — a Python extension module (via `pyo3`) transferring `image::RgbImage` to/from Python (e.g. `torch.Tensor`).
- [`examples/ndarray-candle`](./examples/ndarray-candle/) — a plain binary round-tripping data through DLPack: `ndarray::Array2` → `versioned::Dlpack` → `candle::Tensor` → `versioned::Dlpack` → `ndarray` view, run with `cargo run -p ndarray-candle`.

## Usage Examples

### Converting between Rust and Python

```rust
use dlpark::{Builder, versioned};
use pyo3::prelude::*;

// Rust to Python
#[pyfunction]
fn read_image(filename: &str) -> versioned::Dlpack {
    let img = image::open(filename).unwrap().to_rgb8();
    Builder::from(img).build()
}

// Python to Rust
#[pyfunction]
fn write_image(filename: &str, tensor: versioned::Dlpack) {
    let img: image::RgbImage = (&tensor).try_into().unwrap();
    img.save(filename).unwrap();
}
```

### Image Processing

```rust
use dlpark::{Builder, versioned};
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let tensor: versioned::Dlpack = Builder::from(img).build();
let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&tensor)?;
```

### ndarray

```rust
use dlpark::{Builder, versioned};
use ndarray::{ArrayD, ArrayViewD, arr2};

let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
let tensor: versioned::Dlpack = Builder::from(array).try_build()?;
let view = ArrayViewD::<i32>::try_from(&tensor)?;

assert_eq!(view[[1, 2]], 6);

let dynamic: ArrayD<i32> = arr2(&[[1, 2], [3, 4]]).into_dyn();
let dynamic_tensor: versioned::Dlpack = Builder::from(dynamic).try_build()?;
```

### candle

Zero-copy from `candle::Tensor` to DLPack; the reverse direction (DLPack to `candle::Tensor`) always copies, since candle has no borrowed CPU tensor type.

```rust
use candle_core::Tensor;
use dlpark::{Builder, DlpackFlags, ffi::DLManagedTensorVersioned, versioned};

let tensor = Tensor::new(&[1f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let dlpack: versioned::Dlpack = Builder::try_from(tensor)?.try_build()?;

let tensor2 = Tensor::try_from(&dlpack)?;
assert_eq!(tensor2.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);

let tensor = Tensor::new(&[1f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let builder = Builder::try_from(tensor)?;
let dlpack = builder
    .insert_flags(DlpackFlags::READ_ONLY)?
    .try_build::<DLManagedTensorVersioned>()?;
```

### cudarc

Zero-copy in both directions between a [cudarc] `CudaSlice<T>` and a DLPack tensor. `Builder::try_from` returns a builder with `IS_COPIED` and a contiguous 1-D default layout (`shape = [len]`, `strides = [1]`); replace its metadata for higher-rank tensors. The reverse direction consumes the managed tensor through `TryFrom<ManagedBox<M>> for BorrowedCudaSlice<M, T>`, keeping it alive for as long as the CUDA view exists.

```rust
use dlpark::{
    Builder,
    interop::cudarc::BorrowedCudaSlice,
    metadata::CopiedSlice,
    versioned,
};

let dlpack: versioned::Dlpack = Builder::try_from(cuda_slice)?
    .metadata(CopiedSlice::new([2, 3], [3, 1]))
    .try_build()?;
let borrowed = BorrowedCudaSlice::<_, f32>::try_from(dlpack)?;
```

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image
[ndarray]: https://github.com/rust-ndarray/ndarray
[half]: https://crates.io/crates/half
[candle]: https://github.com/huggingface/candle
[cudarc]: https://crates.io/crates/cudarc
