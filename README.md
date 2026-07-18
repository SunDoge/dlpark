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
- **`BorrowedArray<N>`** — fixed-rank, zero-copy metadata. Its `build` methods are unsafe because the arrays must outlive the managed tensor.
- **`BorrowedSlice`** — dynamic-rank, zero-copy metadata. Its `try_build` methods are unsafe for the same lifetime reason.

Copied metadata and the managed tensor share one allocation. Borrowed metadata only allocates the managed tensor header.

### Python Exchange Paths

The `pyo3` feature supports the standard Python DLPack capsule protocol:

- `legacy::Dlpack` consumes or produces legacy `"dltensor"` capsules.
- `versioned::Dlpack` consumes or produces `"dltensor_versioned"` capsules.
- When extracting a versioned tensor from a Python object, dlpark first checks the object's type for a `__dlpack_c_exchange_api__` PyCapsule named `"dlpack_exchange_api"`. If present, it uses the DLPack C Exchange API no-sync function table. Otherwise it falls back to `obj.__dlpack__(max_version=(1, 3))`, and then to no-arg `obj.__dlpack__()` for older producers.

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
- [`examples/ndarray-candle`](./examples/ndarray-candle/) — a plain binary round-tripping data through DLPack: `ndarray::Array2` → `legacy::Dlpack` → `candle::Tensor` → `legacy::Dlpack` → `ndarray` view, run with `cargo run -p ndarray-candle`.

## Usage Examples

### Converting between Rust and Python

```rust
use dlpark::legacy;
use pyo3::prelude::*;

// Rust to Python
#[pyfunction]
fn read_image(filename: &str) -> legacy::Dlpack {
    let img = image::open(filename).unwrap().to_rgb8();
    legacy::Dlpack::from(img)
}

// Python to Rust
#[pyfunction]
fn write_image(filename: &str, tensor: legacy::Dlpack) {
    let img: image::RgbImage = (&tensor).try_into().unwrap();
    img.save(filename).unwrap();
}
```

### Image Processing

```rust
use dlpark::legacy;
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let tensor = legacy::Dlpack::from(img);
let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&tensor)?;
```

### ndarray

```rust
use dlpark::legacy;
use ndarray::{ArrayD, ArrayViewD, arr2};

let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
let tensor = legacy::Dlpack::try_from(array)?;
let view = ArrayViewD::<i32>::try_from(&tensor)?;

assert_eq!(view[[1, 2]], 6);

let dynamic: ArrayD<i32> = arr2(&[[1, 2], [3, 4]]).into_dyn();
let dynamic_tensor = legacy::Dlpack::try_from(dynamic)?;
```

### candle

Zero-copy from `candle::Tensor` to DLPack; the reverse direction (DLPack to `candle::Tensor`) always copies, since candle has no borrowed CPU tensor type.

```rust
use candle_core::Tensor;
use dlpark::{Builder, DlpackFlags, ffi::DLManagedTensorVersioned, legacy};

let tensor = Tensor::new(&[1f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let dlpack = legacy::Dlpack::try_from(tensor)?;

let tensor2 = Tensor::try_from(&dlpack)?;
assert_eq!(tensor2.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);

let tensor = Tensor::new(&[1f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let builder = Builder::try_from(tensor)?;
let dlpack = builder
    .flags(DlpackFlags::READ_ONLY)
    .try_build::<DLManagedTensorVersioned>()?;
```

### cudarc

Zero-copy in both directions between a [cudarc] `CudaSlice<T>` and a DLPack tensor. `from_cuda_slice`/`from_cuda_slice_versioned` take `shape`/`strides` explicitly (not derivable from a flat device buffer alone); the reverse direction consumes the managed tensor through `TryFrom<ManagedBox<M>> for BorrowedCudaSlice<M, T>`, keeping it alive for as long as the CUDA view exists.

```rust
use dlpark::interop::cudarc::{from_cuda_slice, BorrowedCudaSlice};

let dlpack = from_cuda_slice(cuda_slice, &[2, 3], &[3, 1])?;
let borrowed = BorrowedCudaSlice::<_, f32>::try_from(dlpack)?;
```

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image
[ndarray]: https://github.com/rust-ndarray/ndarray
[half]: https://crates.io/crates/half
[candle]: https://github.com/huggingface/candle
[cudarc]: https://crates.io/crates/cudarc
