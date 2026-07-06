# dlpark

[![Github Actions](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
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

- `Dlpack`: Legacy managed tensor capsule support
- `DlpackVersioned`: Versioned managed tensor capsule support with:
  - Major version: 1
  - Minor version: 3
  - Additional flags for tensor properties (read-only, copied, sub-byte type padding)

### Safe Abstractions

The library provides a Rust ownership wrapper over the C-style `DLPack` structures, `ManagedBox<M>`:

- RAII wrapper around a raw managed `DLPack` tensor pointer
- Automatic cleanup through the DLPack deleter function on drop
- `Dlpack` and `DlpackVersioned` are convenience aliases for `ManagedBox<DLManagedTensor>` and `ManagedBox<DLManagedTensorVersioned>` — the two concrete forms you'll actually use

Other key features:

- Memory safety through Rust's ownership system
- Support for `image` buffers, [ndarray], and [candle] tensors, plus raw DLPack tensor layouts
- Python interoperability through PyO3
- Optional DLPack 1.3 C Exchange API fast path when a producer type exposes `__dlpack_c_exchange_api__`

### Choosing a `DlpackBuilder` Constructor

`DlpackBuilder<M, N>` has four ways to supply `shape`/`strides`. The right one to reach for depends on **whether the tensor's rank is known at compile time**, not on raw speed — for typical tensors (rank ≤ 16), all four are within a few nanoseconds of each other:

- **`with_array_layout`** — `shape`/`strides` as fixed-size arrays `&[T; N]`, `N` known at compile time. Use this whenever the producer has a fixed rank (e.g. the `image` HWC layout is always rank 3). A shape/strides length mismatch is a compile error here, not a runtime failure — there's no `Result` to handle for that case, because it can't happen.
- **`with_slice_layout`** — the same fixed-rank `N` as `with_array_layout`, but takes `&[T]` for when you don't already have a `[T; N]` array in hand (e.g. built at runtime into a `Vec`). Since a slice's length isn't checked at compile time, it returns `Result`, failing with `Error::SliceLengthMismatch` if `shape`/`strides` don't both have length `N`.
- **`with_dynamic_layout`** — for when the rank genuinely isn't known until runtime (wrapping a tensor from a library with dynamic rank — this is what the `ndarray` and `candle` interop use internally). Returns `Result`, since a shape/strides length mismatch and rank overflow are both real possibilities here.
- **`with_pointer_layout`** (`unsafe`) — pass raw `*mut i64` pointers to `shape`/`strides` you already own and guarantee will outlive the built tensor. The only variant that never copies `shape`/`strides` at all.

The const-generic variants (`with_array_layout`/`with_slice_layout`) exist for **correctness, not performance**: they turn a shape/strides length mismatch from a runtime failure mode into a compile error, and let the fixed-rank case skip `Result` handling entirely. Benchmarks below (`cargo bench --bench builder`, ns, lower is better; reproduce with the same command — numbers will vary by machine) show why performance isn't the deciding factor: `with_pointer_layout` is the only one that stays flat, because it's the only one that skips copying `shape`/`strides`. The other three all pay for that copy and converge to similar cost at high rank, regardless of whether `N` was known at compile time:

| ndim | `with_array_layout` | `with_slice_layout` | `with_dynamic_layout` | `with_pointer_layout` |
| ---: | ------------------: | ------------------: | --------------------: | --------------------: |
|    1 |                 7.7 |                 8.6 |                  11.6 |               **7.3** |
|    4 |               8.6\* |                 8.6 |                  10.5 |               **7.4** |
|   16 |                 9.5 |                11.2 |                  11.9 |               **7.4** |
|   64 |                41.0 |                45.0 |                  43.6 |               **7.6** |

\* run-to-run noise is normal at this scale (single-digit nanoseconds); treat differences smaller than ~2ns between `with_array_layout`/`with_slice_layout`/`with_dynamic_layout` as noise, not a real ranking.

If you don't already own `shape`/`strides` memory that will outlive the tensor, `with_pointer_layout` isn't an option — pick `with_array_layout`/`with_slice_layout` if the rank is fixed at compile time, otherwise `with_dynamic_layout`.

### Python Exchange Paths

The `pyo3` feature supports the standard Python DLPack capsule protocol:

- `Dlpack` consumes or produces legacy `"dltensor"` capsules.
- `DlpackVersioned` consumes or produces `"dltensor_versioned"` capsules.
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
- [`examples/ndarray-candle`](./examples/ndarray-candle/) — a plain binary round-tripping data through DLPack: `ndarray::Array2` → `Dlpack` → `candle::Tensor` → `Dlpack` → `ndarray` view, run with `cargo run -p ndarray-candle`.

## Usage Examples

### Converting between Rust and Python

```rust
use dlpark::Dlpack;
use pyo3::prelude::*;

// Rust to Python
#[pyfunction]
fn read_image(filename: &str) -> Dlpack {
    let img = image::open(filename).unwrap().to_rgb8();
    Dlpack::from(img)
}

// Python to Rust
#[pyfunction]
fn write_image(filename: &str, tensor: Dlpack) {
    let img: image::RgbImage = (&tensor).try_into().unwrap();
    img.save(filename).unwrap();
}
```

### Image Processing

```rust
use dlpark::Dlpack;
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let tensor = Dlpack::from(img);
let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&tensor)?;
```

### ndarray

```rust
use dlpark::Dlpack;
use ndarray::{ArrayD, ArrayViewD, arr2};

let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
let tensor = Dlpack::try_from(array)?;
let view = ArrayViewD::<i32>::try_from(&tensor)?;

assert_eq!(view[[1, 2]], 6);

let dynamic: ArrayD<i32> = arr2(&[[1, 2], [3, 4]]).into_dyn();
let dynamic_tensor = Dlpack::try_from(dynamic)?;
```

### candle

Zero-copy from `candle::Tensor` to DLPack; the reverse direction (DLPack to `candle::Tensor`) always copies, since candle has no borrowed CPU tensor type.

```rust
use candle_core::Tensor;
use dlpark::Dlpack;

let tensor = Tensor::new(&[1f32, 2., 3., 4.], &candle_core::Device::Cpu)?;
let dlpack = Dlpack::try_from(tensor)?;

let tensor2 = Tensor::try_from(&dlpack)?;
assert_eq!(tensor2.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
```

### cudarc

Zero-copy in both directions between a [cudarc] `CudaSlice<T>` and a DLPack tensor. `from_cuda_slice`/`from_cuda_slice_versioned` take `shape`/`strides` explicitly (not derivable from a flat device buffer alone); the reverse direction is `TryFrom<&ManagedBox<M>> for BorrowedCudaSlice<T>`.

```rust
use dlpark::interop::cudarc::{from_cuda_slice, BorrowedCudaSlice};

let dlpack = from_cuda_slice(cuda_slice, &[2, 3], &[3, 1])?;
let borrowed: BorrowedCudaSlice<f32> = (&dlpack).try_into()?;
```

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image
[ndarray]: https://github.com/rust-ndarray/ndarray
[half]: https://crates.io/crates/half
[candle]: https://github.com/huggingface/candle
[cudarc]: https://crates.io/crates/cudarc
