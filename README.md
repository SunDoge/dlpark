# dlpark

[![Tests](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge&label=test)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
[![Clippy](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/clippy.yml?branch=main&style=for-the-badge&label=clippy)](https://github.com/SunDoge/dlpark/actions/workflows/clippy.yml)
[![Miri](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/miri.yml?branch=main&style=for-the-badge&label=miri)](https://github.com/SunDoge/dlpark/actions/workflows/miri.yml)
[![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)](https://crates.io/crates/dlpark)
[![docs.rs](https://img.shields.io/docsrs/dlpark/latest?style=for-the-badge)](https://docs.rs/dlpark)

A pure Rust implementation of [DLPack](https://github.com/dmlc/dlpack) for
zero-copy tensor exchange. It supports the legacy and versioned ABIs, Rust
container adapters, the Python Array API protocol, and DLPack 1.3 C Exchange.

## Installation

No features are enabled by default. Enable only the integrations you use:

```bash
cargo add dlpark --features "ndarray half"
cargo add dlpark --features "pyo3 image"
cargo add dlpark --features "cudarc"
cargo add dlpark --features "safetensors"
```

## Quick start

A producer owns its container through the managed tensor's deleter. The
versioned ABI is recommended for new code:

```rust
use dlpark::{allocation::dynamic, ffi::DLManagedTensorVersioned, versioned};
use ndarray::arr2;

let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
    Box::new(arr2(&[[1_i32, 2, 3], [4, 5, 6]])).try_into()?;
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };

let tensor = dlpack.validate()?;
assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(unsafe { tensor.cpu_slice::<i32>()? }, &[1, 2, 3, 4, 5, 6]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`finish` is unsafe because the caller must ensure that the completed
descriptor obeys the DLPack lifetime, pointer, layout, and flag requirements.
Imported descriptors should be checked with `Managed::validate()` before their
metadata is used.

## Features

| Feature | Integration | Data movement |
| --- | --- | --- |
| `pyo3` | Python capsule protocol and C Exchange API | zero-copy protocol layer |
| `image` | `ImageBuffer` producer and consumer | zero-copy |
| `ndarray` | owned arrays and borrowed views | zero-copy |
| `half` | `f16` and `bf16` element types | — |
| `candle` | CPU `Tensor` | zero-copy export, copying import |
| `cudarc` | `CudaSlice<T>` | zero-copy |
| `safetensors` | file/mmap export and serialization views | zero-copy |

Raw CUDA and Metal runtime policy remains application-specific. The
`demos/cuda-python` and `demos/metal-python` projects show complete native
buffer implementations.

## Guides

- [Ownership, validation, metadata, and versioning](docs/core-api.md)
- [Python import, reusable export, streams, and C Exchange](docs/python.md)
- [Interop backend behavior](docs/interop.md)
- [Development and binding regeneration](docs/development.md)

Runnable demos live under [`demos/`](demos/):

- `cuda-python`: CuPy and Torch exchange through a Rust-owned CUDA buffer.
- `metal-python`: MLX exchange through a shared `MTLBuffer`.
- `image-python`: Torch exchange with `image::RgbImage`.
- `ndarray-candle`: an `ndarray → DLPack → candle → DLPack → ndarray` round trip.

The API reference is available on [docs.rs](https://docs.rs/dlpark).
