# dlpark
[![Github Actions](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)](https://crates.io/crates/dlpark)
[![docs.rs](https://img.shields.io/docsrs/dlpark/latest?style=for-the-badge)](https://docs.rs/dlpark)


A pure Rust implementation of [dmlc/dlpack](https://github.com/dmlc/dlpack).

This implementation focuses on transferring tensor from Rust to Python and vice versa.

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

- `ManagedTensor`: Legacy managed tensor capsule support
- `VersionedManagedTensor`: Versioned managed tensor capsule support with:
  - Major version: 1
  - Minor version: 3
  - Additional flags for tensor properties (read-only, copied, sub-byte type padding)

### Safe Abstractions

The library provides Rust ownership wrappers over the C-style `DLPack` structures:

1. `OwnedDlpackTensor<M>`:
   - RAII wrapper around raw managed `DLPack` tensors
   - RAII-style memory management
   - Automatic cleanup through deleter functions
   - Safe conversion between different tensor types

2. Key Features:
   - Memory safety through Rust's ownership system
   - Zero-cost abstractions
   - Support for `image` buffers and raw DLPack tensor layouts
   - Python interoperability through PyO3
   - Optional DLPack 1.3 C Exchange API fast path when a producer type exposes `__dlpack_c_exchange_api__`

### Python Exchange Paths

The `pyo3` feature supports the standard Python DLPack capsule protocol:

- `ManagedTensor` consumes or produces legacy `"dltensor"` capsules.
- `VersionedManagedTensor` consumes or produces `"dltensor_versioned"` capsules.
- When extracting a versioned tensor from a Python object, dlpark first checks the object's type for a `__dlpack_c_exchange_api__` PyCapsule named `"dlpack_exchange_api"`. If present, it uses the DLPack C Exchange API no-sync function table. Otherwise it falls back to `obj.__dlpack__(max_version=(1, 3))`, and then to no-arg `obj.__dlpack__()` for older producers.

The C Exchange API is intended for extension/library use where the consumer can borrow tensors and coordinate work on the producer's current stream. It is not a replacement for the normal `__dlpack__` ingestion path.

## Features

| Feature   | Description                               | Status |
| --------- | ----------------------------------------- | ------ |
| `pyo3`    | Enable Python bindings with [pyo3]        | ✅      |
| `image`   | Enable [image] support                    | ✅      |
| `ndarray` | Enable [ndarray] support                  | ✅      |
| `cuda`    | Enables the optional [cuda] integration   | 🚧     |

## Quick Start

We provide a simple example of how to transfer `image::RgbImage` to Python and `torch.Tensor` to Rust.

[Full code is here](./examples/dlparkimg/).

## Usage Examples

### Converting between Rust and Python

```rust
use dlpark::ManagedTensor;
use pyo3::prelude::*;

// Rust to Python
#[pyfunction]
fn read_image(filename: &str) -> ManagedTensor {
    let img = image::open(filename).unwrap().to_rgb8();
    ManagedTensor::from(img)
}

// Python to Rust
#[pyfunction]
fn write_image(filename: &str, tensor: ManagedTensor) {
    let img: image::RgbImage = (&tensor).try_into().unwrap();
    img.save(filename).unwrap();
}
```

### Image Processing

```rust
use dlpark::ManagedTensor;
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let tensor = ManagedTensor::from(img);
let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&tensor)?;
```

### ndarray

```rust
use dlpark::ManagedTensor;
use ndarray::{ArrayD, ArrayViewD, arr2};

let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
let tensor = ManagedTensor::try_from(array)?;
let view = ArrayViewD::<i32>::try_from(&tensor)?;

assert_eq!(view[[1, 2]], 6);

let dynamic: ArrayD<i32> = arr2(&[[1, 2], [3, 4]]).into_dyn();
let dynamic_tensor = ManagedTensor::try_from(dynamic)?;
```

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image
[ndarray]: https://github.com/rust-ndarray/ndarray
[cuda]: https://github.com/chelsea0x3b/cudarc
