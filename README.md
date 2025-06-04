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

- `SafeManagedTensor`: Legacy implementation without versioning
- `SafeManagedTensorVersioned`: Versioned implementation (current standard) with:
  - Major version: 1
  - Minor version: 1
  - Additional flags for tensor properties (read-only, copied, sub-byte type padding)

### Safe Abstractions

The library provides safe Rust abstractions over the C-style `DLPack` structures:

1. `SafeManagedTensor` and `SafeManagedTensorVersioned`:
   - Safe wrappers around raw `DLPack` tensors
   - RAII-style memory management
   - Automatic cleanup through deleter functions
   - Safe conversion between different tensor types

2. Key Features:
   - Memory safety through Rust's ownership system
   - Zero-cost abstractions
   - Support for various tensor types (ndarray, image, standard containers)
   - Python interoperability through PyO3

## Features

| Feature   | Description                        | Status |
| --------- | ---------------------------------- | ------ |
| `pyo3`    | Enable Python bindings with [pyo3] | ✅      |
| `image`   | Enable [image] support             | ✅      |
| `ndarray` | Enable [ndarray] support           | ✅      |

## Quick Start

We provide a simple example of how to transfer `image::RgbImage` to Python and `torch.Tensor` to Rust.

[Full code is here](./examples/dlparkimg/).

## Usage Examples

### Converting between Rust and Python

```rust
use dlpark::prelude::*;

// Rust to Python
#[pyfunction]
fn send() -> SafeManagedTensorVersioned {
    let v = vec![1i32, 2, 3];
    SafeManagedTensorVersioned::new(v).unwrap()
}

// Python to Rust
#[pyfunction]
fn receive(tensor: SafeManagedTensorVersioned) {
    let s: &[i32] = tensor.as_slice_contiguous().unwrap();
    // Do your work.
}
```

### Working with ndarray

```rust
use dlpark::prelude::*;
use ndarray::ArrayD;

let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1i32, 2, 3, 4, 5, 6])?;
let tensor = SafeManagedTensorVersioned::new(arr)?;
let view = ArrayViewD::<i32>::try_from(&tensor)?;
```

### Image Processing

```rust
use dlpark::prelude::*;
use image::{ImageBuffer, Rgb};

let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])?;
let tensor = SafeManagedTensorVersioned::new(img)?;
let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&tensor)?;
```

[pyo3]: https://github.com/PyO3/pyo3
[image]: https://github.com/image-rs/image 
[ndarray]: https://github.com/rust-ndarray/ndarray