# dlpark
[![Github Actions](https://img.shields.io/github/actions/workflow/status/SunDoge/dlpark/rust.yml?branch=main&style=for-the-badge)](https://github.com/SunDoge/dlpark/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)](https://crates.io/crates/dlpark)
[![docs.rs](https://img.shields.io/docsrs/dlpark/latest?style=for-the-badge)](https://docs.rs/dlpark)


A pure Rust implementation of [dmlc/dlpack](https://github.com/dmlc/dlpack).

This implementation focuses on transferring tensor from Rust to Python and vice versa.


## Features

| Feature | Description             | Status |
| ------- | ----------------------- | ------ |
| `pyo3`  | Enable Python bindings  | ✅      |
| `image` | Enable image-rs support | ✅      |

## Quick Start

We provide a simple example of how to transfer `image::RgbImage` to Python and `torch.Tensor` to Rust.

[Full code is here](./examples/dlparkimg/).