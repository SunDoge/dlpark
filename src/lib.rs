//! Safe ownership and interoperability helpers for [DLPack].
//!
//! Producers convert into an allocation-specific [`allocation::Initialized`],
//! configure scalar fields, and finish it as a locally produced tensor.
//!
//! ```
//! # #[cfg(feature = "ndarray")]
//! # {
//! use dlpark::{DlpackFlags, allocation::dynamic, ffi::DLManagedTensorVersioned, versioned};
//! use ndarray::arr1;
//!
//! let mut initialized: dynamic::Initialized<DLManagedTensorVersioned> =
//!     Box::new(arr1(&[1_i32, 2, 3])).try_into().unwrap();
//! initialized.set_flags(DlpackFlags::READ_ONLY).unwrap();
//! let tensor: versioned::Dlpack = unsafe { initialized.finish() };
//! assert_eq!(tensor.validate().unwrap().shape(), &[3]);
//! # }
//! ```
//!
//! [`legacy::Dlpack`] uses the legacy ABI; [`versioned::Dlpack`] uses the
//! versioned ABI and exposes version and flags.
//!
//! [DLPack]: https://dmlc.github.io/dlpack/latest/
#![allow(
    missing_docs,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
pub mod ffi;

pub mod allocation;
mod borrowed;
mod context;
mod convert;
mod data_type;
mod device;
mod version;

/// Owning managed-tensor handles and data accessors.
pub mod dlpack;
/// Adapters for supported Rust tensor and image libraries.
pub mod interop;
/// Legacy `DLManagedTensor` ownership alias.
pub mod legacy;

mod managed_tensor;
#[cfg(feature = "pyo3")]
/// Python DLPack capsule, stream, and exchange API support.
pub mod python;

/// Validation and data access methods for raw `DLTensor` values.
pub mod tensor;
/// Versioned `DLManagedTensorVersioned` ownership alias.
pub mod versioned;

/// Shape and stride metadata composed with managed tensor allocations.
pub mod metadata;

pub use borrowed::Borrowed;
pub use context::OpaqueContext;
pub use convert::TryFromDlpack;
pub use data_type::DlpackElement;
pub use dlpack::Managed;
pub use managed_tensor::{DlpackFlags, ManagedTensorBase};
pub use tensor::{TensorMut, TensorRef};
pub use version::VersionError;
