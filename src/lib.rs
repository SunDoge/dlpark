//! Safe ownership and interoperability helpers for [DLPack].
//!
//! Producers convert into an allocation-specific [`allocation::Initialized`],
//! configure scalar fields, and finish it as a locally produced tensor.
//!
//! ```
//! # #[cfg(feature = "ndarray")]
//! # {
//! use dlpark::{DlpackFlags, Local, allocation::dynamic, ffi::DLManagedTensorVersioned};
//! use ndarray::arr1;
//!
//! let mut initialized: dynamic::Initialized<DLManagedTensorVersioned> =
//!     Box::new(arr1(&[1_i32, 2, 3])).try_into().unwrap();
//! initialized.set_flags(DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY).unwrap();
//! let tensor: Local<DLManagedTensorVersioned> = unsafe { initialized.finish() };
//! assert_eq!(tensor.shape().unwrap(), &[3]);
//! # }
//! ```
//!
//! [`legacy::Dlpack`] uses `DLManagedTensor`; [`versioned::Dlpack`] uses
//! `DLManagedTensorVersioned` and exposes version and flags.
//!
//! [DLPack]: https://dmlc.github.io/dlpack/latest/

/// Raw C ABI declarations generated from the bundled DLPack headers.
#[allow(
    missing_docs,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
pub mod ffi;

pub mod allocation;
mod borrowed;
mod context;
mod convert;
/// Owning managed-tensor handles and data accessors.
pub mod dlpack;
/// Adapters for supported Rust tensor and image libraries.
pub mod interop;

mod data_type;
mod device;

mod managed_tensor;
#[cfg(feature = "pyo3")]
/// Python DLPack capsule, stream, and exchange API support.
pub mod python;
/// Validation and data access methods for raw `DLTensor` values.
pub mod tensor;
#[cfg(test)]
mod test_support;

/// Legacy `DLManagedTensor` ownership type.
pub mod legacy;
/// Shape and stride metadata composed with managed tensor allocations.
pub mod metadata;
/// Versioned `DLManagedTensorVersioned` ownership type.
pub mod versioned;

pub use borrowed::Borrowed;
pub use context::OpaqueContext;
pub use convert::TryFromDlpack;
pub use data_type::DlpackElement;
pub use dlpack::{Foreign, Local};
pub use managed_tensor::{DlpackFlags, ManagedTensorBase};
