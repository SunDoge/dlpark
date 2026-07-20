//! Safe ownership and interoperability helpers for [DLPack].
//!
//! Producers are converted into a [`Builder`], which keeps the producer alive,
//! owns or borrows shape metadata according to its metadata type, and selects
//! the legacy or versioned ABI only when built. The result is a [`ManagedBox`]
//! that calls the DLPack deleter on drop.
//!
//! ```
//! # #[cfg(feature = "ndarray")]
//! # {
//! use dlpark::{Builder, DlpackFlags, versioned};
//! use ndarray::arr1;
//!
//! let tensor: versioned::Dlpack = Builder::from(arr1(&[1_i32, 2, 3]))
//!     .insert_flags(DlpackFlags::READ_ONLY)
//!     .unwrap()
//!     .try_build()
//!     .unwrap();
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

mod borrowed;
/// Deferred construction of legacy and versioned managed tensors.
pub mod builder;
mod context;
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

/// Legacy `DLManagedTensor` ownership type.
pub mod legacy;
/// Shape and stride storage strategies used by [`Builder`].
pub mod metadata;
/// Versioned `DLManagedTensorVersioned` ownership type.
pub mod versioned;

pub use borrowed::Borrowed;
pub use builder::Builder;
pub use context::OpaqueContext;
pub use data_type::DlpackElement;
pub use dlpack::ManagedBox;
pub use managed_tensor::{DlpackFlags, ManagedTensorBase};
