//! Safe ownership and interoperability helpers for [DLPack].
//!
//! Producers convert into a [`Builder`] or directly into a [`ManagedBox`].
//! Both keep the producer alive, and `ManagedBox` calls the DLPack deleter on
//! drop.
//!
//! ```
//! # #[cfg(feature = "ndarray")]
//! # {
//! use dlpark::{DlpackFlags, versioned};
//! use ndarray::arr1;
//!
//! let array = Box::new(arr1(&[1_i32, 2, 3]));
//! let mut tensor = versioned::Dlpack::try_from(array).unwrap();
//! // SAFETY: retaining READ_ONLY only restricts what consumers may do.
//! unsafe {
//!     tensor.flags_mut().insert(DlpackFlags::READ_ONLY);
//! }
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
