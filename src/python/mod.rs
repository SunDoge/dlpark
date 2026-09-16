//! Python DLPack protocol integration through PyO3.
//!
//! - [`crate::python::consumer`] contains device and stream negotiation used while importing
//!   a Python tensor into Rust.
//! - [`crate::python::exchange`] implements the optional DLPack 1.3 C Exchange API fast path.
//!
//! Conversion between [`crate::Managed`] tensors and Python DLPack capsules is
//! implemented internally and does not add another public API layer.

mod capsule;
/// Helpers for importing Python DLPack producers into Rust.
pub mod consumer;
/// DLPack 1.3 C exchange API integration.
pub mod exchange;
pub use consumer::{DlpackStream, ImportedDlpack, StreamArg, dlpack_device, from_dlpack};

// Keep the former module paths working while presenting `consumer` as the
// documented organization.
#[doc(hidden)]
pub use consumer::{device, stream};
