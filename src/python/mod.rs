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
/// Helpers for implementing Python DLPack producers.
pub mod producer;
/// DLPack 1.3 C Exchange API discovery and invocation.
pub use consumer::exchange;
pub use consumer::{
    DlpackStream, ImportRequest, ImportedDlpack, StreamArg, dlpack_device, from_dlpack,
};
pub use producer::{
    CudaStreamRequest, DlpackExchangeProducer, ExportAbi, ExportRequest, install_exchange_api,
};

// Keep the former module paths working while presenting `consumer` as the
// documented organization.
#[doc(hidden)]
pub use consumer::{device, stream};
