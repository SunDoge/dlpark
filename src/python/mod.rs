//! Python DLPack protocol integration through PyO3.
//!
//! - [`crate::python::consumer`] contains device and stream negotiation used while importing
//!   a Python tensor into Rust.
//! - [`crate::python::consumer::exchange`] exposes the optional DLPack 1.3 C Exchange API.
//!
//! Conversion between [`crate::Managed`] tensors and Python DLPack capsules is
//! implemented internally and does not add another public API layer.

mod capsule;
/// Helpers for importing Python DLPack producers into Rust.
pub mod consumer;
/// Helpers for implementing Python DLPack producers.
pub mod producer;
pub use consumer::{
    DlpackStream, ImportRequest, ImportedDlpack, StreamArg, dlpack_device, from_dlpack,
};
pub use producer::{
    CudaStreamRequest, DlpackExchangeProducer, DlpackExporter, ExportAbi, ExportRequest,
    export_dlpack, install_exchange_api,
};
