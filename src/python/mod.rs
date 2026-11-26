//! Python DLPack protocol integration through PyO3.
//!
//! - **Import a Python tensor.** Use [`from_dlpack`](crate::python::from_dlpack) for one-step import, or
//!   [`ImportRequest`](crate::python::ImportRequest) to discover the producer's device before choosing a stream.
//! - **Export a single-use capsule.** Return a [`crate::legacy::Dlpack`] or
//!   [`crate::versioned::Dlpack`] from a PyO3 function. PyO3 converts it into a capsule.
//! - **Export a reusable Python object.** Implement [`DlpackExporter`](crate::python::DlpackExporter) and call
//!   [`export_dlpack`](crate::python::export_dlpack) from `__dlpack__` to create a fresh capsule on each successful call.
//! - **Use the optional DLPack 1.3 C Exchange API.** See [`consumer::exchange`](crate::python::consumer::exchange)
//!   for discovery and import, and [`DlpackExchangeProducer`](crate::python::DlpackExchangeProducer) for export.

mod capsule;
/// Helpers for importing Python DLPack producers into Rust.
pub mod consumer;
/// Helpers for implementing Python DLPack producers.
pub mod producer;
pub use capsule::mark_dlpack_capsule_consumed;
pub use consumer::{
    DlpackStream, ImportRequest, ImportedDlpack, StreamArg, dlpack_device, from_dlpack,
};
pub use producer::{
    CudaStreamRequest, DlpackExchangeProducer, DlpackExporter, ExportAbi, ExportRequest,
    export_dlpack, install_exchange_api,
};
