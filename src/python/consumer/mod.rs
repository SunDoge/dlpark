//! Import helpers for consuming Python DLPack producers from Rust.
//!
//! Use [`from_dlpack`] when the consumer stream is already available. Use
//! [`ImportRequest`] when the stream must be created for the producer's device:
//!
//! ```text
//! let request = ImportRequest::new(object.as_borrowed())?;
//! let device = request.device()?;
//! let stream = backend_stream_for(device)?;
//! let tensor = request.import(Some(&stream), None)?;
//! ```

/// Queries and validates a Python producer's DLPack device.
pub mod device;
/// DLPack 1.3 C Exchange API discovery and invocation.
pub mod exchange;
mod import;
/// Encodes consumer streams for Python's `__dlpack__(stream=...)` protocol.
pub mod stream;

pub use device::dlpack_device;
pub use import::{ImportRequest, ImportedDlpack, from_dlpack};
pub use stream::{DlpackStream, StreamArg};
