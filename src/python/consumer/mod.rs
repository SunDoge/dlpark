//! Import helpers for consuming Python DLPack producers from Rust.

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
