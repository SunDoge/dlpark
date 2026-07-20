//! Python DLPack protocol integration through PyO3.

mod capsule;
#[cfg(feature = "cudarc")]
mod cudarc;
/// Device discovery for Python DLPack producers.
pub mod device;
/// DLPack 1.3 C exchange API integration.
pub mod exchange;
/// Python stream argument encoding.
pub mod stream;

pub use device::dlpack_device;
pub use stream::{DlpackStream, StreamArg};
