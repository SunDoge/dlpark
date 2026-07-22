//! Ownership wrappers for local and foreign DLPack managed tensors.

mod foreign;
mod local;

pub use foreign::Foreign;
pub use local::Local;
