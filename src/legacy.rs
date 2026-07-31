//! Legacy DLPack managed tensor ABI.

/// Owning handle for the legacy `DLManagedTensor` ABI.
pub type Dlpack = crate::Managed<crate::ffi::DLManagedTensor>;
