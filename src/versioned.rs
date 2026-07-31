//! Versioned DLPack managed tensor ABI.

/// Owning handle for the current `DLManagedTensorVersioned` ABI.
pub type Dlpack = crate::Managed<crate::ffi::DLManagedTensorVersioned>;
