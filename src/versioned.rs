//! Versioned DLPack managed tensor ownership.

use crate::{ManagedBox, ffi::DLManagedTensorVersioned};

/// Owning handle for `DLManagedTensorVersioned`.
pub type Dlpack = ManagedBox<DLManagedTensorVersioned>;
