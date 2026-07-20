//! Legacy DLPack managed tensor ownership.

use crate::{ManagedBox, ffi::DLManagedTensor};

/// Owning handle for the legacy `DLManagedTensor` ABI.
pub type Dlpack = ManagedBox<DLManagedTensor>;
