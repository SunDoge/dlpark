//! Versioned DLPack managed tensor ownership.

use crate::{Local, ffi::DLManagedTensorVersioned};

/// Owning handle for `DLManagedTensorVersioned`.
pub type Dlpack = Local<DLManagedTensorVersioned>;
