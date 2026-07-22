//! Legacy DLPack managed tensor ownership.

use crate::{Local, ffi::DLManagedTensor};

/// Owning handle for the legacy `DLManagedTensor` ABI.
pub type Dlpack = Local<DLManagedTensor>;
