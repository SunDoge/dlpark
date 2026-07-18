use crate::{ManagedBox, ffi::DLManagedTensorVersioned};

pub type Dlpack = ManagedBox<DLManagedTensorVersioned>;
