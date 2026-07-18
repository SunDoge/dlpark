use crate::{ManagedBox, ffi::DLManagedTensor};

pub type Dlpack = ManagedBox<DLManagedTensor>;
