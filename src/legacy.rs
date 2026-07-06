use crate::{Builder, ManagedBox, ffi::DLManagedTensor};

pub type Dlpack = ManagedBox<DLManagedTensor>;
pub type DlpackBuilder<const N: usize> = Builder<DLManagedTensor, N>;
pub type DynamicDlpackBuilder = DlpackBuilder<0>;
