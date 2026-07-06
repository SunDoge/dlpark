use crate::{Builder, ManagedBox, ffi::DLManagedTensorVersioned};

pub type Dlpack = ManagedBox<DLManagedTensorVersioned>;
pub type DlpackBuilder<const N: usize> = Builder<DLManagedTensorVersioned, N>;
pub type DynamicDlpackBuilder = DlpackBuilder<0>;
