pub mod ffi;

pub mod builder;
mod context;
pub mod dlpack;

pub mod interop;

mod data_type;
mod device;

mod managed_tenesor_versioned;
mod managed_tensor;
mod tensor;

pub use builder::{DlpackBox, DlpackBuilder};
pub use context::OpaqueContext;
pub use dlpack::Dlpack;
pub use managed_tenesor_versioned::DlpackFlags;
pub use managed_tensor::ManagedTensor;

pub type SafeManagedTensor = Dlpack<ffi::DLManagedTensor>;
pub type SafeManagedTensorVersioned = Dlpack<ffi::DLManagedTensorVersioned>;

pub mod prelude {
    pub use crate::{
        Dlpack, DlpackBox, DlpackBuilder, DlpackFlags, ManagedTensor,
        SafeManagedTensor, SafeManagedTensorVersioned,
    };
    pub use crate::context::OpaqueContext;
}
