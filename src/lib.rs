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
pub use data_type::DlpackElement;
pub use dlpack::Dlpack;
pub use managed_tenesor_versioned::DlpackFlags;
pub use managed_tensor::ManagedTensor;
pub use tensor::{
    Error as TensorError, compact_strides, compact_strides_array, is_compact_strides,
};

pub mod prelude {
    pub use crate::context::OpaqueContext;
    pub use crate::{
        Dlpack, DlpackBox, DlpackBuilder, DlpackElement, DlpackFlags, ManagedTensor, TensorError,
        compact_strides, compact_strides_array, is_compact_strides,
    };
}
