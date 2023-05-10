/// DLPACK_VERSION 70
/// DLPACK_ABI_VERSION 1
pub mod data_type;
pub mod device;
pub mod ffi;
pub mod tensor;

pub use ffi::{DataType, Device};
pub use tensor::{ManagedTensor, ManagerCtx, Shape, Strides};

#[cfg(feature = "python")]
pub mod python;
