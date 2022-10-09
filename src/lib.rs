/// DLPACK_VERSION 70
/// DLPACK_ABI_VERSION 1
pub mod dlpack;

pub use dlpack::{DataType, DataTypeCode, Device, DeviceType, ManagedTensor, Tensor};
