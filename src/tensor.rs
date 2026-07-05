// use crate::ffi::{DLManagedTensor, DLManagedTensorVersioned, DLTensor};

// pub trait DltensorRef {
//     fn dl_tensor(&self) -> &DLTensor;
// }

// pub trait DltensorRefMut {
//     fn dl_tensor_mut(&mut self) -> &mut DLTensor;
// }

// impl DltensorRef for DLManagedTensor {
//     fn dl_tensor(&self) -> &DLTensor {
//         &self.dl_tensor
//     }
// }

// impl DltensorRefMut for DLManagedTensor {
//     fn dl_tensor_mut(&mut self) -> &mut DLTensor {
//         &mut self.dl_tensor
//     }
// }

// impl DltensorRef for DLManagedTensorVersioned {
//     fn dl_tensor(&self) -> &DLTensor {
//         &self.dl_tensor
//     }
// }

// impl DltensorRefMut for DLManagedTensorVersioned {
//     fn dl_tensor_mut(&mut self) -> &mut DLTensor {
//         &mut self.dl_tensor
//     }
// }

use crate::ffi::{DLDataType, DLDevice, DLTensor};

impl Default for DLTensor {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            device: DLDevice::CPU,
            ndim: 0,
            dtype: DLDataType::FLOAT32,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        }
    }
}
