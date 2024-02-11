use crate::{ffi::{self, DLManagedTensor}, tensor::traits::TensorView};

impl TensorView for ffi::DLManagedTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.dl_tensor.data_ptr()
    }

    fn byte_offset(&self) -> u64 {
        self.dl_tensor.byte_offset()
    }

    fn device(&self) -> ffi::Device {
        self.dl_tensor.device()
    }

    fn dtype(&self) -> ffi::DataType {
        self.dl_tensor.dtype()
    }

    fn shape(&self) -> &[i64] {
        self.dl_tensor.shape()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.dl_tensor.strides()
    }

    fn ndim(&self) -> usize {
        self.dl_tensor.ndim()
    }
}

unsafe impl Send for DLManagedTensor {}