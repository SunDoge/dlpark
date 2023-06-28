use crate::{
    ffi::{self, DLPACK_FLAG_BITMASK_READ_ONLY},
    prelude::TensorView,
};

// TODO: DLManagedTensor may be deprecated in the future.
impl Default for ffi::DLManagedTensorVersioned {
    fn default() -> Self {
        ffi::DLManagedTensorVersioned {
            version: Default::default(),
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
            flags: DLPACK_FLAG_BITMASK_READ_ONLY,
            dl_tensor: Default::default(),
        }
    }
}

// FIXME: it's unsafe to access it when not initialized
impl TensorView for ffi::DLManagedTensorVersioned {
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
