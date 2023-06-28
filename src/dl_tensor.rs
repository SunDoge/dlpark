use crate::ffi;
use crate::tensor::traits::TensorView;

impl Default for ffi::DLTensor {
    fn default() -> Self {
        ffi::DLTensor {
            data: std::ptr::null_mut(),
            ndim: 0,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
            device: Default::default(),
            dtype: Default::default(),
        }
    }
}

impl TensorView for ffi::DLTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.data
    }

    fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.shape, self.ndim()) }
    }

    fn strides(&self) -> Option<&[i64]> {
        if self.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.strides, self.ndim()) })
        }
    }

    fn ndim(&self) -> usize {
        self.ndim as usize
    }

    fn device(&self) -> ffi::Device {
        self.device
    }

    fn dtype(&self) -> ffi::DataType {
        self.dtype
    }
    fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
}
