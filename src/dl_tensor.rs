use crate::ffi::DLTensor;

impl Default for DLTensor {
    fn default() -> Self {
        DLTensor {
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
