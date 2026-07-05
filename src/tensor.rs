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

impl DLTensor {
    /// Returns the shape of the tensor as a slice.
    ///
    /// Returns an empty slice if `ndim <= 0` or `shape` is null.
    pub fn shape(&self) -> &[i64] {
        if self.ndim <= 0 || self.shape.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.shape, self.ndim as usize) }
    }

    /// Returns the strides of the tensor as a slice, or `None` for compact row-major layout.
    ///
    /// Per the DLPack spec, a null `strides` pointer indicates a compact row-major (C-contiguous)
    /// layout where strides are implicitly derived from the shape.
    pub fn strides(&self) -> Option<&[i64]> {
        if self.strides.is_null() || self.ndim <= 0 {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.strides, self.ndim as usize) })
    }
}
