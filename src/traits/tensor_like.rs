use crate::ffi;
use crate::ffi::Tensor;

pub trait TensorLike {
    type Error;
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    /// Shape of the tensor. Each element is the size of the corresponding dimension.
    fn shape(&self) -> Vec<i64>;
    /// Strides of the tensor in number of elements (not bytes).
    /// Return `None` to indicate the tensor is compact and row-major.
    fn strides(&self) -> Option<Vec<i64>>;
    fn device(&self) -> Result<ffi::Device, Self::Error>;
    fn data_type(&self) -> Result<ffi::DataType, Self::Error>;
    fn byte_offset(&self) -> u64;
}

impl Tensor {
    pub fn update<T>(&mut self, t: &T, shape: &[i64], strides: Option<&[i64]>) -> Result<(), T::Error>
    where
        T: TensorLike,
    {
        self.data = t.data_ptr();
        self.device = t.device()?;
        self.dtype = t.data_type()?;
        self.byte_offset = t.byte_offset();
        self.ndim = shape.len() as i32;
        self.shape = shape.as_ptr() as *mut i64;
        self.strides = strides
            .map(|s| s.as_ptr() as *mut i64)
            .unwrap_or(std::ptr::null_mut());
        Ok(())
    }
}
