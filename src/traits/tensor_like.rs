use super::MemoryLayout;
use crate::ffi;
use crate::ffi::Tensor;

pub trait TensorLike<L>
where
    L: MemoryLayout,
{
    type Error;
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn memory_layout(&self) -> L;
    fn device(&self) -> Result<ffi::Device, Self::Error>;
    fn data_type(&self) -> Result<ffi::DataType, Self::Error>;
    fn byte_offset(&self) -> u64;
}

impl Tensor {
    pub fn update<T, L>(&mut self, t: &T, layout: &L) -> Result<(), T::Error>
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        self.data = t.data_ptr();
        self.device = t.device()?;
        self.dtype = t.data_type()?;
        self.byte_offset = t.byte_offset();
        self.ndim = layout.ndim();
        self.shape = layout.shape_ptr();
        self.strides = layout.strides_ptr();
        Ok(())
    }
}
