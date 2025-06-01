use super::MemoryLayout;
use crate::ffi;
use crate::ffi::Tensor;

pub trait TensorLike<L>
where
    L: MemoryLayout,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn memory_layout(&self) -> L;
    fn device(&self) -> ffi::Device;
    fn data_type(&self) -> ffi::DataType;
    fn byte_offset(&self) -> u64;
}

impl Tensor {
    pub fn update<T, L>(&mut self, t: &T, layout: &L)
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        self.data = t.data_ptr();
        self.device = t.device();
        self.dtype = t.data_type();
        self.byte_offset = t.byte_offset();
        self.ndim = layout.ndim();
        self.shape = layout.shape_ptr();
        self.strides = layout.strides_ptr();
    }
}
