use crate::{
    data_type::InferDataType, device::Device, manager_context::TensorLike,
    memory_layout::ContiguousLayout,
};

impl<A> TensorLike<ContiguousLayout> for Vec<A>
where
    A: InferDataType,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut A as *mut _
    }

    fn data_type(&self) -> crate::data_type::DataType {
        A::infer_dtype()
    }

    fn memory_layout(&self) -> ContiguousLayout {
        ContiguousLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> crate::device::Device {
        Device::CPU
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}
