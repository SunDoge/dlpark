use std::ptr::NonNull;

use crate::{
    data_type::DataType,
    device::Device,
    managed_tensor::{IntoDlpack, ManagedTensor},
    memory_layout::MemoryLayout,
};

pub trait TensorLike<L>
where
    L: MemoryLayout,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn memory_layout(&self) -> L;
    fn device(&self) -> Device;
    fn data_type(&self) -> DataType;
    fn byte_offset(&self) -> u64;
}

pub struct ManagerContext<T, L> {
    inner: T,
    memory_layout: L,
    managed_tensor: ManagedTensor,
}

unsafe extern "C" fn deleter<T>(managed_tensor: *mut ManagedTensor) {
    // https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_raw
    // Use from_raw to clean it.
    unsafe {
        let ctx = (*managed_tensor).manager_ctx as *mut T;
        let _ = Box::from_raw(ctx);
    };
}

impl<T, L> ManagerContext<T, L>
where
    T: TensorLike<L>,
    L: MemoryLayout,
{
    pub fn new(tensor: T) -> Box<Self> {
        let memory_layout = tensor.memory_layout();
        Box::new(Self {
            inner: tensor,
            memory_layout,
            managed_tensor: ManagedTensor::default(),
        })
    }
}

impl<T, L> IntoDlpack for Box<ManagerContext<T, L>>
where
    T: TensorLike<L>,
    L: MemoryLayout,
{
    fn into_dlpack(mut self) -> NonNull<ManagedTensor> {
        self.managed_tensor
            .dl_tensor
            .update(&self.inner, &self.memory_layout);
        self.managed_tensor.deleter.replace(deleter::<Self>);
        let ptr = Box::into_raw(self);
        unsafe {
            (*ptr).managed_tensor.manager_ctx = ptr as *mut _;
            NonNull::new_unchecked(&mut (*ptr).managed_tensor)
        }
    }
}
