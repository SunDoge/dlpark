use std::ptr::NonNull;

use crate::traits::{MemoryLayout, TensorLike};

use crate::ffi;

pub struct ManagerContext<T, L> {
    inner: T,
    memory_layout: L,
    managed_tensor: ffi::ManagedTensor,
}

unsafe extern "C" fn deleter<T>(managed_tensor: *mut ffi::ManagedTensor) {
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
            managed_tensor: ffi::ManagedTensor::default(),
        })
    }

    pub fn into_dlpack(mut self: Box<Self>) -> ffi::Dlpack {
        self.managed_tensor
            .dl_tensor
            .update(&self.inner, &self.memory_layout);
        self.managed_tensor.deleter.replace(deleter::<Self>);
        let ptr = Box::into_raw(self);
        unsafe {
            let managed_tensor = &mut (*ptr).managed_tensor;
            managed_tensor.manager_ctx = ptr as *mut _;
            NonNull::new_unchecked(managed_tensor)
        }
    }
}
