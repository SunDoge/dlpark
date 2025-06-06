use std::ptr::NonNull;

use crate::ffi::{self, DlpackVersioned, Flags};
use crate::traits::{MemoryLayout, TensorLike};

pub struct ManagerContext<T, L> {
    inner: T,
    memory_layout: L,
    managed_tensor_versioned: ffi::ManagedTensorVersioned,
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
            managed_tensor_versioned: ffi::ManagedTensorVersioned::default(),
        })
    }

    pub fn into_dlpack_versioned(
        mut self: Box<Self>,
        flags: Flags,
    ) -> std::result::Result<DlpackVersioned, T::Error> {
        self.managed_tensor_versioned
            .dl_tensor
            .update(&self.inner, &self.memory_layout)?;

        self.managed_tensor_versioned
            .deleter
            .replace(deleter::<Self>);

        self.managed_tensor_versioned.flags = flags;

        let ptr = Box::into_raw(self);
        unsafe {
            let managed_tensor_versioned = &mut (*ptr).managed_tensor_versioned;
            managed_tensor_versioned.manager_ctx = ptr as *mut _;
            Ok(NonNull::new_unchecked(managed_tensor_versioned))
        }
    }
}

unsafe extern "C" fn deleter<T>(managed_tensor: *mut ffi::ManagedTensorVersioned) {
    // https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_raw
    // Use from_raw to clean it.
    unsafe {
        let ctx = (*managed_tensor).manager_ctx as *mut T;
        let _ = Box::from_raw(ctx);
    };
}
