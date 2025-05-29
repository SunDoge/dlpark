use std::ptr::NonNull;

use crate::managed_tensor_versioned::{IntoDlpackVersioned, ManagedTensorVersioned};
use crate::manager_context::TensorLike;
use crate::memory_layout::MemoryLayout;

pub struct ManagerContextVersioned<T, L> {
    inner: T,
    memory_layout: L,
    managed_tensor_versioned: ManagedTensorVersioned,
}
impl<T, L> ManagerContextVersioned<T, L>
where
    T: TensorLike<L>,
    L: MemoryLayout,
{
    pub fn new(tensor: T) -> Box<Self> {
        let memory_layout = tensor.memory_layout();
        Box::new(Self {
            inner: tensor,
            memory_layout,
            managed_tensor_versioned: ManagedTensorVersioned::default(),
        })
    }
}

unsafe extern "C" fn deleter<T>(managed_tensor: *mut ManagedTensorVersioned) {
    // https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_raw
    // Use from_raw to clean it.
    unsafe {
        let ctx = (*managed_tensor).manager_ctx as *mut T;
        let _ = Box::from_raw(ctx);
    };
}

impl<T, L> IntoDlpackVersioned for Box<ManagerContextVersioned<T, L>>
where
    T: TensorLike<L>,
    L: MemoryLayout,
{
    fn into_dlpack_versioned(
        mut self,
        flags: crate::managed_tensor_versioned::Flags,
    ) -> NonNull<ManagedTensorVersioned> {
        self.managed_tensor_versioned
            .dl_tensor
            .update(&self.inner, &self.memory_layout);

        self.managed_tensor_versioned
            .deleter
            .replace(deleter::<Self>);

        self.managed_tensor_versioned.flags = flags.bits();

        let ptr = Box::into_raw(self);
        unsafe {
            let managed_tensor_versioned = &mut (*ptr).managed_tensor_versioned;
            managed_tensor_versioned.manager_ctx = ptr as *mut _;
            NonNull::new_unchecked(managed_tensor_versioned)
        }
    }
}
