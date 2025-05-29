use crate::{
    managed_tensor_versioned::ManagedTensorVersioned, manager_context::TensorLike,
    memory_layout::MemoryLayout,
};

pub struct ManagerContext<T, L> {
    inner: T,
    memory_layout: L,
    managed_tensor_versioned: ManagedTensorVersioned,
}

unsafe extern "C" fn deleter<T>(managed_tensor_versioned: *mut ManagedTensorVersioned) {
    unsafe {
        let ctx = (*managed_tensor_versioned).manager_ctx as *mut T;
        let _ = Box::from_raw(ctx);
    }
}

impl<T, L> ManagerContext<T, L>
where
    T: TensorLike<L>,
    L: MemoryLayout,
{
}
