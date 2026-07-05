use crate::ManagedTensor;
use crate::ffi::DLTensor;
use std::ptr::NonNull;

pub struct Dlpack<M: ManagedTensor>(NonNull<M>);

impl<M> Dlpack<M>
where
    M: ManagedTensor,
{
    pub fn new(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(Dlpack)
    }

    /// Create a new `Dlpack` instance from a raw pointer without checking if it is null.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is not null and points to a valid `M`.
    pub unsafe fn new_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    pub fn dl_tensor(&self) -> &DLTensor {
        unsafe { self.0.as_ref().get_dltensor() }
    }
}

impl<M> Drop for Dlpack<M>
where
    M: ManagedTensor,
{
    fn drop(&mut self) {
        unsafe {
            M::call_deleter(self.0.as_ptr());
        }
    }
}
