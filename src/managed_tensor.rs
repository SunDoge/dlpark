use std::{ffi::c_void, ptr::NonNull};

use crate::{
    DlpackFlags,
    ffi::{DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor},
};

pub trait ManagedTensor {
    fn new(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self;
    fn get_dltensor(&self) -> &DLTensor;
    fn get_dltensor_mut(&mut self) -> &mut DLTensor;
    fn manager_ctx_ptr(&self) -> *mut c_void;

    /// Call the FFI deleter on the given pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is a valid pointer to `Self` and has not been dropped/freed yet.
    unsafe fn call_deleter(ptr: *mut Self);
}

impl ManagedTensor for DLManagedTensor {
    fn new(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self {
        Self {
            dl_tensor: tensor,
            manager_ctx,
            deleter,
        }
    }

    fn get_dltensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn get_dltensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx_ptr(&self) -> *mut c_void {
        self.manager_ctx
    }
    unsafe fn call_deleter(ptr: *mut Self) {
        if let Some(deleter) = unsafe { (*ptr).deleter } {
            unsafe { deleter(ptr) };
        }
    }
}

impl ManagedTensor for DLManagedTensorVersioned {
    fn new(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self {
        Self {
            version: DLPackVersion::default(),
            manager_ctx,
            deleter,
            flags: DlpackFlags::empty(),
            dl_tensor: tensor,
        }
    }

    fn get_dltensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn get_dltensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx_ptr(&self) -> *mut c_void {
        self.manager_ctx
    }
    unsafe fn call_deleter(ptr: *mut Self) {
        if let Some(deleter) = unsafe { (*ptr).deleter } {
            unsafe { deleter(ptr) };
        }
    }
}

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
