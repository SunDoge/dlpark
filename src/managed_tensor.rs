use std::ffi::c_void;

use crate::{
    DlpackFlags,
    ffi::{DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor},
};

pub trait ManagedTensorBase {
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

impl ManagedTensorBase for DLManagedTensor {
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

impl ManagedTensorBase for DLManagedTensorVersioned {
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
