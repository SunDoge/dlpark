use std::{ffi::c_void, ptr::NonNull};

use crate::{
    DLPackFlags,
    ffi::{DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor},
};

pub trait AsManagedTensor {
    fn new(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self;
    fn get_dltensor(&self) -> &DLTensor;
    fn get_dltensor_mut(&mut self) -> &mut DLTensor;
    fn manager_ctx_ptr(&self) -> *mut c_void;
}

impl AsManagedTensor for DLManagedTensor {
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
}

impl AsManagedTensor for DLManagedTensorVersioned {
    fn new(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self {
        Self {
            version: DLPackVersion::default(),
            manager_ctx,
            deleter,
            flags: DLPackFlags::empty(),
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
}

pub struct ManagedTensor<M>(NonNull<M>);

impl<M> ManagedTensor<M>
where
    M: AsManagedTensor,
{
    pub fn new(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(ManagedTensor)
    }

    pub unsafe fn new_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    pub fn dl_tensor(&self) -> &DLTensor {
        unsafe { self.0.as_ref().get_dltensor() }
    }
}
