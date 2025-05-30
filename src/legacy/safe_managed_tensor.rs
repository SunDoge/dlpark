use super::ManagerContext;
use crate::ffi;
use crate::ffi::TensorView;
use crate::traits::{MemoryLayout, TensorLike};
use std::ptr::NonNull;

pub struct SafeManagedTensor(ffi::Dlpack);

impl Drop for SafeManagedTensor {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl SafeManagedTensor {
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensor) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    pub unsafe fn from_non_null(ptr: ffi::Dlpack) -> Self {
        Self(ptr)
    }

    pub unsafe fn into_raw(self) -> *mut ffi::ManagedTensor {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    pub fn into_non_null(self) -> ffi::Dlpack {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    pub fn new<T, L>(t: T) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        let ctx = ManagerContext::new(t);
        Self(ctx.into_dlpack())
    }
}

impl TensorView for SafeManagedTensor {
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}
