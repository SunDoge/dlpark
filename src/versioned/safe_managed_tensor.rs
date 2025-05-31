use std::ptr::NonNull;

use crate::ffi::{self, Flags};
use crate::traits::{MemoryLayout, TensorLike, TensorView};

use super::ManagerContext;

pub struct SafeManagedTensorVersioned(ffi::DlpackVersioned);

impl Drop for SafeManagedTensorVersioned {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl SafeManagedTensorVersioned {
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensorVersioned) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    pub unsafe fn from_non_null(ptr: ffi::DlpackVersioned) -> Self {
        Self(ptr)
    }

    pub unsafe fn into_raw(self) -> *mut ffi::ManagedTensorVersioned {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    pub fn into_non_null(self) -> ffi::DlpackVersioned {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    pub fn new<T, L>(t: T) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        Self::with_flags(t, Flags::default())
    }

    pub fn with_flags<T, L>(t: T, flags: Flags) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        let ctx = ManagerContext::new(t);
        Self(ctx.into_dlpack_versioned(flags))
    }

    pub fn flags_bits(&self) -> u64 {
        unsafe { self.0.as_ref().flags }
    }

    pub fn flags(&self) -> Option<Flags> {
        Flags::from_bits(self.flags_bits())
    }

    pub fn flags_truncate(&self) -> Flags {
        Flags::from_bits_truncate(self.flags_bits())
    }

    pub fn read_only(&self) -> bool {
        self.flags_truncate().contains(Flags::READ_ONLY)
    }

    pub fn is_copied(&self) -> bool {
        self.flags_truncate().contains(Flags::IS_COPIED)
    }

    pub fn is_subbtype_type_padded(&self) -> bool {
        self.flags_truncate()
            .contains(Flags::IS_SUBBYTE_TYPE_PADDED)
    }
}

impl TensorView for SafeManagedTensorVersioned {
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl std::ops::Deref for SafeManagedTensorVersioned {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice_untyped()
    }
}
