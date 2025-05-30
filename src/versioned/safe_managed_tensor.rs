use std::ptr::NonNull;

use crate::ffi::{self, DataType, Device, Flags};
use crate::{
    error::Result,
    traits::{MemoryLayout, TensorLike},
};

use super::ManagerContext;

pub struct SafeManagedTensor(ffi::DlpackVersioned);

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

    pub fn into_non_null(self) -> NonNull<ffi::ManagedTensorVersioned> {
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

    pub fn shape(&self) -> &[i64] {
        unsafe { self.0.as_ref().dl_tensor.get_shape() }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        unsafe { self.0.as_ref().dl_tensor.get_strides() }
    }

    pub fn data_type(&self) -> &DataType {
        unsafe { &self.0.as_ref().dl_tensor.dtype }
    }

    pub fn device(&self) -> &Device {
        unsafe { &self.0.as_ref().dl_tensor.device }
    }

    pub fn as_slice_untyped(&self) -> &[u8] {
        unsafe { self.0.as_ref().dl_tensor.as_slice_untyped() }
    }

    pub unsafe fn as_slice<A>(&self) -> Result<&[A]> {
        unsafe { self.0.as_ref().dl_tensor.as_slice() }
    }

    pub fn is_contiguous(&self) -> bool {
        unsafe { self.0.as_ref().dl_tensor.is_contiguous() }
    }
}
