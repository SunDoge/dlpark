use std::ptr::NonNull;

use crate::{
    data_type::DataType,
    device::Device,
    error::Result,
    managed_tensor_versioned::{Flags, IntoDlpackVersioned, ManagedTensorVersioned},
    manager_context::TensorLike,
    manager_context_versioned::ManagerContextVersioned,
    memory_layout::{BorrowedLayout, MemoryLayout},
};

pub struct SafeManagedTensorVersioned(NonNull<ManagedTensorVersioned>);

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
    pub unsafe fn from_raw(ptr: *mut ManagedTensorVersioned) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    pub unsafe fn from_non_null(ptr: NonNull<ManagedTensorVersioned>) -> Self {
        Self(ptr)
    }

    pub unsafe fn into_raw(self) -> *mut ManagedTensorVersioned {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    pub fn into_non_null(self) -> NonNull<ManagedTensorVersioned> {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    pub fn with_flags<T, L>(t: T, flags: Flags) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        let ctx = ManagerContextVersioned::new(t);
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
}
