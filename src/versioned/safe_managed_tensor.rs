use std::ptr::NonNull;

use crate::ffi::{self, Flags};
use crate::traits::{TensorLike, TensorView};

use super::manager_context::{TensorLikeContext, into_dlpack_versioned};

/// A safe wrapper around a versioned DLPack tensor that manages its memory lifecycle.
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
    /// Creates a new `SafeManagedTensorVersioned` from a tensor-like type.
    pub fn new<T>(t: T) -> std::result::Result<Self, T::Error>
    where
        T: TensorLike,
    {
        Self::with_flags(t, Flags::empty())
    }

    /// Creates a new `SafeManagedTensorVersioned` with explicit flags.
    pub fn with_flags<T>(t: T, flags: Flags) -> std::result::Result<Self, T::Error>
    where
        T: TensorLike,
    {
        let ctx = TensorLikeContext::new(t)?;
        Ok(Self(into_dlpack_versioned(ctx, flags)))
    }

    /// Creates from a raw pointer (transfers ownership).
    ///
    /// # Safety
    /// The pointer must be valid, non-null, and point to a properly initialized
    /// `ManagedTensorVersioned` with a valid deleter.
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensorVersioned) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    /// Creates from a `NonNull` pointer (transfers ownership).
    ///
    /// # Safety
    /// The pointer must be valid and point to a properly initialized
    /// `ManagedTensorVersioned` with a valid deleter.
    pub unsafe fn from_non_null(ptr: ffi::DlpackVersioned) -> Self {
        Self(ptr)
    }

    /// Converts into a raw pointer, transferring ownership to the caller.
    pub fn into_raw(self) -> *mut ffi::ManagedTensorVersioned {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Converts into a `NonNull` pointer, transferring ownership to the caller.
    pub fn into_non_null(self) -> ffi::DlpackVersioned {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    pub fn flags(&self) -> &Flags {
        unsafe { &self.0.as_ref().flags }
    }
    pub fn read_only(&self) -> bool { self.flags().contains(Flags::READ_ONLY) }
    pub fn is_copied(&self) -> bool { self.flags().contains(Flags::IS_COPIED) }
    pub fn is_subbyte_type_padded(&self) -> bool { self.flags().contains(Flags::IS_SUBBYTE_TYPE_PADDED) }
}

impl TensorView for SafeManagedTensorVersioned {
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl AsRef<SafeManagedTensorVersioned> for SafeManagedTensorVersioned {
    fn as_ref(&self) -> &Self { self }
}