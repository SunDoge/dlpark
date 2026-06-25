use super::manager_context::{TensorLikeContext, into_dlpack};
use crate::ffi;
use crate::traits::{TensorLike, TensorView};
use std::ptr::NonNull;

/// A safe wrapper around a legacy DLPack tensor that manages its memory lifecycle.
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
    /// Creates a new `SafeManagedTensor` from a tensor-like type.
    pub fn new<T>(t: T) -> std::result::Result<Self, T::Error>
    where
        T: TensorLike,
    {
        let ctx = TensorLikeContext::new(t)?;
        Ok(Self(into_dlpack(ctx)))
    }

    /// Creates from a raw pointer (transfers ownership).
    ///
    /// # Safety
    /// The pointer must be valid, non-null, and point to a properly initialized
    /// `ManagedTensor` with a valid deleter.
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensor) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    /// Creates from a `NonNull` pointer (transfers ownership).
    ///
    /// # Safety
    /// The pointer must be valid and point to a properly initialized
    /// `ManagedTensor` with a valid deleter.
    pub unsafe fn from_non_null(ptr: ffi::Dlpack) -> Self {
        Self(ptr)
    }

    /// Converts into a raw pointer, transferring ownership to the caller.
    pub fn into_raw(self) -> *mut ffi::ManagedTensor {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Converts into a `NonNull` pointer, transferring ownership to the caller.
    pub fn into_non_null(self) -> ffi::Dlpack {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }
}

impl TensorView for SafeManagedTensor {
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl AsRef<SafeManagedTensor> for SafeManagedTensor {
    fn as_ref(&self) -> &Self { self }
}