//! Externally supplied DLPack managed tensors.

use super::local::Local;
use crate::{ManagedTensorBase, tensor};
use std::ptr::NonNull;

/// An owning handle to a managed tensor received from external code.
///
/// Only ownership and destruction are trusted. Descriptor fields and pointers
/// remain untrusted and therefore require unsafe access.
#[repr(transparent)]
pub struct Foreign<M: ManagedTensorBase>(NonNull<M>);

impl<M: ManagedTensorBase> Foreign<M> {
    pub(crate) unsafe fn from_raw_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Takes ownership of a foreign managed tensor pointer.
    ///
    /// # Safety
    ///
    /// If non-null, `ptr` must point to an initialized `M` owned by the caller.
    /// Its deleter, if present, must be valid to call exactly once and must not
    /// unwind. No validity is assumed for the embedded `DLTensor`.
    pub unsafe fn from_raw(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    /// Returns the foreign pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }

    /// Transfers ownership of the foreign tensor through its raw pointer.
    pub fn into_raw(self) -> *mut M {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Treats this externally supplied tensor as a validated local tensor.
    ///
    /// # Safety
    ///
    /// The managed tensor and every pointer in its embedded descriptor must
    /// satisfy the DLPack contract for the remainder of its lifetime. The
    /// descriptor must not be concurrently mutated through another alias.
    pub unsafe fn into_local(self) -> Local<M> {
        unsafe { Local::from_raw_unchecked(self.into_raw()) }
    }

    /// Returns the untrusted embedded descriptor.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `M` and its embedded descriptor are readable
    /// and are not concurrently mutated for the returned reference's lifetime.
    pub unsafe fn tensor(&self) -> &crate::ffi::DLTensor {
        unsafe { (&*self.0.as_ptr()).tensor() }
    }

    /// Returns the DLPack bitmask flags.
    pub fn flags(&self) -> crate::DlpackFlags {
        unsafe { (&*self.0.as_ptr()).flags() }
    }

    /// Returns the foreign shape.
    ///
    /// # Safety
    ///
    /// `ndim` and `shape` must describe readable immutable memory for the
    /// returned slice's lifetime.
    pub unsafe fn shape(&self) -> Result<&[i64], tensor::Error> {
        unsafe { self.tensor().shape() }
    }

    /// Returns explicit foreign strides, or `None` for implicit strides.
    ///
    /// # Safety
    ///
    /// `ndim` and `strides` must describe readable immutable memory for the
    /// returned slice's lifetime.
    pub unsafe fn strides(&self) -> Result<Option<&[i64]>, tensor::Error> {
        unsafe { self.tensor().strides() }
    }
}

impl<M: ManagedTensorBase> Drop for Foreign<M> {
    fn drop(&mut self) {
        unsafe { M::drop_raw(self.0.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManagedTensorBase, ffi::DLManagedTensor};

    #[test]
    fn local_raw_roundtrip_becomes_foreign() {
        let allocation = crate::allocation::dynamic::Allocation::<DLManagedTensor>::allocate(0)
            .expect("allocation must succeed");
        let initialized = allocation
            .initialize(Box::new(()), 0)
            .expect("scalar rank must fit");
        let local = unsafe { initialized.finish() };

        let raw = local.into_raw();
        unsafe { (*raw).tensor_mut().ndim = i32::MAX };
        let foreign = unsafe { Foreign::from_raw(raw) }.expect("raw pointer is non-null");
        drop(foreign);
    }
}
