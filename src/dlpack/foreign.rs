//! Externally supplied DLPack managed tensors.

use crate::{ManagedTensorBase, tensor};
use snafu::Snafu;
use std::{borrow::Cow, ptr::NonNull};

#[derive(Debug, Snafu)]
pub enum FromRawError {
    #[snafu(display("managed tensor pointer is null"))]
    Null,

    #[snafu(transparent)]
    Version {
        source: crate::version::VersionError,
    },
}

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
    /// `ptr` must point to an initialized `M` owned by the caller.
    /// Its deleter, if present, must be valid to call exactly once and must not
    /// unwind. No validity is assumed for the embedded `DLTensor`. An
    /// incompatible versioned tensor is released through its deleter before
    /// this function returns an error.
    pub unsafe fn from_raw(ptr: *mut M) -> Result<Self, FromRawError> {
        let ptr = NonNull::new(ptr).ok_or(FromRawError::Null)?;
        let foreign = Self(ptr);
        if let Some(version) = unsafe { ptr.as_ref() }.version() {
            crate::version::validate_version(version)?;
        }
        Ok(foreign)
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

    pub fn device(&self) -> crate::ffi::DLDevice {
        unsafe { self.tensor() }.device
    }

    pub fn dtype(&self) -> crate::ffi::DLDataType {
        unsafe { self.tensor() }.dtype
    }

    pub fn ndim(&self) -> i32 {
        unsafe { self.tensor() }.ndim
    }

    pub fn byte_offset(&self) -> u64 {
        unsafe { self.tensor() }.byte_offset
    }

    pub fn version(&self) -> Option<crate::ffi::DLPackVersion> {
        unsafe { self.0.as_ref() }.version()
    }

    /// Returns explicit strides or computes compact strides when implicit.
    ///
    /// # Safety
    ///
    /// The foreign shape and strides pointers must reference readable metadata
    /// for the returned value's lifetime.
    pub unsafe fn strides_or_compact(&self) -> Result<Cow<'_, [i64]>, tensor::Error> {
        unsafe { self.tensor().strides_or_compact() }
    }

    /// Returns the logical element count.
    ///
    /// # Safety
    ///
    /// The foreign shape pointer must reference readable metadata.
    pub unsafe fn num_elements(&self) -> Result<usize, tensor::Error> {
        unsafe { self.tensor().num_elements() }
    }

    /// Returns the logical byte count.
    ///
    /// # Safety
    ///
    /// The foreign shape pointer must reference readable metadata.
    pub unsafe fn num_bytes(&self) -> Result<usize, tensor::Error> {
        unsafe { self.tensor().num_bytes() }
    }

    /// Returns whether the foreign layout is compact row-major.
    ///
    /// # Safety
    ///
    /// The foreign shape and strides pointers must reference readable metadata.
    pub unsafe fn is_compact(&self) -> Result<bool, tensor::Error> {
        unsafe { self.tensor().is_compact() }
    }

    /// Returns the byte-offset-adjusted typed data pointer.
    ///
    /// # Safety
    ///
    /// Shape metadata must be readable and the adjusted address must lie in
    /// the foreign device allocation when the tensor is non-empty.
    pub unsafe fn offset_data_ptr<T: crate::DlpackElement>(
        &self,
    ) -> Result<*const T, tensor::Error> {
        unsafe { self.tensor().offset_data_ptr::<T>() }
    }

    /// Returns the byte-offset-adjusted untyped data pointer.
    ///
    /// # Safety
    ///
    /// Shape metadata must be readable and the adjusted address must lie in
    /// the foreign device allocation when the tensor is non-empty.
    pub unsafe fn offset_bytes_ptr(&self) -> Result<*const u8, tensor::Error> {
        unsafe { self.tensor().offset_bytes_ptr() }
    }

    /// Borrows compact foreign CPU data as typed elements.
    ///
    /// # Safety
    ///
    /// All descriptor pointers must be readable, and the adjusted CPU data
    /// pointer must reference the reported number of initialized `T` values
    /// for the returned slice's lifetime.
    pub unsafe fn cpu_slice<T: crate::DlpackElement>(&self) -> Result<&[T], tensor::Error> {
        unsafe { self.tensor().cpu_slice::<T>() }
    }

    /// Borrows compact foreign CPU data as bytes.
    ///
    /// # Safety
    ///
    /// All descriptor pointers must be readable, and the adjusted CPU data
    /// pointer must reference the reported number of initialized bytes for the
    /// returned slice's lifetime.
    pub unsafe fn cpu_bytes(&self) -> Result<&[u8], tensor::Error> {
        unsafe { self.tensor().cpu_bytes() }
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
    use crate::{
        ManagedTensorBase,
        ffi::{DLManagedTensor, DLManagedTensorVersioned},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

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
        let foreign = unsafe { Foreign::from_raw(raw) }.expect("raw pointer is valid");
        drop(foreign);
    }

    #[test]
    fn incompatible_version_is_rejected_and_dropped() {
        let drops = Arc::new(AtomicUsize::new(0));
        let allocation =
            crate::allocation::dynamic::Allocation::<DLManagedTensorVersioned>::allocate(0)
                .unwrap();
        let initialized = allocation
            .initialize(Box::new(DropCounter(Arc::clone(&drops))), 0)
            .unwrap();
        let local = unsafe { initialized.finish() };
        let raw = local.into_raw();
        unsafe { (*raw).version.major = crate::ffi::DLPACK_MAJOR_VERSION + 1 };

        let error = match unsafe { Foreign::from_raw(raw) } {
            Ok(_) => panic!("incompatible version must be rejected"),
            Err(error) => error,
        };

        assert!(matches!(error, FromRawError::Version { .. }));
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
