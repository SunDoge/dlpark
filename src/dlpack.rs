use crate::DlpackFlags;
use crate::ManagedTensorBase;
use crate::ffi::{DLManagedTensorVersioned, DLPackVersion, DLTensor};
use std::ptr::NonNull;

/// Owning RAII handle for a DLPack managed tensor pointer.
///
/// Drops by calling the DLPack managed tensor deleter.
pub struct ManagedBox<M: ManagedTensorBase>(NonNull<M>);

impl<M> ManagedBox<M>
where
    M: ManagedTensorBase,
{
    /// # Safety
    ///
    /// TODO
    pub unsafe fn new(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(ManagedBox)
    }

    /// Create a new `ManagedBox` from a raw pointer without checking if it is null.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is not null and points to a valid `M`.
    pub unsafe fn new_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    pub fn dl_tensor(&self) -> &DLTensor {
        unsafe { self.0.as_ref() }.dl_tensor()
    }

    pub fn dl_tensor_mut(&mut self) -> &mut DLTensor {
        unsafe { self.0.as_mut() }.dl_tensor_mut()
    }

    /// Returns the shape of the tensor as a slice.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::shape`] for error conditions.
    pub fn shape(&self) -> Result<&[i64], crate::tensor::Error> {
        self.dl_tensor().shape()
    }

    pub fn shape_mut(&mut self) {}

    /// Returns the strides of the tensor as a slice, or `None` for compact row-major layout.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::strides`] for error conditions.
    pub fn strides(&self) -> Result<Option<&[i64]>, crate::tensor::Error> {
        self.dl_tensor().strides()
    }

    /// Returns the total number of elements in the tensor (product of all shape dimensions).
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::num_elements`] for error conditions.
    pub fn num_elements(&self) -> Result<usize, crate::tensor::Error> {
        self.dl_tensor().num_elements()
    }

    /// Returns the total size of the tensor data in bytes.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::num_bytes`] for error conditions.
    pub fn num_bytes(&self) -> Result<usize, crate::tensor::Error> {
        self.dl_tensor().num_bytes()
    }

    /// Consumes the `ManagedBox`, returning the wrapped raw pointer.
    ///
    /// The caller takes ownership of the managed tensor and is responsible for calling the FFI deleter later.
    pub fn into_raw(self) -> *mut M {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Returns the wrapped raw pointer without consuming the `ManagedBox`.
    ///
    /// The `ManagedBox` still owns the managed tensor and will call its deleter on drop.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }
}

impl ManagedBox<DLManagedTensorVersioned> {
    /// Returns the DLPack bitmask flags (e.g. `READ_ONLY`, `IS_COPIED`).
    ///
    /// Only present on the versioned tensor ABI; the legacy `DLManagedTensor`
    /// has no `flags` field.
    pub fn flags(&self) -> DlpackFlags {
        unsafe { self.0.as_ref() }.flags
    }

    pub fn flags_mut(&mut self) -> &mut DlpackFlags {
        &mut unsafe { self.0.as_mut() }.flags
    }

    pub fn version(&self) -> DLPackVersion {
        unsafe { self.0.as_ref() }.version
    }

    pub fn version_mut(&mut self) -> &mut DLPackVersion {
        &mut unsafe { self.0.as_mut() }.version
    }
}

impl<M> Drop for ManagedBox<M>
where
    M: ManagedTensorBase,
{
    fn drop(&mut self) {
        unsafe {
            M::drop_raw(self.0.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;

    #[test]
    fn versioned_flags_roundtrip_through_builder() {
        let data = Box::new(vec![1i32, 2, 3]);
        let dlpack =
            Builder::<DLManagedTensorVersioned, 1>::with_array_layout(data, &[3i64], &[1i64])
                .flags(DlpackFlags::READ_ONLY)
                .build();

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
    }

    #[test]
    fn versioned_flags_default_to_empty() {
        let data = Box::new(vec![1i32, 2, 3]);
        let dlpack =
            Builder::<DLManagedTensorVersioned, 1>::with_array_layout(data, &[3i64], &[1i64])
                .build();

        assert_eq!(dlpack.flags(), DlpackFlags::empty());
    }
}
