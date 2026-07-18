use crate::DlpackElement;
use crate::DlpackFlags;
use crate::ManagedTensorBase;
use crate::ffi::{DLManagedTensorVersioned, DLPackVersion};
use crate::tensor;
use std::ptr::NonNull;

/// Owning RAII handle for a DLPack managed tensor pointer.
///
/// Drops by calling the DLPack managed tensor deleter.
pub struct ManagedBox<M: ManagedTensorBase>(NonNull<M>);

impl<M> ManagedBox<M>
where
    M: ManagedTensorBase,
{
    /// Creates an owning managed tensor handle from a raw pointer.
    ///
    /// # Safety
    ///
    /// If `ptr` is non-null, it must point to a valid `M` whose ownership is
    /// transferred to the returned `ManagedBox`. The managed tensor must not
    /// have been freed or wrapped by another owner, and its deleter, if
    /// present, must be valid to call exactly once and must not unwind.
    pub unsafe fn new(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(ManagedBox)
    }

    /// Create a new `ManagedBox` from a raw pointer without checking if it is null.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to a valid `M` whose ownership is
    /// transferred to the returned `ManagedBox`. The managed tensor must not
    /// have been freed or wrapped by another owner, and its deleter, if
    /// present, must be valid to call exactly once and must not unwind.
    pub unsafe fn new_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
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

    pub fn tensor(&self) -> &crate::ffi::DLTensor {
        unsafe { self.0.as_ref() }.tensor()
    }

    pub fn shape(&self) -> Result<&[i64], tensor::Error> {
        self.tensor().shape()
    }

    pub fn strides(&self) -> Result<Option<&[i64]>, tensor::Error> {
        self.tensor().strides()
    }

    pub fn num_elements(&self) -> Result<usize, tensor::Error> {
        self.tensor().num_elements()
    }

    pub fn num_bytes(&self) -> Result<usize, tensor::Error> {
        self.tensor().num_bytes()
    }

    pub fn cpu_data_slice<T: DlpackElement>(&self) -> Result<&[T], tensor::Error> {
        self.tensor().cpu_data_slice()
    }

    /// Returns the CPU tensor data as a mutable typed slice.
    ///
    /// This rejects versioned tensors carrying [`DlpackFlags::READ_ONLY`].
    /// Legacy tensors have no flags and are treated as writable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other references access the underlying
    /// data for the lifetime of the returned slice. Exclusive access to this
    /// `ManagedBox` alone does not prove that the producer has no aliases.
    pub unsafe fn cpu_data_slice_mut<T: DlpackElement>(
        &mut self,
    ) -> Result<&mut [T], tensor::Error> {
        if self.flags().contains(DlpackFlags::READ_ONLY) {
            return Err(tensor::Error::ReadOnly);
        }

        let tensor = self.tensor();
        if !tensor.is_compact()? {
            return Err(tensor::Error::NonCompactStrides);
        }
        let len = tensor.num_elements()?;
        let data = tensor.cpu_data_ptr::<T>()?.cast_mut();
        Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
    }
}

impl<M> std::ops::Deref for ManagedBox<M>
where
    M: ManagedTensorBase,
{
    type Target = M;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
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
    use crate::{builder::Builder, ffi::DLManagedTensor, metadata};
    use std::ffi::c_void;

    #[test]
    fn versioned_flags_roundtrip_through_builder() {
        let data = Box::new(vec![1i32, 2, 3]);
        let dlpack = Builder::new(data, metadata::CopiedArray::new([3i64], [1i64]))
            .flags(DlpackFlags::READ_ONLY)
            .build::<DLManagedTensorVersioned>();

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
    }

    #[test]
    fn versioned_flags_default_to_empty() {
        let data = Box::new(vec![1i32, 2, 3]);
        let dlpack = Builder::new(data, metadata::CopiedArray::new([3i64], [1i64]))
            .build::<DLManagedTensorVersioned>();

        assert_eq!(dlpack.flags(), DlpackFlags::empty());
    }

    #[test]
    fn mutable_cpu_slice_updates_writable_tensor() {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = Builder::new(data, metadata::CopiedArray::new([3i64], [1i64]))
            .data(data_ptr)
            .dtype(crate::ffi::DLDataType::of::<i32>())
            .build::<DLManagedTensor>();

        unsafe {
            dlpack.cpu_data_slice_mut::<i32>().unwrap()[1] = 7;
        }

        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 7, 3]);
    }

    #[test]
    fn mutable_cpu_slice_rejects_read_only_tensor() {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = Builder::new(data, metadata::CopiedArray::new([3i64], [1i64]))
            .data(data_ptr)
            .dtype(crate::ffi::DLDataType::of::<i32>())
            .flags(DlpackFlags::READ_ONLY)
            .build::<DLManagedTensorVersioned>();

        let error = unsafe { dlpack.cpu_data_slice_mut::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_rejects_non_compact_strides() {
        let data = Box::new(vec![1i32, 2, 3, 4]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = Builder::new(data, metadata::CopiedArray::new([2, 2], [1, 2]))
            .data(data_ptr)
            .dtype(crate::ffi::DLDataType::of::<i32>())
            .build::<DLManagedTensor>();

        let error = unsafe { dlpack.cpu_data_slice_mut::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NonCompactStrides));
    }
}
