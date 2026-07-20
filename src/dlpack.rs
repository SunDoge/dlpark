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

    /// Returns the CPU tensor data as a mutable typed slice, without proving exclusivity.
    ///
    /// This rejects versioned tensors carrying [`DlpackFlags::READ_ONLY`].
    /// Legacy tensors have no flags and are treated as writable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other references access the underlying
    /// data for the lifetime of the returned slice. Exclusive access to this
    /// `ManagedBox` alone does not prove that the producer has no aliases.
    pub unsafe fn cpu_data_slice_mut_unchecked<T: DlpackElement>(
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

    /// Returns the CPU tensor data as a mutable typed slice.
    ///
    /// This rejects tensors carrying [`DlpackFlags::READ_ONLY`] and requires
    /// [`DlpackFlags::IS_COPIED`] to be set. Legacy tensors have no flags
    /// field and therefore cannot satisfy this requirement; use
    /// [`Self::cpu_data_slice_mut_unchecked`] when the caller can prove
    /// exclusivity independently.
    pub fn cpu_data_slice_mut<T: DlpackElement>(&mut self) -> Result<&mut [T], tensor::Error> {
        if !self.flags().contains(DlpackFlags::IS_COPIED) {
            return Err(tensor::Error::NotCopied);
        }

        unsafe { self.cpu_data_slice_mut_unchecked() }
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

    /// Returns mutable access to the DLPack bitmask flags.
    ///
    /// # Safety
    ///
    /// The caller must preserve the producer's mutability and ownership
    /// guarantees. In particular, setting [`DlpackFlags::IS_COPIED`] asserts
    /// that no other reference to the tensor data exists, while clearing
    /// [`DlpackFlags::READ_ONLY`] asserts that consumers may modify it.
    pub unsafe fn flags_mut(&mut self) -> &mut DlpackFlags {
        &mut unsafe { self.0.as_mut() }.flags
    }

    pub fn version(&self) -> DLPackVersion {
        unsafe { self.0.as_ref() }.version
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

    /// Builds a `[1, 2, 3]` i32 tensor of type `M` with the given flags.
    ///
    /// `flags` is a no-op for `M = DLManagedTensor`, which has no flags field.
    fn dlpack_with_flags<M: ManagedTensorBase>(flags: DlpackFlags) -> ManagedBox<M> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let builder = Builder::new(data, metadata::CopiedArray::new([3i64], [1i64]))
            .data(data_ptr)
            .dtype(crate::ffi::DLDataType::of::<i32>());
        // Safety: the fixture data above has no other live references.
        unsafe { builder.flags_unchecked(flags) }.build::<M>()
    }

    #[test]
    fn versioned_flags_roundtrip_through_builder() {
        let dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::READ_ONLY);

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
    }

    #[test]
    fn versioned_flags_default_to_empty() {
        let dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());

        assert_eq!(dlpack.flags(), DlpackFlags::empty());
    }

    #[test]
    fn mutable_cpu_slice_unchecked_updates_writable_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());

        unsafe {
            dlpack.cpu_data_slice_mut_unchecked::<i32>().unwrap()[1] = 7;
        }

        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 7, 3]);
    }

    #[test]
    fn mutable_cpu_slice_unchecked_rejects_read_only_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::READ_ONLY);

        let error = unsafe { dlpack.cpu_data_slice_mut_unchecked::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_unchecked_rejects_non_compact_strides() {
        let data = Box::new(vec![1i32, 2, 3, 4]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = Builder::new(data, metadata::CopiedArray::new([2, 2], [1, 2]))
            .data(data_ptr)
            .dtype(crate::ffi::DLDataType::of::<i32>())
            .build::<DLManagedTensor>();

        let error = unsafe { dlpack.cpu_data_slice_mut_unchecked::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NonCompactStrides));
    }

    #[test]
    fn mutable_cpu_slice_updates_is_copied_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::IS_COPIED);

        dlpack.cpu_data_slice_mut::<i32>().unwrap()[1] = 7;

        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 7, 3]);
    }

    #[test]
    fn mutable_cpu_slice_rejects_tensor_without_is_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());

        let error = dlpack.cpu_data_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::NotCopied));
    }

    #[test]
    fn mutable_cpu_slice_rejects_read_only_tensor_even_if_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(
            DlpackFlags::READ_ONLY | DlpackFlags::IS_COPIED,
        );

        let error = dlpack.cpu_data_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_rejects_legacy_tensor_as_never_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());

        let error = dlpack.cpu_data_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::NotCopied));
    }

    #[test]
    fn flags_mut_updates_versioned_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());

        unsafe {
            *dlpack.flags_mut() |= DlpackFlags::READ_ONLY;
        }

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
    }
}
