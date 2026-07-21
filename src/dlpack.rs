//! Owning handles for DLPack managed tensors.

use crate::DlpackElement;
use crate::DlpackFlags;
use crate::ManagedTensorBase;
use crate::ffi::{DLManagedTensorVersioned, DLPackVersion};
use crate::tensor;
use std::ptr::NonNull;

/// Owning RAII handle for a DLPack managed tensor pointer.
///
/// Drops by calling the DLPack managed tensor deleter. If the managed tensor
/// carries a NULL deleter (per the DLPack spec: the producer retains
/// ownership and the consumer must not free it), `Drop` is a no-op and the
/// allocation plus `manager_ctx` are *not* released — the caller that
/// constructed such a tensor is responsible for reclaiming them through their
/// original owner. `ManagedBox` therefore never calls a NULL deleter, which
/// preserves the producer-ownership contract but means drop is not always a
/// full release.
pub struct ManagedBox<M: ManagedTensorBase>(NonNull<M>);

/// A managed tensor descriptor produced and initialized locally.
///
/// This newtype preserves the invariant that the descriptor was validated
/// before construction and has not been exposed through a raw pointer since.
#[repr(transparent)]
pub struct Local<M: ManagedTensorBase>(ManagedBox<M>);

impl<M: ManagedTensorBase> Local<M> {
    pub(crate) unsafe fn from_managed(managed: ManagedBox<M>) -> Self {
        Self(managed)
    }

    /// Consumes the local tensor and transfers it through a raw pointer.
    ///
    /// A tensor reconstructed from this pointer must be treated as
    /// [`Foreign`], because external code may have changed its descriptor.
    pub fn into_raw(self) -> *mut M {
        self.0.into_raw()
    }

    /// Returns the managed tensor pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }
}

impl<M: ManagedTensorBase> std::ops::Deref for Local<M> {
    type Target = ManagedBox<M>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M: ManagedTensorBase> std::ops::DerefMut for Local<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An owning handle to a managed tensor received from external code.
///
/// Only ownership and destruction are trusted. Descriptor fields and pointers
/// remain untrusted and therefore require unsafe access.
#[repr(transparent)]
pub struct Foreign<M: ManagedTensorBase>(ManagedBox<M>);

impl<M: ManagedTensorBase> Foreign<M> {
    /// Takes ownership of a foreign managed tensor pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null and point to an initialized `M` owned by the
    /// caller. Its deleter, if present, must be valid to call exactly once and
    /// must not unwind. No validity is assumed for the embedded `DLTensor`.
    pub unsafe fn from_raw(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(|ptr| Self(ManagedBox(ptr)))
    }

    /// Returns the foreign pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }

    /// Transfers ownership of the foreign tensor through its raw pointer.
    pub fn into_raw(self) -> *mut M {
        self.0.into_raw()
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

    /// Returns the foreign shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ndim` and `shape` describe readable memory
    /// which remains immutable for the returned slice's lifetime.
    pub unsafe fn shape(&self) -> Result<&[i64], tensor::Error> {
        unsafe { self.tensor().shape() }
    }

    /// Returns the foreign explicit strides, or `None` for implicit strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ndim` and `strides` describe readable
    /// memory which remains immutable for the returned slice's lifetime.
    pub unsafe fn strides(&self) -> Result<Option<&[i64]>, tensor::Error> {
        unsafe { self.tensor().strides() }
    }
}

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
    /// The embedded `DLTensor` pointers must satisfy the DLPack contract for
    /// the descriptor's shape, strides, dtype, device, and byte offset for the
    /// entire lifetime of the managed tensor.
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
    /// The embedded `DLTensor` pointers must satisfy the DLPack contract for
    /// the descriptor's shape, strides, dtype, device, and byte offset for the
    /// entire lifetime of the managed tensor.
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

    /// Returns the embedded raw tensor descriptor.
    #[inline]
    pub fn tensor(&self) -> &crate::ffi::DLTensor {
        unsafe { self.0.as_ref() }.tensor()
    }

    /// Returns the tensor shape.
    #[inline]
    pub fn shape(&self) -> Result<&[i64], tensor::Error> {
        unsafe { self.tensor().shape() }
    }

    /// Returns explicit element strides, or `None` for an implicit compact
    /// layout.
    #[inline]
    pub fn strides(&self) -> Result<Option<&[i64]>, tensor::Error> {
        unsafe { self.tensor().strides() }
    }

    /// Returns the product of all shape dimensions.
    #[inline]
    pub fn num_elements(&self) -> Result<usize, tensor::Error> {
        unsafe { self.tensor().num_elements() }
    }

    /// Returns the logical data size in bytes, including packed sub-byte
    /// element handling.
    #[inline]
    pub fn num_bytes(&self) -> Result<usize, tensor::Error> {
        unsafe { self.tensor().num_bytes() }
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
    pub unsafe fn cpu_slice_mut_unchecked<T: DlpackElement>(
        &mut self,
    ) -> Result<&mut [T], tensor::Error> {
        if self.flags().contains(DlpackFlags::READ_ONLY) {
            return Err(tensor::Error::ReadOnly);
        }

        let tensor = self.tensor();
        if !unsafe { tensor.is_compact()? } {
            return Err(tensor::Error::NonCompactStrides);
        }
        let len = unsafe { tensor.num_elements()? };
        let data = unsafe { tensor.offset_data_ptr::<T>()? }.cast_mut();
        Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
    }

    /// Returns compact CPU tensor storage as mutable bytes, without proving exclusivity.
    ///
    /// This rejects versioned tensors carrying [`DlpackFlags::READ_ONLY`] and
    /// supports any DLPack dtype, including packed sub-byte types.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other references access the underlying
    /// data for the lifetime of the returned slice. Exclusive access to this
    /// `ManagedBox` alone does not prove that the producer has no aliases.
    pub unsafe fn cpu_bytes_mut_unchecked(&mut self) -> Result<&mut [u8], tensor::Error> {
        if self.flags().contains(DlpackFlags::READ_ONLY) {
            return Err(tensor::Error::ReadOnly);
        }

        let tensor = self.tensor();
        if !unsafe { tensor.is_compact()? } {
            return Err(tensor::Error::NonCompactStrides);
        }
        let len = unsafe { tensor.num_bytes()? };
        let data = unsafe { tensor.offset_bytes_ptr()? }.cast_mut();
        Ok(unsafe { std::slice::from_raw_parts_mut(data, len) })
    }

    /// Returns the CPU tensor data as a mutable typed slice.
    ///
    /// This rejects tensors carrying [`DlpackFlags::READ_ONLY`] and requires
    /// [`DlpackFlags::IS_COPIED`] to be set. Legacy tensors have no flags
    /// field and therefore cannot satisfy this requirement; use
    /// [`Self::cpu_slice_mut_unchecked`] when the caller can prove
    /// exclusivity independently.
    pub fn cpu_slice_mut<T: DlpackElement>(&mut self) -> Result<&mut [T], tensor::Error> {
        if !self.flags().contains(DlpackFlags::IS_COPIED) {
            return Err(tensor::Error::NotCopied);
        }

        unsafe { self.cpu_slice_mut_unchecked() }
    }

    /// Returns compact CPU tensor storage as mutable bytes.
    ///
    /// This rejects tensors carrying [`DlpackFlags::READ_ONLY`] and requires
    /// [`DlpackFlags::IS_COPIED`] to be set. Legacy tensors have no flags
    /// field and therefore cannot satisfy this requirement; use
    /// [`Self::cpu_bytes_mut_unchecked`] when the caller can prove
    /// exclusivity independently.
    pub fn cpu_bytes_mut(&mut self) -> Result<&mut [u8], tensor::Error> {
        if !self.flags().contains(DlpackFlags::IS_COPIED) {
            return Err(tensor::Error::NotCopied);
        }

        unsafe { self.cpu_bytes_mut_unchecked() }
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
    #[inline]
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

    /// Returns the ABI version declared by this managed tensor.
    #[inline]
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
        let builder = unsafe {
            Builder::new(data, metadata::CopiedArray::new([3i64], [1i64])).data(data_ptr)
        }
        .dtype(crate::ffi::DLDataType::of::<i32>());
        // Safety: the fixture data above has no other live references.
        unsafe { builder.flags_unchecked(flags) }.build::<M>()
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
        unsafe { (*raw).dl_tensor.ndim = i32::MAX };
        let foreign = unsafe { Foreign::from_raw(raw) }.expect("raw pointer is non-null");
        drop(foreign);
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
            dlpack.cpu_slice_mut_unchecked::<i32>().unwrap()[1] = 7;
        }

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 7, 3]
        );
    }

    #[test]
    fn mutable_cpu_slice_unchecked_rejects_read_only_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::READ_ONLY);

        let error = unsafe { dlpack.cpu_slice_mut_unchecked::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_unchecked_rejects_non_compact_strides() {
        let data = Box::new(vec![1i32, 2, 3, 4]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = unsafe {
            Builder::new(data, metadata::CopiedArray::new([2, 2], [1, 2])).data(data_ptr)
        }
        .dtype(crate::ffi::DLDataType::of::<i32>())
        .build::<DLManagedTensor>();

        let error = unsafe { dlpack.cpu_slice_mut_unchecked::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NonCompactStrides));
    }

    #[test]
    fn mutable_cpu_slice_updates_is_copied_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::IS_COPIED);

        dlpack.cpu_slice_mut::<i32>().unwrap()[1] = 7;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 7, 3]
        );
    }

    #[test]
    fn mutable_cpu_slice_rejects_tensor_without_is_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());

        let error = dlpack.cpu_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::NotCopied));
    }

    #[test]
    fn mutable_cpu_slice_rejects_read_only_tensor_even_if_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(
            DlpackFlags::READ_ONLY | DlpackFlags::IS_COPIED,
        );

        let error = dlpack.cpu_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_rejects_legacy_tensor_as_never_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());

        let error = dlpack.cpu_slice_mut::<i32>().unwrap_err();

        assert!(matches!(error, tensor::Error::NotCopied));
    }

    #[test]
    fn mutable_cpu_bytes_updates_is_copied_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::IS_COPIED);

        dlpack.cpu_bytes_mut().unwrap()[..size_of::<i32>()].copy_from_slice(&7i32.to_ne_bytes());

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[7, 2, 3]
        );
    }

    #[test]
    fn mutable_cpu_bytes_rejects_tensor_without_is_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());

        let error = dlpack.cpu_bytes_mut().unwrap_err();

        assert!(matches!(error, tensor::Error::NotCopied));
    }

    #[test]
    fn mutable_cpu_bytes_rejects_read_only_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY,
        );

        let error = dlpack.cpu_bytes_mut().unwrap_err();

        assert!(matches!(error, tensor::Error::ReadOnly));
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
