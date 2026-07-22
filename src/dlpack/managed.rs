//! Owning DLPack managed tensors.

use crate::DlpackFlags;
use crate::ManagedTensorBase;
use crate::ffi::{DLManagedTensorVersioned, DLPackVersion};
use crate::tensor;
use snafu::Snafu;
use std::ptr::NonNull;

#[derive(Debug, Snafu)]
pub enum FromRawError {
    #[snafu(display("managed tensor pointer is null"))]
    Null,

    #[snafu(transparent)]
    Version { source: crate::VersionError },
}

/// An owning handle to a DLPack managed tensor.
///
/// Drops by calling the DLPack managed tensor deleter. If the managed tensor
/// carries a NULL deleter (per the DLPack spec: the producer retains
/// ownership and the consumer must not free it), `Drop` is a no-op and the
/// allocation plus `manager_ctx` are *not* released — the caller that
/// constructed such a tensor is responsible for reclaiming them through their
/// original owner. `Managed` therefore never calls a NULL deleter, which
/// preserves the producer-ownership contract but means drop is not always a
/// full release.
#[repr(transparent)]
pub struct Managed<M: ManagedTensorBase>(NonNull<M>);

impl<M: ManagedTensorBase> Managed<M> {
    pub(crate) unsafe fn from_raw_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Takes ownership of a managed tensor pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to an initialized `M` owned by the caller. Its shape and
    /// optional strides storage must remain readable and immutable while this
    /// handle exists. Its deleter, if present, must be valid to call exactly
    /// once and must not unwind. An incompatible versioned tensor is released
    /// before returning an error.
    pub unsafe fn from_raw(ptr: *mut M) -> Result<Self, FromRawError> {
        let ptr = NonNull::new(ptr).ok_or(FromRawError::Null)?;
        let managed = Self(ptr);
        if let Some(version) = unsafe { ptr.as_ref() }.version() {
            crate::version::validate_version(version)?;
        }
        Ok(managed)
    }

    /// Consumes the managed tensor and transfers it through a raw pointer.
    pub fn into_raw(self) -> *mut M {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Returns the managed tensor pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }
}

impl<M> Managed<M>
where
    M: ManagedTensorBase,
{
    /// Returns the embedded raw tensor descriptor without validating it.
    ///
    /// # Safety
    ///
    /// The descriptor must be readable and not concurrently mutated for the
    /// returned reference's lifetime.
    #[inline]
    pub unsafe fn tensor(&self) -> &crate::ffi::DLTensor {
        unsafe { self.0.as_ref() }.tensor()
    }

    /// Validates the descriptor metadata and returns a safe metadata view.
    ///
    pub fn validate(&self) -> Result<tensor::TensorRef<'_>, tensor::Error> {
        unsafe { tensor::TensorRef::from_raw(self.tensor()) }
    }

    /// Validates the descriptor for mutable access.
    ///
    /// `READ_ONLY` tensors are rejected. `IS_COPIED` is exchange metadata and
    /// does not affect Rust mutable-access validation.
    pub fn validate_mut(&mut self) -> Result<tensor::TensorMut<'_>, tensor::Error> {
        let flags = unsafe { self.0.as_ref() }.flags();
        let tensor = unsafe { self.0.as_mut() }.tensor_mut();
        unsafe { tensor::TensorMut::from_raw(tensor, flags) }
    }
}

impl<M> std::ops::Deref for Managed<M>
where
    M: ManagedTensorBase,
{
    type Target = M;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl Managed<DLManagedTensorVersioned> {
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

impl<M> Drop for Managed<M>
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
    use crate::{
        Managed,
        allocation::fixed::make_test_tensor,
        ffi::{DLDevice, DLManagedTensor},
    };
    use std::ffi::c_void;

    /// Builds a `[1, 2, 3]` i32 tensor of type `M` with the given flags.
    ///
    /// `flags` is a no-op for `M = DLManagedTensor`, which has no flags field.
    fn dlpack_with_flags<M: ManagedTensorBase>(flags: DlpackFlags) -> Managed<M> {
        dlpack_with_flags_on_device(flags, DLDevice::CPU)
    }

    fn dlpack_with_flags_on_device<M: ManagedTensorBase>(
        flags: DlpackFlags,
        device: DLDevice,
    ) -> Managed<M> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        make_test_tensor(
            data,
            data_ptr,
            crate::ffi::DLDataType::of::<i32>(),
            device,
            [3],
            [1],
            flags,
        )
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
    fn from_raw_rejects_null() {
        let error = match unsafe { Managed::<DLManagedTensor>::from_raw(std::ptr::null_mut()) } {
            Ok(_) => panic!("null pointer must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, FromRawError::Null));
    }

    #[test]
    fn from_raw_rejects_and_drops_incompatible_version() {
        let dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let raw = dlpack.into_raw();
        unsafe { (*raw).version.major = crate::ffi::DLPACK_MAJOR_VERSION + 1 };

        let error = match unsafe { Managed::from_raw(raw) } {
            Ok(_) => panic!("incompatible version must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, FromRawError::Version { .. }));
    }

    #[test]
    fn mutable_cpu_slice_updates_writable_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());
        {
            let mut tensor = dlpack.validate_mut().unwrap();
            unsafe { tensor.cpu_slice_mut::<i32>() }.unwrap()[1] = 7;
        }

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 7, 3]
        );
    }

    #[test]
    fn validate_exposes_metadata() {
        let dlpack = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());
        let tensor = dlpack.validate().unwrap();

        assert_eq!(unsafe { tensor.cpu_slice::<i32>() }.unwrap(), &[1, 2, 3]);
        assert_eq!(
            unsafe { tensor.cpu_bytes() }.unwrap().len(),
            3 * size_of::<i32>()
        );
        assert_eq!(tensor.device().device_type, DLDevice::CPU.device_type);
        assert_eq!(tensor.device().device_id, 0);
        let dtype = tensor.dtype();
        let expected_dtype = crate::ffi::DLDataType::of::<i32>();
        assert_eq!(dtype.code, expected_dtype.code);
        assert_eq!(dtype.bits, expected_dtype.bits);
        assert_eq!(dtype.lanes, expected_dtype.lanes);
        assert_eq!(tensor.byte_offset(), 0);
        assert!(tensor.is_compact().unwrap());
        assert_eq!(&*tensor.strides_or_compact().unwrap(), &[1]);
    }

    #[test]
    fn validate_mut_rejects_read_only_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::READ_ONLY);
        let error = match dlpack.validate_mut() {
            Ok(_) => panic!("read-only tensor must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, tensor::Error::ReadOnly));
    }

    #[test]
    fn mutable_cpu_slice_rejects_non_compact_strides() {
        let data = Box::new(vec![1i32, 2, 3, 4]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let mut dlpack = make_test_tensor::<_, DLManagedTensor, 2>(
            data,
            data_ptr,
            crate::ffi::DLDataType::of::<i32>(),
            DLDevice::CPU,
            [2, 2],
            [1, 2],
            DlpackFlags::empty(),
        );

        let mut tensor = dlpack.validate_mut().unwrap();
        let error = unsafe { tensor.cpu_slice_mut::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NonCompactStrides));
    }

    #[test]
    fn is_copied_does_not_change_mutable_validation() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::IS_COPIED);
        {
            let mut tensor = dlpack.validate_mut().unwrap();
            unsafe { tensor.cpu_slice_mut::<i32>() }.unwrap()[1] = 7;
        }

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 7, 3]
        );
    }

    #[test]
    fn mutable_cpu_slice_accepts_tensor_without_is_copied() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let mut tensor = dlpack.validate_mut().unwrap();
        unsafe { tensor.cpu_slice_mut::<i32>() }.unwrap()[0] = 9;
        assert_eq!(unsafe { tensor.cpu_slice::<i32>() }.unwrap(), &[9, 2, 3]);
    }

    #[test]
    fn mutable_cpu_slice_rejects_non_cpu_tensor() {
        let mut dlpack = dlpack_with_flags_on_device::<DLManagedTensorVersioned>(
            DlpackFlags::empty(),
            DLDevice::cuda(0),
        );
        let mut tensor = dlpack.validate_mut().unwrap();
        let error = unsafe { tensor.cpu_slice_mut::<i32>() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NotCpu { .. }));
    }

    #[test]
    fn mutable_cpu_bytes_updates_writable_tensor() {
        let mut dlpack = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let mut tensor = dlpack.validate_mut().unwrap();
        unsafe { tensor.cpu_bytes_mut() }.unwrap()[..size_of::<i32>()]
            .copy_from_slice(&7i32.to_ne_bytes());
        assert_eq!(unsafe { tensor.cpu_slice::<i32>() }.unwrap(), &[7, 2, 3]);
    }

    #[test]
    fn mutable_cpu_bytes_rejects_non_cpu_tensor() {
        let mut dlpack = dlpack_with_flags_on_device::<DLManagedTensorVersioned>(
            DlpackFlags::empty(),
            DLDevice::cuda(0),
        );
        let mut tensor = dlpack.validate_mut().unwrap();
        let error = unsafe { tensor.cpu_bytes_mut() }.unwrap_err();

        assert!(matches!(error, tensor::Error::NotCpu { .. }));
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
