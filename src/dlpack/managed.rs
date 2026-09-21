//! Owning DLPack managed tensors.

use crate::ManagedTensorBase;
use crate::ffi::{DLManagedTensorVersioned, DLPackVersion};
use crate::tensor;
use crate::{AllocationDeleter, DlpackFlags};
use snafu::Snafu;
use std::{ffi::c_void, ptr::NonNull};

unsafe fn drop_managed<M: ManagedTensorBase>(context: *mut c_void) {
    unsafe { M::drop_raw(context.cast::<M>()) };
}

/// Errors raised when taking ownership of a raw managed tensor pointer.
#[derive(Debug, Snafu)]
pub enum FromRawError {
    /// The pointer is null.
    #[snafu(display("managed tensor pointer is null"))]
    Null,

    /// The managed tensor declares an incompatible DLPack version.
    #[snafu(transparent)]
    Version {
        /// The underlying version error.
        source: crate::VersionError,
    },
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
///
/// The handle is `Send + Sync`: DLPack metadata remains immutable while owned,
/// and the managed-tensor deleter must be callable from any thread.
#[repr(transparent)]
pub struct Managed<M: ManagedTensorBase>(NonNull<M>);

// SAFETY: `ManagedTensorBase` requires immutable metadata and a deleter which
// may be called from any thread. Shared access only exposes immutable metadata;
// mutable access requires `&mut self` and dereferencing the data pointer is
// always unsafe.
unsafe impl<M: ManagedTensorBase> Send for Managed<M> {}
unsafe impl<M: ManagedTensorBase> Sync for Managed<M> {}

impl<M> Managed<M>
where
    M: ManagedTensorBase,
{
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
            version.ensure_compatible_with(DLPackVersion::CURRENT)?;
        }
        Ok(managed)
    }

    /// Consumes the managed tensor and transfers it through a raw pointer.
    pub fn into_raw(self) -> *mut M {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Erases the managed-tensor representation into an exactly-once deleter.
    ///
    /// This is useful when importing DLPack into a container that stores its
    /// own pointer and metadata but must preserve the producer's allocation
    /// lifetime without parameterizing itself over the legacy or versioned
    /// header type. Dropping the returned value invokes the original DLPack
    /// deleter; the original header remains alive until then.
    pub fn into_deleter(self) -> AllocationDeleter {
        let raw = self.into_raw().cast::<c_void>();
        // SAFETY: ManagedTensorBase requires its deleter to be callable exactly
        // once from any thread without unwinding. Ownership of `raw` moved out
        // of self and is now held exclusively by this deleter.
        unsafe { AllocationDeleter::from_raw_parts(raw, drop_managed::<M>) }
    }

    /// Returns the managed tensor pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }

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
        self.validate_declared_stride_semantics()?;
        unsafe { tensor::TensorRef::from_raw(self.tensor()) }
    }

    /// Validates a locally produced descriptor before exporting it.
    ///
    /// Legacy and versioned pre-1.2 imports may use a null strides pointer as
    /// the traditional compact row-major representation. dlpark's own
    /// non-scalar exports are stricter and must always carry explicit strides.
    pub fn validate_export(&self) -> Result<tensor::TensorRef<'_>, tensor::Error> {
        let tensor = self.validate()?;
        if tensor.ndim() != 0 && tensor.strides().is_none() {
            return Err(tensor::Error::MissingExportStrides {
                ndim: i32::try_from(tensor.ndim()).expect("validated DLPack rank fits i32"),
            });
        }
        Ok(tensor)
    }

    /// Validates the descriptor for mutable access.
    ///
    /// `READ_ONLY` tensors are rejected. `IS_COPIED` is exchange metadata and
    /// does not affect Rust mutable-access validation.
    pub fn validate_mut(&mut self) -> Result<tensor::TensorMut<'_>, tensor::Error> {
        self.validate_declared_stride_semantics()?;
        let flags = unsafe { self.0.as_ref() }.flags();
        let tensor = unsafe { self.0.as_mut() }.tensor_mut();
        unsafe { tensor::TensorMut::from_raw(tensor, flags) }
    }

    /// Returns the DLPack bitmask flags (e.g. `READ_ONLY`, `IS_COPIED`).
    #[inline]
    pub fn flags(&self) -> DlpackFlags {
        unsafe { self.0.as_ref() }.flags()
    }

    fn validate_declared_stride_semantics(&self) -> Result<(), tensor::Error> {
        let managed = unsafe { self.0.as_ref() };
        let Some(version) = managed.version() else {
            return Ok(());
        };
        let descriptor = managed.tensor();
        if version.supports(DLPackVersion { major: 1, minor: 2 })
            && descriptor.ndim != 0
            && descriptor.strides.is_null()
        {
            return Err(tensor::Error::MissingVersionedStrides {
                major: version.major,
                minor: version.minor,
                ndim: descriptor.ndim,
            });
        }
        Ok(())
    }
}

impl Managed<DLManagedTensorVersioned> {
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
    use std::{
        ffi::c_void,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

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
    fn managed_handles_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Managed<DLManagedTensor>>();
        assert_send_sync::<Managed<DLManagedTensorVersioned>>();
    }

    #[test]
    fn into_deleter_releases_the_managed_tensor_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let tensor = make_test_tensor::<_, DLManagedTensor, 1>(
            Box::new(DropCounter(Arc::clone(&drops))),
            std::ptr::null_mut(),
            crate::ffi::DLDataType::U8,
            DLDevice::CPU,
            [0],
            [1],
            DlpackFlags::empty(),
        );

        let deleter = tensor.into_deleter();
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(deleter);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
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
    fn current_versioned_import_rejects_implicit_strides() {
        let tensor = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let raw = tensor.into_raw();
        unsafe { (*raw).dl_tensor.strides = std::ptr::null_mut() };
        let tensor = unsafe { Managed::from_raw(raw) }.unwrap();

        assert!(matches!(
            tensor.validate(),
            Err(tensor::Error::MissingVersionedStrides { minor, ndim: 1, .. })
                if minor >= 2
        ));
    }

    #[test]
    fn newer_minor_import_rejects_implicit_strides() {
        let tensor = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let raw = tensor.into_raw();
        unsafe {
            (*raw).version.minor = DLPackVersion::CURRENT.minor + 1;
            (*raw).dl_tensor.strides = std::ptr::null_mut();
        }
        let tensor = unsafe { Managed::from_raw(raw) }.unwrap();

        assert!(matches!(
            tensor.validate(),
            Err(tensor::Error::MissingVersionedStrides { minor, ndim: 1, .. })
                if minor > DLPackVersion::CURRENT.minor
        ));
    }

    #[test]
    fn legacy_and_pre_1_2_imports_accept_implicit_strides() {
        let versioned = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let raw = versioned.into_raw();
        unsafe {
            (*raw).version.minor = 1;
            (*raw).dl_tensor.strides = std::ptr::null_mut();
        }
        let versioned = unsafe { Managed::from_raw(raw) }.unwrap();
        assert_eq!(
            &*versioned.validate().unwrap().strides_or_compact().unwrap(),
            &[1]
        );

        let legacy = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());
        let raw = legacy.into_raw();
        unsafe { (*raw).dl_tensor.strides = std::ptr::null_mut() };
        let legacy = unsafe { Managed::from_raw(raw) }.unwrap();
        assert_eq!(
            &*legacy.validate().unwrap().strides_or_compact().unwrap(),
            &[1]
        );
    }

    #[test]
    fn export_validation_requires_explicit_non_scalar_strides() {
        let versioned = dlpack_with_flags::<DLManagedTensorVersioned>(DlpackFlags::empty());
        let raw = versioned.into_raw();
        unsafe {
            (*raw).version.minor = 1;
            (*raw).dl_tensor.strides = std::ptr::null_mut();
        }
        let versioned = unsafe { Managed::from_raw(raw) }.unwrap();
        assert!(matches!(
            versioned.validate_export(),
            Err(tensor::Error::MissingExportStrides { ndim: 1 })
        ));

        let legacy = dlpack_with_flags::<DLManagedTensor>(DlpackFlags::empty());
        let raw = legacy.into_raw();
        unsafe { (*raw).dl_tensor.strides = std::ptr::null_mut() };
        let legacy = unsafe { Managed::from_raw(raw) }.unwrap();
        assert!(matches!(
            legacy.validate_export(),
            Err(tensor::Error::MissingExportStrides { ndim: 1 })
        ));
    }

    #[test]
    fn export_validation_allows_scalar_without_strides() {
        let tensor = make_test_tensor::<_, DLManagedTensorVersioned, 0>(
            Box::new(()),
            std::ptr::null_mut(),
            crate::ffi::DLDataType::U8,
            DLDevice::CPU,
            [],
            [],
            DlpackFlags::empty(),
        );

        assert!(tensor.validate_export().is_ok());
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
