use std::ffi::c_void;

use crate::ffi::{DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor};
use bitflags::bitflags;

impl Default for DLPackVersion {
    fn default() -> Self {
        Self {
            major: crate::ffi::DLPACK_MAJOR_VERSION,
            minor: crate::ffi::DLPACK_MINOR_VERSION,
        }
    }
}

bitflags! {
    /// Flags carried by `DLManagedTensorVersioned`.
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DlpackFlags: u64 {
        /// Consumers must not modify the tensor data.
        const READ_ONLY = 1 << 0;
        /// The export owns an unaliased copy of the tensor data.
        const IS_COPIED = 1 << 1;
        /// Packed sub-byte data has per-element padding.
        const IS_SUBBYTE_TYPE_PADDED = 1 << 2;
    }
}

impl DlpackFlags {
    /// Whether setting `self` as the new flags, given the tensor's `current`
    /// flags, would newly assert [`DlpackFlags::IS_COPIED`] — turn it on
    /// when it wasn't already set.
    ///
    /// Turning it on is the risky direction: `ManagedBox::cpu_data_slice_mut`
    /// and `array_view_from_dlpack_mut` trust it unconditionally to skip
    /// aliasing checks. Leaving an already-set `IS_COPIED` on asserts nothing
    /// new, so that case is not flagged.
    pub(crate) fn newly_asserts_is_copied(self, current: DlpackFlags) -> bool {
        self.contains(DlpackFlags::IS_COPIED) && !current.contains(DlpackFlags::IS_COPIED)
    }
}

/// Common operations implemented by the legacy and versioned managed tensor
/// ABIs.
///
/// This trait lets [`crate::Builder`] and [`crate::ManagedBox`] operate
/// generically while preserving the concrete C layout selected by the caller.
/// # Safety
///
/// Implementations must use a stable C-compatible layout for which
/// [`Self::from_parts`] fully initializes a valid managed tensor. Accessors
/// must return fields from that same value, and `deleter` must be safe to call
/// exactly once with the original pointer. The type must have nonzero size and
/// an alignment accepted by Rust's global allocator.
pub unsafe trait ManagedTensorBase {
    /// Constructs a managed tensor from its embedded tensor and ownership
    /// fields.
    fn from_parts(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self;

    /// Returns the embedded tensor descriptor.
    fn tensor(&self) -> &DLTensor;
    /// Returns mutable access used while initializing the descriptor.
    fn tensor_mut(&mut self) -> &mut DLTensor;
    /// Returns the producer-owned opaque context pointer.
    fn manager_ctx(&self) -> *mut c_void;
    /// Returns the managed tensor deleter.
    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)>;
    /// Returns versioned flags, or empty flags for the legacy ABI.
    fn flags(&self) -> DlpackFlags {
        DlpackFlags::empty()
    }

    /// Applies DLPack flags to this managed tensor verbatim, including
    /// [`DlpackFlags::IS_COPIED`].
    ///
    /// Only `DLManagedTensorVersioned` carries a `flags` field; the legacy
    /// `DLManagedTensor` has none and inherits the default no-op, so callers
    /// can set flags generically over `M` without knowing which ABI it is.
    ///
    /// # Safety
    ///
    /// If `flags` includes `IS_COPIED`, the caller must ensure that no other
    /// reference to the tensor's data exists: `ManagedBox::cpu_data_slice_mut`
    /// and `array_view_from_dlpack_mut` trust that bit unconditionally and
    /// skip aliasing checks accordingly.
    unsafe fn set_flags_unchecked(&mut self, _flags: crate::DlpackFlags) {}

    /// Drops a raw managed tensor pointer through its DLPack deleter.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is a valid pointer to `Self` and has not been dropped/freed yet.
    unsafe fn drop_raw(ptr: *mut Self) {
        if let Some(deleter) = unsafe { (*ptr).deleter() } {
            unsafe { deleter(ptr) };
        }
    }
}

unsafe impl ManagedTensorBase for DLManagedTensor {
    fn from_parts(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self {
        Self {
            dl_tensor: tensor,
            manager_ctx,
            deleter,
        }
    }

    fn tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }
}

unsafe impl ManagedTensorBase for DLManagedTensorVersioned {
    fn from_parts(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self {
        Self {
            version: DLPackVersion::default(),
            manager_ctx,
            deleter,
            flags: DlpackFlags::empty(),
            dl_tensor: tensor,
        }
    }

    fn tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }

    unsafe fn set_flags_unchecked(&mut self, flags: crate::DlpackFlags) {
        self.flags = flags;
    }

    fn flags(&self) -> DlpackFlags {
        self.flags
    }
}
