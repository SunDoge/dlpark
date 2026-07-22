use std::ffi::c_void;

use crate::ffi::{DLManagedTensor, DLManagedTensorVersioned, DLPackVersion, DLTensor};
use bitflags::bitflags;
use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(display("incompatible DLPack major version: expected {expected}, got {actual}"))]
pub struct VersionError {
    pub expected: u32,
    pub actual: u32,
}

pub(crate) fn validate_version(version: DLPackVersion) -> Result<(), VersionError> {
    if !version.is_compatible_with(DLPackVersion::CURRENT) {
        return Err(VersionError {
            expected: DLPackVersion::CURRENT.major,
            actual: version.major,
        });
    }
    Ok(())
}

impl Default for DLPackVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl DLPackVersion {
    /// The DLPack version provided by the bundled headers.
    pub const CURRENT: Self = Self {
        major: crate::ffi::DLPACK_MAJOR_VERSION,
        minor: crate::ffi::DLPACK_MINOR_VERSION,
    };

    /// Returns whether two versions use a compatible ABI.
    pub const fn is_compatible_with(self, other: Self) -> bool {
        self.major == other.major
    }

    /// Returns whether this version includes the requested feature level.
    pub const fn supports(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl PartialEq for DLPackVersion {
    fn eq(&self, other: &Self) -> bool {
        self.major == other.major && self.minor == other.minor
    }
}

impl Eq for DLPackVersion {}

impl std::hash::Hash for DLPackVersion {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.major, state);
        std::hash::Hash::hash(&self.minor, state);
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
    /// Turning it on is the risky direction: `Local::cpu_slice_mut`
    /// and safe mutable ndarray conversion trust it unconditionally to skip
    /// aliasing checks. Leaving an already-set `IS_COPIED` on asserts nothing
    /// new, so that case is not flagged.
    pub(crate) fn newly_asserts_is_copied(self, current: DlpackFlags) -> bool {
        self.contains(DlpackFlags::IS_COPIED) && !current.contains(DlpackFlags::IS_COPIED)
    }
}

/// Common operations implemented by the legacy and versioned managed tensor
/// ABIs.
///
/// This trait lets allocation and ownership APIs operate
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
    /// Returns the declared ABI version, or `None` for the legacy ABI.
    #[inline]
    fn version(&self) -> Option<DLPackVersion> {
        None
    }
    /// Returns versioned flags, or empty flags for the legacy ABI.
    #[inline]
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
    /// reference to the tensor's data exists: `Local::cpu_slice_mut`
    /// and safe mutable ndarray conversion trust that bit unconditionally and
    /// skip aliasing checks accordingly.
    unsafe fn set_flags_unchecked(&mut self, _flags: crate::DlpackFlags) {}

    /// Drops a raw managed tensor pointer through its DLPack deleter.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is a valid pointer to `Self` and has not been dropped/freed yet.
    #[inline]
    unsafe fn drop_raw(ptr: *mut Self) {
        if let Some(deleter) = unsafe { (*ptr).deleter() } {
            unsafe { deleter(ptr) };
        }
    }
}

unsafe impl ManagedTensorBase for DLManagedTensor {
    #[inline]
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

    #[inline]
    fn tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    #[inline]
    fn tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    #[inline]
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    #[inline]
    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }
}

unsafe impl ManagedTensorBase for DLManagedTensorVersioned {
    #[inline]
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

    #[inline]
    fn tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    #[inline]
    fn tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    #[inline]
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    #[inline]
    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }

    #[inline]
    fn version(&self) -> Option<DLPackVersion> {
        Some(self.version)
    }

    #[inline]
    unsafe fn set_flags_unchecked(&mut self, flags: crate::DlpackFlags) {
        self.flags = flags;
    }

    #[inline]
    fn flags(&self) -> DlpackFlags {
        self.flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_comparisons_are_semantic() {
        let current = DLPackVersion::CURRENT;
        let newer_minor = DLPackVersion {
            major: current.major,
            minor: current.minor + 1,
        };
        let other_major = DLPackVersion {
            major: current.major + 1,
            minor: 0,
        };

        assert!(current.is_compatible_with(newer_minor));
        assert!(newer_minor.supports(current));
        assert!(!current.supports(newer_minor));
        assert!(!current.is_compatible_with(other_major));
        assert!(!current.supports(other_major));
        assert_eq!(DLPackVersion::default(), current);
    }
}
