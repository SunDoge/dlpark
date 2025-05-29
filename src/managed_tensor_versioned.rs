use std::ffi::c_void;
use std::ptr::NonNull;

use bitflags::bitflags;

use crate::{pack_version::PackVersion, tensor::Tensor};

bitflags! {
    pub struct Flags: u64 {
        const READ_ONLY = 1 << 0;
        const IS_COPIED = 1 << 1;
        const IS_SUBBYTE_TYPE_PADDED = 1 << 2;
    }
}

impl Default for Flags {
    fn default() -> Self {
        Flags::READ_ONLY
    }
}

/// A versioned and managed C Tensor object, manage memory of DLTensor.
/// This data structure is intended to facilitate the borrowing of DLTensor by
/// another framework. It is not meant to transfer the tensor. When the
/// borrowing framework doesn't need the tensor, it should call the deleter to
/// notify the host that the resource is no longer needed.
///
/// This is the current standard DLPack exchange data structure.
#[repr(C)]
#[derive(Debug)]
pub struct ManagedTensorVersioned {
    /// The API and ABI version of the current managed Tensor
    pub version: PackVersion,
    /// The context of the original host framework.
    /// Stores DLManagedTensorVersioned is used in the
    /// framework. It can also be NULL.
    pub manager_ctx: *mut c_void,

    /// Destructor.
    /// This should be called to destruct manager_ctx which holds the
    /// DLManagedTensorVersioned. It can be NULL if there is no way for the
    /// caller to provide a reasonable destructor. The destructors deletes
    /// the argument self as well.
    pub deleter: Option<unsafe extern "C" fn(*mut Self)>,
    /// Additional bitmask flags information about the tensor.
    /// By default the flags should be set to 0.
    /// Future ABI changes should keep everything until this field
    /// stable, to ensure that deleter can be correctly called.
    /// Default: `DLPACK_FLAG_BITMASK_READ_ONLY`
    pub flags: u64,
    // DLTensor which is being memory managed
    pub dl_tensor: Tensor,
}

impl Default for ManagedTensorVersioned {
    fn default() -> Self {
        Self {
            version: PackVersion::default(),
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
            flags: Flags::default().bits(),
            dl_tensor: Tensor::default(),
        }
    }
}

pub trait IntoDlpackVersioned {
    fn into_dlpack_versioned(self, flags: Flags) -> NonNull<ManagedTensorVersioned>;
}

pub trait FromDlpackVersioned {
    fn from_dlpack_versioned(pack: NonNull<ManagedTensorVersioned>) -> Self;
}
