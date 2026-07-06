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
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DlpackFlags: u64 {
        const READ_ONLY = 1 << 0;
        const IS_COPIED = 1 << 1;
        const IS_SUBBYTE_TYPE_PADDED = 1 << 2;
    }
}

pub trait ManagedTensorBase {
    fn from_parts(
        tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(self_: *mut Self)>,
    ) -> Self;

    fn dl_tensor(&self) -> &DLTensor;
    fn dl_tensor_mut(&mut self) -> &mut DLTensor;
    fn manager_ctx(&self) -> *mut c_void;
    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)>;

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

impl ManagedTensorBase for DLManagedTensor {
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

    fn dl_tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn dl_tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }
}

impl ManagedTensorBase for DLManagedTensorVersioned {
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

    fn dl_tensor(&self) -> &DLTensor {
        &self.dl_tensor
    }
    fn dl_tensor_mut(&mut self) -> &mut DLTensor {
        &mut self.dl_tensor
    }
    fn manager_ctx(&self) -> *mut c_void {
        self.manager_ctx
    }

    fn deleter(&self) -> Option<unsafe extern "C" fn(self_: *mut Self)> {
        self.deleter
    }
}
