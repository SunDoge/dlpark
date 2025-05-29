use std::{ffi::c_void, ptr::NonNull};

use crate::tensor::Tensor;

/// C Tensor object, manage memory of DLTensor. This data structure is
/// intended to facilitate the borrowing of DLTensor by another framework. It is
/// not meant to transfer the tensor. When the borrowing framework doesn't need
/// the tensor, it should call the deleter to notify the host that the resource
/// is no longer needed.
#[repr(C)]
#[derive(Debug)]
pub struct ManagedTensor {
    pub dl_tensor: Tensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut Self)>,
}

impl Default for ManagedTensor {
    fn default() -> Self {
        Self {
            dl_tensor: Tensor::default(),
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        }
    }
}

pub trait IntoDlpack {
    fn into_dlpack(self) -> NonNull<ManagedTensor>;
}

pub trait FromDlpack {
    fn from_dlpack(pack: NonNull<ManagedTensor>) -> Self;
}
