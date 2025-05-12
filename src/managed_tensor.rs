use std::ffi::c_void;

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

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        // SAFETY: The pointer is valid and the memory is managed by the DLPack library.
        if let Some(deleter) = self.deleter {
            unsafe { deleter(self) }
        }
    }
}
