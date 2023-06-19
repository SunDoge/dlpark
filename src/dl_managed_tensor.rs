use crate::ffi::DLManagedTensor;

impl Default for DLManagedTensor {
    fn default() -> Self {
        DLManagedTensor {
            dl_tensor: Default::default(),
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        }
    }
}
