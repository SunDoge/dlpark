use std::ptr::NonNull;

use crate::ffi::{self, DLManagedTensor, DataType, Device};

use crate::manager_ctx::CowIntArray;

// User should define their own InferDtype trait.
pub(crate) trait InferDtype {
    fn infer_dtype() -> DataType;
}

pub trait TensorView {
    /// Get untyped data ptr
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    /// Get shape as slice.
    fn shape(&self) -> &[i64];
    /// Get strides as slice, can have no strides.
    fn strides(&self) -> Option<&[i64]>;
    fn ndim(&self) -> usize;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;

    // Get num elements in Tensor.
    fn num_elements(&self) -> usize {
        self.shape().iter().product::<i64>() as usize
    }

    /// For given DLTensor, the size of memory required to store the contents of
    /// data is calculated as follows:
    ///
    /// ```c
    /// static inline size_t GetDataSize(const DLTensor* t) {
    ///   size_t size = 1;
    ///   for (tvm_index_t i = 0; i < t->ndim; ++i) {
    ///     size *= t->shape[i];
    ///   }
    ///   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    ///   return size;
    /// }
    /// ```
    fn data_size(&self) -> usize {
        self.num_elements() * self.dtype().size()
    }
}

pub(crate) trait GetInitializedDLTensor {
    fn get_initialized_dl_tensor(&self) -> &ffi::DLTensor;
}

pub trait ToTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn shape(&self) -> CowIntArray;
    fn strides(&self) -> Option<CowIntArray>;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;
}

pub trait ToDLPack {
    fn to_dlpack(self) -> NonNull<DLManagedTensor>;
}

pub trait FromDLPack {
    fn from_dlpack(src: NonNull<DLManagedTensor>) -> Self;
}
