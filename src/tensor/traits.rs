use std::ptr::NonNull;

use crate::ffi::{self, DataType, Device};

use crate::manager_ctx::CowIntArray;

use super::calculate_contiguous_strides;

pub type DLPack = NonNull<ffi::DLManagedTensor>;

/// Infer DataType from generic parameter.
pub trait InferDtype {
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

    /// Return true if tensor is contiguous in memory in the order specified by memory format.
    fn is_contiguous(&self) -> bool {
        match self.strides() {
            Some(strides) => strides == calculate_contiguous_strides(self.shape()),
            None => true,
        }
    }
}

/// User should implement this trait for their tensor.
pub trait ToTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn shape(&self) -> CowIntArray;
    /// If return None, tensor must be contiguous.
    /// Or you can calculate contiguous strides by
    /// [`self.calcualte_contiguous_strides()`](ToTensor::calculate_contiguous_strides)
    fn strides(&self) -> Option<CowIntArray>;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;

    fn calculate_contiguous_strides(&self) -> CowIntArray {
        let strides = calculate_contiguous_strides(self.shape().as_slice());
        CowIntArray::from_owned(strides)
    }
}

// TODO: we should add `try_to_dlpack` fn
pub trait ToDLPack {
    fn to_dlpack(self) -> DLPack;
}

// TODO: we should add `try_from_dlpack` fn
pub trait FromDLPack {
    // TODO: DLManagedTensor will be deprecated in th future.
    fn from_dlpack(dlpack: DLPack) -> Self;
}
