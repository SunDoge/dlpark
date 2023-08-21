use std::ptr::NonNull;

use crate::ffi::{self, DataType, Device};
use crate::utils::is_contiguous;
use crate::ShapeAndStrides;

/// DLPack is a data structure that can be used to describe tensor data.
/// It's a pointer to a DLManagedTensor.
pub type DLPack = NonNull<ffi::DLManagedTensor>;

/// Infer DataType from generic parameter.
pub trait InferDtype {
    fn infer_dtype() -> DataType;
}

/// Access Tensor data.
pub trait TensorView {
    /// Get untyped data ptr
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    /// Get shape as slice.
    fn shape(&self) -> &[i64];
    /// Get strides as slice. If strides is None, Tensor is assumed to be contiguous.
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
            Some(strides) => is_contiguous(self.shape(), strides),
            None => true,
        }
    }
}

/// User should implement this trait for their tensor.
pub trait ToTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    /// If return None, tensor must be contiguous.
    fn shape_and_strides(&self) -> ShapeAndStrides;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;
}

// TODO: we should add `try_to_dlpack` fn
// We may have to define error type for this.
/// Convert into [`DLPack`](crate::DLPack)
pub trait IntoDLPack {
    fn into_dlpack(self) -> DLPack;
}

// TODO: we should add `try_from_dlpack` fn
/// Make Tensor from [`DLPack`](crate::DLPack)
pub trait FromDLPack {
    // TODO: DLManagedTensor will be deprecated in th future.
    fn from_dlpack(dlpack: DLPack) -> Self;
}
