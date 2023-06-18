use std::{ffi::c_void, ptr::NonNull};

use crate::ffi::{DLManagedTensor, DataType, Device};

use super::{Shape, Strides};

pub trait HasData {
    fn data(&self) -> *mut c_void;
}

pub trait HasShape {
    fn shape(&self) -> Shape;
}

/// Can be `None`, indicating tensor is compact and row-majored.
pub trait HasStrides {
    fn strides(&self) -> Option<Strides> {
        None
    }
}

pub trait HasByteOffset {
    fn byte_offset(&self) -> u64;
}

pub trait HasDevice {
    fn device(&self) -> Device;
}

pub trait HasDtype {
    fn dtype(&self) -> DataType;
}

pub trait InferDtype {
    fn infer_dtype() -> DataType;
}

pub trait AsTensor {
    fn data<T>(&self) -> *const T;
    fn shape(&self) -> &[i64];
    fn strides(&self) -> Option<&[i64]>;
    fn ndim(&self) -> usize;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;

    fn byte_offset(&self) -> u64 {
        0
    }

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

pub trait HasTensor<T>
where
    T: AsTensor,
{
    fn tensor(&self) -> &T;
}

pub trait ToDLPack {
    fn to_dlpack(self) -> NonNull<DLManagedTensor>;
}

pub trait FromDLPack {
    fn from_dlpack(src: NonNull<DLManagedTensor>) -> Self;
}
