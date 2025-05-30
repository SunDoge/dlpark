use std::ffi::c_void;

use super::{data_type::DataType, device::Device};

use crate::{
    error::{DataTypeSizeMismatchSnafu, Result},
    utils,
};

use snafu::ensure;

/// Plain C Tensor object, does not manage memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    /// The data pointer points to the allocated data. This will be CUDA
    /// device pointer or cl_mem handle in OpenCL. It may be opaque on some
    /// device types. This pointer is always aligned to 256 bytes as in
    /// CUDA. The `byte_offset` field should be used to point to the
    /// beginning of the data.
    ///
    /// Note that as of Nov 2021, multiply libraries (CuPy, PyTorch, TensorFlow,
    /// TVM, perhaps others) do not adhere to this 256 byte aligment requirement
    /// on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
    /// (after which this note will be updated); at the moment it is recommended
    /// to not rely on the data pointer being correctly aligned.
    ///
    /// For given DLTensor, the size of memory required to store the contents of
    /// data is calculated as follows:
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
    pub data: *mut c_void,
    /// The device of the tensor
    pub device: Device,
    /// Number of dimensions
    pub ndim: i32,
    /// The data type of the pointer
    pub dtype: DataType,
    /// The shape of the tensor
    pub shape: *mut i64,
    /// strides of the tensor (in number of elements, not bytes)
    /// can be NULL, indicating tensor is compact and row-majored.
    pub strides: *mut i64,
    /// The offset in bytes to the beginning pointer to data
    pub byte_offset: u64,
}

impl Tensor {
    pub fn get_shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.shape, self.num_dimensions()) }
    }

    pub fn get_strides(&self) -> Option<&[i64]> {
        if self.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.strides, self.num_dimensions()) })
        }
    }

    pub fn num_dimensions(&self) -> usize {
        self.ndim as usize
    }

    pub fn num_elements(&self) -> usize {
        self.get_shape().iter().product::<i64>() as usize
    }

    pub fn as_slice_untyped(&self) -> &[u8] {
        let length = self.num_elements() * self.dtype.size();
        unsafe {
            std::slice::from_raw_parts(self.data.add(self.byte_offset as usize).cast(), length)
        }
    }

    pub unsafe fn as_slice<A>(&self) -> Result<&[A]> {
        let size = std::mem::size_of::<A>();
        let expected = self.dtype.size();
        ensure!(
            size == expected,
            DataTypeSizeMismatchSnafu { size, expected }
        );

        let s = unsafe {
            std::slice::from_raw_parts(
                self.data.add(self.byte_offset as usize).cast(),
                self.num_elements(),
            )
        };
        Ok(s)
    }

    pub fn is_contiguous(&self) -> bool {
        match (self.get_shape(), self.get_strides()) {
            (_shape, None) => true,
            (shape, Some(strides)) => utils::is_contiguous(shape, strides),
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            device: Device::CPU,
            ndim: 0,
            dtype: DataType::F32,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        }
    }
}
