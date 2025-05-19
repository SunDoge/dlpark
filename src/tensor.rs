use std::ffi::c_void;

use crate::{
    data_type::DataType, device::Device, manager_context::TensorLike, memory_layout::MemoryLayout,
};

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

    pub fn as_slice<A>(&self) -> &[A] {
        assert_eq!(
            std::mem::size_of::<A>(),
            self.dtype.size(),
            "dtype and A size mismatch"
        );
        unsafe {
            std::slice::from_raw_parts(
                self.data.add(self.byte_offset as usize).cast(),
                self.num_elements(),
            )
        }
    }

    pub fn update<T, L>(&mut self, t: &T, layout: &L)
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        self.data = t.data_ptr();
        self.device = t.device();
        self.dtype = t.data_type();
        self.byte_offset = t.byte_offset();
        self.ndim = layout.ndim();
        self.shape = layout.shape_ptr();
        self.strides = layout.strides_ptr();
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

// pub trait TensorLike {
//     fn data_ptr(&self) -> *mut c_void;
//     fn shape(&self) -> &[i64];
//     fn strides(&self) -> Option<&[i64]>;
//     fn device(&self) -> Device;
//     fn dtype(&self) -> DataType;
//     fn byte_offset(&self) -> u64;

//     fn ndim(&self) -> i32 {
//         self.shape().len() as i32
//     }
// }

// impl<T> From<&T> for Tensor
// where
//     T: TensorLike,
// {
//     fn from(value: &T) -> Self {}
// }

pub trait ToTensor {
    fn data_ptr(&self) -> *mut c_void;
    fn shape(&self) -> &[i64];
    fn strides(&self) -> Option<&[i64]>;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;

    fn to_tensor(&self) -> Tensor {
        let shape = self.shape();
        Tensor {
            data: self.data_ptr(),
            shape: shape.as_ptr() as *mut _,
            strides: match self.strides() {
                Some(s) => s.as_ptr() as *mut _,
                None => std::ptr::null_mut(),
            },
            device: self.device(),
            dtype: self.dtype(),
            byte_offset: self.byte_offset(),
            ndim: shape.len() as i32,
        }
    }
}
