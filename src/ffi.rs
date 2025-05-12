/// This is raw unsafe dlpack code.
/// Please use the safe wrapper provided by dlpark.
use std::ffi::c_void;

use bitflags::bitflags;


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
    // pub device: Device,
    /// Number of dimensions
    pub ndim: i32,
    /// The data type of the pointer
    // pub dtype: DataType,
    /// The shape of the tensor
    pub shape: *mut i64,
    /// strides of the tensor (in number of elements, not bytes)
    /// can be NULL, indicating tensor is compact and row-majored.
    pub strides: *mut i64,
    /// The offset in bytes to the beginning pointer to data
    pub byte_offset: u64,
}

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

bitflags! {
    pub struct Flags: u64 {
        const READ_ONLY = 1 << 0;
        const IS_COPIED = 1 << 1;
        const IS_SUBBYTE_TYPE_PADDED = 1 << 2;
    }
}

/// A versioned and managed C Tensor object, manage memory of DLTensor.
/// This data structure is intended to facilitate the borrowing of DLTensor by
/// another framework. It is not meant to transfer the tensor. When the
/// borrowing framework doesn't need the tensor, it should call the deleter to
/// notify the host that the resource is no longer needed.
///
/// This is the current standard DLPack exchange data structure.
#[repr(C)]
#[derive(Debug)]
pub struct ManagedTensorVersioned {
    /// The API and ABI version of the current managed Tensor
    // pub version: PackVersion,
    /// The context of the original host framework.
    /// Stores DLManagedTensorVersioned is used in the
    /// framework. It can also be NULL.
    pub manager_ctx: *mut c_void,

    /// Destructor.
    /// This should be called to destruct manager_ctx which holds the
    /// DLManagedTensorVersioned. It can be NULL if there is no way for the
    /// caller to provide a reasonable destructor. The destructors deletes
    /// the argument self as well.
    pub deleter: Option<unsafe extern "C" fn(*mut Self)>,
    /// Additional bitmask flags information about the tensor.
    /// By default the flags should be set to 0.
    /// Future ABI changes should keep everything until this field
    /// stable, to ensure that deleter can be correctly called.
    /// Default: `DLPACK_FLAG_BITMASK_READ_ONLY`
    pub flags: u64,

    // DLTensor which is being memory managed
    pub dl_tensor: Tensor,
}
