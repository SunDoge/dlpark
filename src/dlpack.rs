use std::os::raw::c_void;

#[repr(C)]
#[derive(Debug)]
pub enum DeviceType {
    /// CPU device
    Cpu = 1,
    /// CUDA GPU device
    Cuda = 2,
    /// Pinned CUDA CPU memory by cudaMallocHost
    CudaHost = 3,
    /// OpenCL devices.
    OpenCl = 4,
    /// Vulkan buffer for next generation graphics.
    Vulkan = 7,
    /// Metal for Apple GPU.
    Metal = 8,
    /// Verilog simulator buffer
    Vpi = 9,
    /// ROCm GPUs for AMD GPUs
    Rocm = 10,
    /// Pinned ROCm CPU memory allocated by hipMallocHost
    RocmHost = 11,
    /// Reserved extension device type,
    /// used for quickly test extension device
    /// The semantics can differ depending on the implementation.
    ExtDev = 12,
    /// CUDA managed/unified memory allocated by cudaMallocManaged
    CudaManaged = 13,
    /// Unified shared memory allocated on a oneAPI non-partititioned
    /// device. Call to oneAPI runtime is required to determine the device
    /// type, the USM allocation type and the sycl context it is bound to.
    OneApi = 14,
    /// GPU support for next generation WebGPU standard.
    WebGpu = 15,
    /// Qualcomm Hexagon DSP
    Hexagon = 16,
}

/// A Device for Tensor and operator.
#[repr(C)]
#[derive(Debug)]
pub struct Device {
    /// The device type used in the device.
    pub device_type: DeviceType,
    /// The device index.
    /// For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
    pub device_id: i32,
}

#[repr(u8)]
#[derive(Debug)]
pub enum DataTypeCode {
    /// signed integer
    Int = 0,
    /// unsigned integer
    Uint = 1,
    /// IEEE floating point
    Float = 2,
    /// Opaque handle type, reserved for testing purposes.
    /// Frameworks need to agree on the handle data type for the exchange to be well-defined.
    OpaqueHandle = 3,
    /// bfloat16
    Bfloat = 4,
    /// complex number
    /// (C/C++/Python layout: compact struct per complex number)
    Complex = 5,
}

/// The data type the tensor can hold. The data type is assumed to follow the
/// native endian-ness. An explicit error message should be raised when attempting to
/// export an array with non-native endianness
/// Examples
/// - float: type_code = 2, bits = 32, lanes=1
/// - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
/// - int8: type_code = 0, bits = 8, lanes=1
/// - std::complex<float>: type_code = 5, bits = 64, lanes = 1
#[repr(C)]
#[derive(Debug)]
pub struct DataType {
    /// Type code of base types.
    pub code: DataTypeCode,
    /// Number of bits, common choices are 8, 16, 32.
    pub bits: u8,
    /// Number of lanes in the type, used for vector types.
    pub lanes: u16,
}

/// Plain C Tensor object, does not manage memory.
#[repr(C)]
#[derive(Debug)]
pub struct Tensor {
    /// The data pointer points to the allocated data. This will be CUDA
    /// device pointer or cl_mem handle in OpenCL. It may be opaque on some device
    /// types. This pointer is always aligned to 256 bytes as in CUDA. The
    /// `byte_offset` field should be used to point to the beginning of the data.
    ///
    /// Note that as of Nov 2021, multiply libraries (CuPy, PyTorch, TensorFlow,
    /// TVM, perhaps others) do not adhere to this 256 byte aligment requirement
    /// on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
    /// (after which this note will be updated); at the moment it is recommended
    /// to not rely on the data pointer being correctly aligned.
    ///
    /// For given DLTensor, the size of memory required to store the contents of
    /// data is calculated as follows:
    /// ```rust
    /// fn get_data_size(tensor: Tensor) {}
    /// ```
    /// 
    pub data: *mut c_void,
    pub device: Device,
    pub ndim: i32,
    pub dtype: DataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct ManagedTensor {
    pub dl_tensor: Tensor,
    pub managed_ctx: *mut c_void,
    pub deleter: Option<extern "C" fn(*mut ManagedTensor)>,
}
