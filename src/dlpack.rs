/// DLPACK_VERSION 70
/// DLPACK_ABI_VERSION 1
use std::os::raw::c_void;

#[repr(C)]
#[derive(Debug)]
pub enum DeviceType {
    Cpu = 1,
    Cuda = 2,
    CudaHost = 3,
    OpenCl = 4,
    Vulkan = 7,
    Metal = 8,
    Vpi = 9,
    Rocm = 10,
    RocmHost = 11,
    ExtDev = 12,
    CudaManaged = 13,
    OneApi = 14,
    WebGpu = 15,
    Hexagon = 16,
}

#[repr(C)]
#[derive(Debug)]
pub struct Device {
    pub device_type: DeviceType,
    pub device_id: i32,
}

#[repr(u8)]
#[derive(Debug)]
pub enum DataTypeCode {
    Int = 0,
    Uint = 1,
    Float = 2,
    OpaqueHandle = 3,
    Bfloat = 4,
    Complex = 5,
}

#[repr(C)]
#[derive(Debug)]
pub struct DataType {
    pub code: DataTypeCode,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Debug)]
pub struct Tensor {
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
