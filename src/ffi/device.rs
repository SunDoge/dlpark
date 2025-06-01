#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
    /// Microsoft MAIA devices
    Maia = 17,
}

/// A Device for Tensor and operator.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Device {
    /// The device type used in the device.
    pub device_type: DeviceType,
    /// The device index.
    /// For vanilla CPU memory, pinned memory, or managed memory, this is set to
    /// 0.
    pub device_id: i32,
}

impl From<(DeviceType, i32)> for Device {
    fn from(value: (DeviceType, i32)) -> Self {
        Self {
            device_type: value.0,
            device_id: value.1,
        }
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Cpu
    }
}

impl Default for Device {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
        }
    }
}

impl Device {
    /// CPU device with `device_id = 0`.
    pub const CPU: Self = Self {
        device_type: DeviceType::Cpu,
        device_id: 0,
    };

    /// Create CUDA device.
    pub fn cuda(index: usize) -> Self {
        Self {
            device_type: DeviceType::Cuda,
            device_id: index as i32,
        }
    }
}
