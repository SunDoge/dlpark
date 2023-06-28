use crate::ffi::{Device, DeviceType};

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
