use crate::ffi::{DLDevice, DLDeviceType};

impl DLDevice {
    pub const CPU: Self = Self {
        device_type: DLDeviceType::CPU,
        device_id: 0,
    };

    pub fn cuda(device_id: i32) -> Self {
        Self {
            device_type: DLDeviceType::CUDA,
            device_id,
        }
    }
}
