use crate::ffi::{DLDevice, DLDeviceType};

impl DLDevice {
    /// The process-local CPU device.
    pub const CPU: Self = Self {
        device_type: DLDeviceType::CPU,
        device_id: 0,
    };

    /// Constructs a CUDA device descriptor with the given ordinal.
    pub fn cuda(device_id: i32) -> Self {
        Self {
            device_type: DLDeviceType::CUDA,
            device_id,
        }
    }

    /// Constructs a Metal device descriptor with the given registry index.
    pub fn metal(device_id: i32) -> Self {
        Self {
            device_type: DLDeviceType::METAL,
            device_id,
        }
    }
}

impl DLDeviceType {
    /// Returns whether this device type is defined by the bundled DLPack
    /// headers.
    pub const fn is_known(self) -> bool {
        const CPU: u32 = DLDeviceType::CPU.0;
        const OPENCL: u32 = DLDeviceType::OPENCL.0;
        const VULKAN: u32 = DLDeviceType::VULKAN.0;
        const TRN: u32 = DLDeviceType::TRN.0;

        matches!(self.0, CPU..=OPENCL | VULKAN..=TRN)
    }
}

impl Default for DLDevice {
    fn default() -> Self {
        Self {
            device_type: DLDeviceType(0),
            device_id: 0,
        }
    }
}

impl From<DLDevice> for (u32, i32) {
    fn from(device: DLDevice) -> Self {
        (device.device_type.0, device.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_converts_to_python_protocol_tuple() {
        for (device, expected) in [
            (DLDevice::CPU, (DLDeviceType::CPU.0, 0)),
            (DLDevice::cuda(3), (DLDeviceType::CUDA.0, 3)),
            (DLDevice::metal(7), (DLDeviceType::METAL.0, 7)),
        ] {
            assert_eq!(<(u32, i32)>::from(device), expected);
        }
    }

    #[test]
    fn device_type_knownness_tracks_bundled_header() {
        for value in [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] {
            assert!(DLDeviceType(value).is_known());
        }
        for value in [0, 5, 6, 19, u32::MAX] {
            assert!(!DLDeviceType(value).is_known());
        }
    }
}
