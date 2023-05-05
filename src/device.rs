use crate::dlpack::{Device, DeviceType};

impl From<(DeviceType, i32)> for Device {
    fn from(value: (DeviceType, i32)) -> Self {
        Self {
            device_type: value.0,
            device_id: value.1,
        }
    }
}
