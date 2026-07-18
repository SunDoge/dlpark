use pyo3::{
    Borrowed, PyAny,
    exceptions::PyValueError,
    types::{PyAnyMethods, PyString},
};

use crate::ffi::{DLDevice, DLDeviceType};

/// Calls `array.__dlpack_device__()` and validates its DLPack device tuple.
pub fn dlpack_device(array: Borrowed<'_, '_, PyAny>) -> pyo3::PyResult<DLDevice> {
    let (device_type, device_id): (u32, i32) = array
        .call_method0(PyString::intern(array.py(), "__dlpack_device__"))?
        .extract()?;
    if device_id < 0 {
        return Err(PyValueError::new_err(format!(
            "DLPack device ID must be non-negative, got {device_id}"
        )));
    }

    Ok(DLDevice {
        device_type: DLDeviceType(device_type),
        device_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::{Python, types::PyModule};

    #[test]
    fn preserves_known_and_extension_device_types() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"from enum import IntEnum


class DeviceType(IntEnum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10
    CUDA_MANAGED = 13
    ONE_API = 14


class Array:
    def __init__(self, device_type, device_id):
        self.device_type = device_type
        self.device_id = device_id

    def __dlpack_device__(self):
        return (self.device_type, self.device_id)
"#,
                c"device.py",
                c"device",
            )?;
            let array = module.getattr("Array")?;
            let device_types = [1u32, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 999];

            for device_type in device_types {
                let object = array.call1((device_type, 3))?;
                let device = dlpack_device(object.as_borrowed())?;
                assert_eq!(device.device_type, DLDeviceType(device_type));
                assert_eq!(device.device_id, 3);
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn rejects_negative_device_id() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Array:
    def __init__(self, device):
        self.device = device

    def __dlpack_device__(self):
        return self.device
"#,
                c"invalid_device.py",
                c"invalid_device",
            )?;
            let array = module.getattr("Array")?;

            let negative_id = array.call1(((2, -1),))?;
            assert!(dlpack_device(negative_id.as_borrowed()).is_err());

            Ok(())
        })
        .unwrap();
    }
}
