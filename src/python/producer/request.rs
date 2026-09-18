//! Parsing and validation for Python `__dlpack__` export requests.

use crate::{
    DlpackFlags, Managed,
    ffi::{
        DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLPACK_MAJOR_VERSION,
    },
};
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
    exceptions::{PyBufferError, PyValueError},
    types::PyAnyMethods,
};

/// Managed-tensor ABI selected from a consumer's `max_version` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportAbi {
    /// The legacy `DLManagedTensor` ABI and `"dltensor"` capsule name.
    Legacy,
    /// The `DLManagedTensorVersioned` ABI and `"dltensor_versioned"` capsule name.
    Versioned,
}

/// Parsed CUDA stream argument supplied to Python `__dlpack__`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStreamRequest {
    /// The consumer omitted `stream`.
    ///
    /// The producer must use the protocol's legacy behavior. This is distinct
    /// from [`Self::NoSync`] and must not be converted to a native stream.
    Omitted,
    /// The consumer passed `-1` and requests no synchronization.
    NoSync,
    /// CUDA's legacy default stream sentinel (`1`).
    LegacyDefault,
    /// CUDA's per-thread default stream sentinel (`2`).
    PerThreadDefault,
    /// A positive native `cudaStream_t` address.
    Pointer(usize),
}

/// Parsed arguments supplied to a producer's Python `__dlpack__` method.
pub struct ExportRequest<'py> {
    stream: Option<Bound<'py, PyAny>>,
    max_version: Option<(u32, u32)>,
    device: Option<DLDevice>,
    copy: Option<bool>,
}

impl<'py> ExportRequest<'py> {
    /// Parses and validates the standard Python `__dlpack__` arguments.
    pub fn parse(
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Self> {
        let device = dl_device
            .map(|(device_type, device_id)| {
                if device_id < 0 {
                    return Err(PyValueError::new_err(format!(
                        "DLPack device ID must be non-negative, got {device_id}"
                    )));
                }
                Ok(DLDevice {
                    device_type: DLDeviceType(device_type),
                    device_id,
                })
            })
            .transpose()?;
        Ok(Self {
            stream: stream.cloned(),
            max_version,
            device,
            copy,
        })
    }

    /// Returns the consumer stream argument for backend-specific synchronization.
    pub fn stream(&self) -> Option<&Bound<'py, PyAny>> {
        self.stream.as_ref()
    }

    /// Parses `stream` according to the CUDA Python DLPack convention.
    pub fn cuda_stream(&self) -> PyResult<CudaStreamRequest> {
        let Some(stream) = &self.stream else {
            return Ok(CudaStreamRequest::Omitted);
        };
        let value = stream
            .extract::<isize>()
            .map_err(|_| PyValueError::new_err("CUDA DLPack stream must be an integer"))?;
        match value {
            -1 => Ok(CudaStreamRequest::NoSync),
            0 => Err(PyValueError::new_err(
                "CUDA DLPack stream 0 is ambiguous; use sentinel 1 for the legacy default stream",
            )),
            1 => Ok(CudaStreamRequest::LegacyDefault),
            2 => Ok(CudaStreamRequest::PerThreadDefault),
            value if value > 2 => Ok(CudaStreamRequest::Pointer(value as usize)),
            _ => Err(PyValueError::new_err(
                "CUDA DLPack stream must be -1, 1, 2, or a positive stream pointer",
            )),
        }
    }

    /// Returns the consumer's maximum supported DLPack version.
    pub fn max_version(&self) -> Option<(u32, u32)> {
        self.max_version
    }

    /// Returns the requested destination device, if any.
    pub fn device(&self) -> Option<DLDevice> {
        self.device
    }

    /// Returns the tri-state copy request.
    pub fn copy(&self) -> Option<bool> {
        self.copy
    }

    /// Selects the ABI requested by `max_version`.
    pub fn abi(&self) -> ExportAbi {
        if self
            .max_version
            .is_some_and(|(major, _minor)| major >= DLPACK_MAJOR_VERSION)
        {
            ExportAbi::Versioned
        } else {
            ExportAbi::Legacy
        }
    }

    /// Validates a zero-copy export and produces the selected capsule.
    ///
    /// Backend-specific stream synchronization must be completed before this
    /// method is called. Each closure must create a fresh managed-tensor header.
    pub fn export_zero_copy<L, V>(
        &self,
        py: Python<'py>,
        source_device: DLDevice,
        flags: DlpackFlags,
        legacy: L,
        versioned: V,
    ) -> PyResult<Py<PyAny>>
    where
        L: FnOnce() -> PyResult<Managed<DLManagedTensor>>,
        V: FnOnce() -> PyResult<Managed<DLManagedTensorVersioned>>,
    {
        if self.copy == Some(true) {
            return Err(PyBufferError::new_err(
                "this producer only supports zero-copy export",
            ));
        }
        if flags.contains(DlpackFlags::IS_COPIED) {
            return Err(PyBufferError::new_err(
                "a zero-copy export cannot set the DLPack IS_COPIED flag",
            ));
        }
        if let Some(requested) = self.device
            && requested != source_device
        {
            return Err(PyBufferError::new_err(format!(
                "cross-device copies are not supported: tensor is on {:?}:{}, requested {:?}:{}",
                source_device.device_type,
                source_device.device_id,
                requested.device_type,
                requested.device_id
            )));
        }

        match self.abi() {
            ExportAbi::Versioned => Ok(versioned()?.into_pyobject(py)?.unbind()),
            ExportAbi::Legacy => {
                if !flags.is_empty() {
                    return Err(PyBufferError::new_err(
                        "the legacy DLPack ABI cannot represent versioned tensor flags",
                    ));
                }
                Ok(legacy()?.into_pyobject(py)?.unbind())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocation::fixed::make_test_tensor,
        ffi::{DLDataType, DLManagedTensor, DLManagedTensorVersioned},
        python::{ImportedDlpack, from_dlpack},
    };
    use pyo3::types::PyInt;

    fn tensor<M: crate::ManagedTensorBase>() -> Managed<M> {
        let data = Box::new(vec![1_i32, 2, 3]);
        let data_ptr = data.as_ptr().cast_mut().cast();
        make_test_tensor(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice::CPU,
            [3],
            [1],
            DlpackFlags::empty(),
        )
    }

    #[test]
    fn selects_abi_and_validates_zero_copy() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let legacy = ExportRequest::parse(None, None, None, None)?;
            assert_eq!(legacy.abi(), ExportAbi::Legacy);
            let capsule = legacy.export_zero_copy(
                py,
                DLDevice::CPU,
                DlpackFlags::empty(),
                || Ok(tensor::<DLManagedTensor>()),
                || panic!("legacy request selected the versioned closure"),
            )?;
            assert!(matches!(
                from_dlpack(capsule.bind(py).as_borrowed(), None, None)?,
                ImportedDlpack::Legacy(_)
            ));

            let versioned = ExportRequest::parse(None, Some((1, 0)), None, Some(false))?;
            assert_eq!(versioned.abi(), ExportAbi::Versioned);
            assert!(
                versioned
                    .export_zero_copy(
                        py,
                        DLDevice::CPU,
                        DlpackFlags::empty(),
                        || panic!("versioned request selected the legacy closure"),
                        || Ok(tensor::<DLManagedTensorVersioned>()),
                    )
                    .is_ok()
            );

            let copy = ExportRequest::parse(None, Some((1, 0)), None, Some(true))?;
            assert!(
                copy.export_zero_copy(
                    py,
                    DLDevice::CPU,
                    DlpackFlags::empty(),
                    || unreachable!(),
                    || unreachable!(),
                )
                .is_err()
            );

            for flags in [
                DlpackFlags::READ_ONLY,
                DlpackFlags::IS_COPIED,
                DlpackFlags::IS_SUBBYTE_TYPE_PADDED,
            ] {
                let legacy = ExportRequest::parse(None, None, None, None)?;
                assert!(
                    legacy
                        .export_zero_copy(
                            py,
                            DLDevice::CPU,
                            flags,
                            || unreachable!(),
                            || unreachable!(),
                        )
                        .is_err()
                );
            }

            let copied = ExportRequest::parse(None, Some((1, 0)), None, None)?;
            assert!(
                copied
                    .export_zero_copy(
                        py,
                        DLDevice::CPU,
                        DlpackFlags::IS_COPIED,
                        || unreachable!(),
                        || unreachable!(),
                    )
                    .is_err()
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parses_cuda_stream_protocol_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let request = ExportRequest::parse(None, None, None, None)?;
            assert_eq!(request.cuda_stream()?, CudaStreamRequest::Omitted);

            for (value, expected) in [
                (-1, CudaStreamRequest::NoSync),
                (1, CudaStreamRequest::LegacyDefault),
                (2, CudaStreamRequest::PerThreadDefault),
                (17, CudaStreamRequest::Pointer(17)),
            ] {
                let stream = PyInt::new(py, value);
                let request = ExportRequest::parse(Some(stream.as_any()), None, None, None)?;
                assert_eq!(request.cuda_stream()?, expected);
            }

            let zero = PyInt::new(py, 0);
            let request = ExportRequest::parse(Some(zero.as_any()), None, None, None)?;
            assert!(request.cuda_stream().is_err());
            Ok(())
        })
        .unwrap();
    }
}
