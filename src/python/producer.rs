//! Helpers for implementing Python DLPack producers.

use crate::{
    DlpackFlags, Managed,
    ffi::{
        DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLPACK_MAJOR_VERSION,
        DLPACK_MINOR_VERSION, DLPackExchangeAPI, DLPackExchangeAPIHeader, DLPackVersion, DLTensor,
    },
};
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyClass, PyRef, PyResult, Python,
    exceptions::{PyBufferError, PyNotImplementedError, PyRuntimeError, PyValueError},
    types::PyAnyMethods,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

const DLPACK_EXCHANGE_API: &CStr = c"dlpack_exchange_api";

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
    /// The consumer omitted `stream`; the producer must conservatively make
    /// the tensor ready before returning.
    Unspecified,
    /// The consumer passed `-1` and requests no synchronization.
    NoSync,
    /// CUDA's legacy default stream sentinel (`1`).
    LegacyDefault,
    /// CUDA's per-thread default stream sentinel (`2`).
    PerThreadDefault,
    /// A positive native `cudaStream_t` address.
    Pointer(usize),
}

impl CudaStreamRequest {
    /// Returns the native `cudaStream_t` representation when synchronization
    /// should target a consumer stream.
    pub fn as_raw(self) -> Option<*mut c_void> {
        match self {
            Self::Unspecified | Self::NoSync => None,
            Self::LegacyDefault => Some(ptr::null_mut()),
            Self::PerThreadDefault => Some(ptr::without_provenance_mut(2)),
            Self::Pointer(address) => Some(ptr::without_provenance_mut(address)),
        }
    }

    /// Returns the integer value used by the Python DLPack protocol.
    pub fn python_value(self) -> Option<isize> {
        match self {
            Self::Unspecified => None,
            Self::NoSync => Some(-1),
            Self::LegacyDefault => Some(1),
            Self::PerThreadDefault => Some(2),
            Self::Pointer(address) => isize::try_from(address).ok(),
        }
    }
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
            return Ok(CudaStreamRequest::Unspecified);
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
                if flags.contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED) {
                    return Err(PyBufferError::new_err(
                        "the legacy DLPack ABI cannot describe padded sub-byte elements",
                    ));
                }
                Ok(legacy()?.into_pyobject(py)?.unbind())
            }
        }
    }
}

/// Rust implementation behind a PyO3 class's DLPack C Exchange API table.
///
/// The callbacks are no-sync by definition.
///
/// # Safety
///
/// `tensor_view_no_sync` must return data, shape, and stride pointers that stay
/// valid and immutable until that callback returns. Managed tensors must obey
/// DLPack's ownership contract, and `current_work_stream` must return a live
/// backend stream for the requested device.
pub unsafe trait DlpackExchangeProducer: PyClass {
    /// Creates a fresh owning versioned managed tensor without synchronization.
    fn managed_tensor_no_sync(&self, py: Python<'_>)
    -> PyResult<Managed<DLManagedTensorVersioned>>;

    /// Returns a temporary borrowed tensor descriptor without synchronization.
    fn tensor_view_no_sync(&self, py: Python<'_>) -> PyResult<DLTensor>;

    /// Returns the producer's current native work stream for `device`.
    fn current_work_stream(py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void>;

    /// Allocates a new tensor from a prototype descriptor.
    fn allocate(_prototype: &DLTensor) -> Result<Managed<DLManagedTensorVersioned>, String> {
        Err("DLPack managed-tensor allocation is not supported".into())
    }

    /// Converts an incoming versioned managed tensor into this Python class.
    fn from_managed_tensor_no_sync(
        _py: Python<'_>,
        _tensor: Managed<DLManagedTensorVersioned>,
    ) -> PyResult<Py<Self>> {
        Err(PyNotImplementedError::new_err(
            "DLPack managed-tensor import is not supported",
        ))
    }
}

/// Installs a process-lifetime DLPack C Exchange API capsule on a PyO3 class.
///
/// Call this once while initializing the extension module, after registering
/// the class with `module.add_class::<T>()`.
pub fn install_exchange_api<T>(py: Python<'_>) -> PyResult<()>
where
    T: DlpackExchangeProducer,
{
    let api = Box::leak(Box::new(DLPackExchangeAPI {
        header: DLPackExchangeAPIHeader {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            prev_api: ptr::null_mut(),
        },
        managed_tensor_allocator: Some(allocate::<T>),
        managed_tensor_from_py_object_no_sync: Some(managed_from_python::<T>),
        managed_tensor_to_py_object_no_sync: Some(managed_to_python::<T>),
        dltensor_from_py_object_no_sync: Some(tensor_view::<T>),
        current_work_stream: Some(current_work_stream::<T>),
    }));
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            (api as *mut DLPackExchangeAPI).cast(),
            DLPACK_EXCHANGE_API.as_ptr(),
            None,
        )
    };
    let capsule = unsafe { Bound::from_owned_ptr_or_err(py, capsule)? };
    T::type_object(py).setattr("__dlpack_c_exchange_api__", capsule)
}

unsafe extern "C" fn managed_from_python<T>(
    object: *mut c_void,
    out: *mut *mut DLManagedTensorVersioned,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if object.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let object = unsafe { Bound::from_borrowed_ptr(py, object.cast()) };
        let producer: PyRef<'_, T> = object.extract()?;
        let tensor = producer.managed_tensor_no_sync(py)?;
        unsafe { out.write(tensor.into_raw()) };
        Ok(())
    })
}

unsafe extern "C" fn tensor_view<T>(object: *mut c_void, out: *mut DLTensor) -> c_int
where
    T: DlpackExchangeProducer,
{
    python_callback(|| {
        if object.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let object = unsafe { Bound::from_borrowed_ptr(py, object.cast()) };
        let producer: PyRef<'_, T> = object.extract()?;
        unsafe { out.write(producer.tensor_view_no_sync(py)?) };
        Ok(())
    })
}

unsafe extern "C" fn managed_to_python<T>(
    raw: *mut DLManagedTensorVersioned,
    out: *mut *mut c_void,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if raw.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let tensor = unsafe { Managed::from_raw(raw) }
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let object = T::from_managed_tensor_no_sync(py, tensor)?;
        unsafe { out.write(object.into_ptr().cast()) };
        Ok(())
    })
}

unsafe extern "C" fn current_work_stream<T>(
    device_type: DLDeviceType,
    device_id: i32,
    out: *mut *mut c_void,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API output"));
        }
        let py = unsafe { Python::assume_attached() };
        let stream = T::current_work_stream(
            py,
            DLDevice {
                device_type,
                device_id,
            },
        )?;
        unsafe { out.write(stream) };
        Ok(())
    })
}

unsafe extern "C" fn allocate<T>(
    prototype: *mut DLTensor,
    out: *mut *mut DLManagedTensorVersioned,
    error_ctx: *mut c_void,
    set_error: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char)>,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let prototype = unsafe { prototype.as_ref() }.ok_or("prototype is null")?;
        let out = unsafe { out.as_mut() }.ok_or("output is null")?;
        *out = T::allocate(prototype)?.into_raw();
        Ok::<(), String>(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => {
            set_exchange_error(error_ctx, set_error, &message);
            -1
        }
        Err(_) => {
            set_exchange_error(error_ctx, set_error, "Rust panic in DLPack allocator");
            -1
        }
    }
}

fn python_callback(callback: impl FnOnce() -> PyResult<()>) -> c_int {
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            error.restore(unsafe { Python::assume_attached() });
            -1
        }
        Err(_) => {
            PyRuntimeError::new_err("Rust panic in DLPack C Exchange API callback")
                .restore(unsafe { Python::assume_attached() });
            -1
        }
    }
}

fn set_exchange_error(
    context: *mut c_void,
    callback: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char)>,
    message: &str,
) {
    let Some(callback) = callback else {
        return;
    };
    let message = CString::new(message.replace('\0', "\\0"))
        .expect("replacing NUL characters produces a valid CString");
    unsafe { callback(context, c"RuntimeError".as_ptr(), message.as_ptr()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocation::fixed::make_test_tensor,
        ffi::DLDataType,
        python::{ImportedDlpack, exchange::DlpackExchangeApiRef, from_dlpack},
    };
    use pyo3::{prelude::*, types::PyInt};
    use std::ffi::c_void;

    fn tensor() -> Managed<DLManagedTensorVersioned> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
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

    fn legacy_tensor() -> Managed<DLManagedTensor> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
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

    #[pyclass]
    struct TestProducer;

    unsafe impl DlpackExchangeProducer for TestProducer {
        fn managed_tensor_no_sync(
            &self,
            _py: Python<'_>,
        ) -> PyResult<Managed<DLManagedTensorVersioned>> {
            Ok(tensor())
        }

        fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<DLTensor> {
            static DATA: [i32; 3] = [1, 2, 3];
            static SHAPE: [i64; 1] = [3];
            static STRIDES: [i64; 1] = [1];
            Ok(DLTensor {
                data: DATA.as_ptr() as *mut c_void,
                device: DLDevice::CPU,
                ndim: 1,
                dtype: DLDataType::of::<i32>(),
                shape: SHAPE.as_ptr().cast_mut(),
                strides: STRIDES.as_ptr().cast_mut(),
                byte_offset: 0,
            })
        }

        fn current_work_stream(_py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void> {
            assert_eq!(device, DLDevice::CPU);
            Ok(ptr::null_mut())
        }

        fn from_managed_tensor_no_sync(
            py: Python<'_>,
            _tensor: Managed<DLManagedTensorVersioned>,
        ) -> PyResult<Py<Self>> {
            Py::new(py, Self)
        }
    }

    #[test]
    fn export_request_selects_abi_and_validates_zero_copy() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let legacy = ExportRequest::parse(None, None, None, None)?;
            assert_eq!(legacy.abi(), ExportAbi::Legacy);
            let capsule = legacy.export_zero_copy(
                py,
                DLDevice::CPU,
                DlpackFlags::empty(),
                || Ok(legacy_tensor()),
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
                        || Ok(tensor()),
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
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parses_cuda_stream_protocol_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let request = ExportRequest::parse(None, None, None, None)?;
            assert_eq!(request.cuda_stream()?, CudaStreamRequest::Unspecified);

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

    #[test]
    fn installs_working_exchange_api_on_pyclass_type() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            install_exchange_api::<TestProducer>(py)?;
            let object = Py::new(py, TestProducer)?.into_bound(py);
            let api = DlpackExchangeApiRef::from_object(object.as_any().as_borrowed())?.unwrap();

            api.with_dltensor_view_no_sync(object.as_any().as_borrowed(), |view| {
                assert_eq!(view.device, DLDevice::CPU);
                assert_eq!(view.ndim, 1);
            })?;
            assert!(api.current_work_stream(DLDevice::CPU)?.is_null());
            let managed =
                api.managed_tensor_from_py_object_no_sync(object.as_any().as_borrowed())?;
            assert_eq!(managed.validate().unwrap().shape(), &[3]);

            let converted = api.managed_tensor_to_py_object_no_sync(tensor(), py)?;
            assert!(converted.is_instance_of::<TestProducer>());
            Ok(())
        })
        .unwrap();
    }
}
