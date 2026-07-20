use pyo3::exceptions::{PyAttributeError, PyBufferError, PyRuntimeError};
use pyo3::{Borrowed, Bound, PyAny, PyErr, PyTypeInfo, Python};
use std::ffi::CStr;
use std::ptr::NonNull;

use crate::{
    ManagedBox,
    ffi::{
        DLDevice, DLManagedTensorVersioned, DLPACK_MAJOR_VERSION, DLPackExchangeAPI,
        DLPackExchangeAPIHeader, DLTensor,
    },
};

const DLPACK_EXCHANGE_API: &CStr = c"dlpack_exchange_api";

/// Borrowed reference to a producer's `DLPackExchangeAPI` function table.
///
/// The table is owned by the producer framework and must remain alive for the
/// process lifetime per the DLPack spec.
pub struct DlpackExchangeApiRef {
    api: NonNull<DLPackExchangeAPI>,
}

impl DlpackExchangeApiRef {
    pub fn from_object(obj: Borrowed<'_, '_, PyAny>) -> pyo3::PyResult<Option<Self>> {
        let capsule = unsafe {
            let ty = pyo3::ffi::Py_TYPE(obj.as_ptr()) as *mut pyo3::ffi::PyObject;
            let attr = pyo3::intern!(obj.py(), "__dlpack_c_exchange_api__");
            let capsule = pyo3::ffi::PyObject_GetAttr(ty, attr.as_ptr());
            if capsule.is_null() {
                let attr_error =
                    PyAttributeError::type_object_raw(pyo3::Python::assume_attached()).cast();
                if pyo3::ffi::PyErr_ExceptionMatches(attr_error) != 0 {
                    pyo3::ffi::PyErr_Clear();
                    return Ok(None);
                }
                return Err(fetch_python_error());
            }
            capsule
        };

        let api_ptr = unsafe {
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLPACK_EXCHANGE_API.as_ptr());
            pyo3::ffi::Py_DecRef(capsule);
            if ptr.is_null() {
                return Err(fetch_python_error());
            }
            ptr.cast::<DLPackExchangeAPI>()
        };

        let Some(api) = compatible_api(api_ptr) else {
            return Err(PyRuntimeError::new_err(
                "no compatible DLPackExchangeAPI version found",
            ));
        };

        Ok(Some(Self { api }))
    }

    /// Converts a Python tensor object into an owning versioned DLPack tensor.
    ///
    /// This does not perform stream synchronization. Consumers running kernels
    /// should also query [`Self::current_work_stream`] for the tensor device and
    /// launch work on the producer's stream.
    pub fn managed_tensor_from_py_object_no_sync(
        &self,
        obj: Borrowed<'_, '_, PyAny>,
    ) -> pyo3::PyResult<ManagedBox<DLManagedTensorVersioned>> {
        let api = unsafe { self.api.as_ref() };
        let Some(from_py_object) = api.managed_tensor_from_py_object_no_sync else {
            return Err(PyRuntimeError::new_err(
                "DLPackExchangeAPI managed_tensor_from_py_object_no_sync is null",
            ));
        };

        let mut out = std::ptr::null_mut();
        let rc = unsafe { from_py_object(obj.as_ptr().cast(), &mut out) };
        if rc != 0 {
            return Err(fetch_python_error());
        }
        if out.is_null() {
            return Err(PyBufferError::new_err(
                "DLPackExchangeAPI returned a null managed tensor",
            ));
        }

        Ok(unsafe { ManagedBox::new_unchecked(out) })
    }

    /// Transfers an owning managed tensor directly into a Python tensor
    /// without creating an intermediate DLPack capsule.
    pub fn managed_tensor_to_py_object_no_sync<'py>(
        &self,
        tensor: ManagedBox<DLManagedTensorVersioned>,
        py: Python<'py>,
    ) -> pyo3::PyResult<Bound<'py, PyAny>> {
        let api = unsafe { self.api.as_ref() };
        let Some(to_py_object) = api.managed_tensor_to_py_object_no_sync else {
            return Err(PyRuntimeError::new_err(
                "DLPackExchangeAPI managed_tensor_to_py_object_no_sync is null",
            ));
        };

        let raw = tensor.into_raw();
        let mut out = std::ptr::null_mut();
        let rc = unsafe { to_py_object(raw, &mut out) };
        if rc != 0 {
            return Err(fetch_python_error());
        }
        if out.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPackExchangeAPI returned a null Python object",
            ));
        }

        unsafe { Bound::from_owned_ptr_or_err(py, out.cast()) }
    }

    /// Returns the producer's current work stream for `device`.
    ///
    /// CPU producers may return null, which means no stream handling is needed.
    pub fn current_work_stream(&self, device: DLDevice) -> pyo3::PyResult<*mut std::ffi::c_void> {
        let api = unsafe { self.api.as_ref() };
        let Some(current_work_stream) = api.current_work_stream else {
            return Err(PyRuntimeError::new_err(
                "DLPackExchangeAPI current_work_stream is null",
            ));
        };

        let mut stream = std::ptr::null_mut();
        let rc = unsafe { current_work_stream(device.device_type, device.device_id, &mut stream) };
        if rc != 0 {
            return Err(fetch_python_error());
        }
        Ok(stream)
    }

    /// Borrows a temporary non-owning `DLTensor` view from the producer.
    ///
    /// The producer owns the shape, strides, and data pointers. The view is only
    /// valid during the callback and must not be stored or wrapped as an
    /// owning managed tensor.
    pub fn with_dltensor_view_no_sync<R>(
        &self,
        obj: Borrowed<'_, '_, PyAny>,
        f: impl FnOnce(&DLTensor) -> R,
    ) -> pyo3::PyResult<R> {
        let api = unsafe { self.api.as_ref() };
        let Some(from_py_object) = api.dltensor_from_py_object_no_sync else {
            return Err(PyRuntimeError::new_err(
                "DLPackExchangeAPI dltensor_from_py_object_no_sync is null",
            ));
        };

        let mut tensor = DLTensor::default();
        let rc = unsafe { from_py_object(obj.as_ptr().cast(), &mut tensor) };
        if rc != 0 {
            return Err(fetch_python_error());
        }

        Ok(f(&tensor))
    }
}

/// Walks the `prev_api` chain to find a header whose major version matches.
///
/// # Safety assumption
///
/// Per the DLPack spec the chain is made of full `DLPackExchangeAPI` tables
/// (each beginning with a `DLPackExchangeAPIHeader`), published by the
/// producer framework and alive for the process lifetime. We therefore read
/// only the header fields while walking, then — on a major-version match —
/// cast back to `*mut DLPackExchangeAPI` and let the caller read the
/// function-pointer fields that follow the header. A producer that violated
/// the spec by chaining a bare 16-byte header would make those downstream
/// reads out of bounds; the spec's "framework-owned static table" contract is
/// what rules that out.
fn compatible_api(api: *mut DLPackExchangeAPI) -> Option<NonNull<DLPackExchangeAPI>> {
    let mut header = api.cast::<DLPackExchangeAPIHeader>();
    while let Some(current_header) = NonNull::new(header) {
        let current = unsafe { current_header.as_ref() };
        if current.version.major == DLPACK_MAJOR_VERSION {
            return NonNull::new(header.cast::<DLPackExchangeAPI>());
        }
        header = current.prev_api;
    }
    None
}

fn fetch_python_error() -> PyErr {
    // Exchange API calls are made from PyO3 conversion code, so the current
    // thread is attached to the Python interpreter.
    unsafe { PyErr::fetch(pyo3::Python::assume_attached()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Builder, ManagedTensorBase,
        ffi::{DLDataType, DLDevice, DLDeviceType, DLPACK_MINOR_VERSION, DLPackVersion},
    };
    use pyo3::conversion::FromPyObject;
    use pyo3::types::{PyAnyMethods, PyModule};
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" fn mock_allocator(
        _prototype: *mut DLTensor,
        _out: *mut *mut DLManagedTensorVersioned,
        _error_ctx: *mut c_void,
        _set_error: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char)>,
    ) -> c_int {
        -1
    }

    unsafe extern "C" fn mock_managed_from_py_object(
        _py_object: *mut c_void,
        out: *mut *mut DLManagedTensorVersioned,
    ) -> c_int {
        let data = Box::new(vec![7i32, 8, 9]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let raw = Builder::new(data, crate::metadata::CopiedArray::new([3i64], [1i64]))
            .data(data_ptr)
            .dtype(DLDataType::of::<i32>())
            .build_raw::<DLManagedTensorVersioned>();

        unsafe {
            *out = raw;
        }
        0
    }

    static VIEW_DATA: [i32; 3] = [1, 2, 3];
    static VIEW_SHAPE: [i64; 1] = [3];
    static VIEW_STRIDES: [i64; 1] = [1];

    unsafe extern "C" fn mock_dltensor_from_py_object(
        _py_object: *mut c_void,
        out: *mut DLTensor,
    ) -> c_int {
        unsafe {
            *out = DLTensor {
                data: VIEW_DATA.as_ptr() as *mut c_void,
                device: DLDevice::CPU,
                ndim: 1,
                dtype: DLDataType::of::<i32>(),
                shape: VIEW_SHAPE.as_ptr() as *mut i64,
                strides: VIEW_STRIDES.as_ptr() as *mut i64,
                byte_offset: 0,
            };
        }
        0
    }

    unsafe extern "C" fn mock_current_work_stream(
        device_type: DLDeviceType,
        _device_id: i32,
        out_current_stream: *mut *mut c_void,
    ) -> c_int {
        unsafe {
            *out_current_stream = if device_type == DLDeviceType::CPU {
                std::ptr::null_mut()
            } else {
                std::ptr::dangling_mut::<c_void>()
            };
        }
        0
    }

    unsafe extern "C" fn mock_tensor_to_py_object(
        tensor: *mut DLManagedTensorVersioned,
        out_py_object: *mut *mut c_void,
    ) -> c_int {
        unsafe {
            DLManagedTensorVersioned::drop_raw(tensor);
            *out_py_object = pyo3::ffi::PyLong_FromLong(42).cast();
        }
        0
    }

    fn leak_mock_api() -> *mut DLPackExchangeAPI {
        Box::leak(Box::new(DLPackExchangeAPI {
            header: DLPackExchangeAPIHeader {
                version: DLPackVersion {
                    major: DLPACK_MAJOR_VERSION,
                    minor: DLPACK_MINOR_VERSION,
                },
                prev_api: std::ptr::null_mut(),
            },
            managed_tensor_allocator: Some(mock_allocator),
            managed_tensor_from_py_object_no_sync: Some(mock_managed_from_py_object),
            managed_tensor_to_py_object_no_sync: Some(mock_tensor_to_py_object),
            dltensor_from_py_object_no_sync: Some(mock_dltensor_from_py_object),
            current_work_stream: Some(mock_current_work_stream),
        }))
    }

    #[test]
    fn exchange_api_fast_path_extracts_versioned_tensor() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class MockTensor:
    pass
"#,
                c"mock_tensor.py",
                c"mock_tensor",
            )?;
            let cls = module.getattr("MockTensor")?;
            let obj = cls.call0()?;

            let api = leak_mock_api();
            let capsule =
                unsafe { pyo3::ffi::PyCapsule_New(api.cast(), DLPACK_EXCHANGE_API.as_ptr(), None) };
            let capsule = unsafe { pyo3::Bound::from_owned_ptr(py, capsule) };
            cls.setattr("__dlpack_c_exchange_api__", capsule)?;

            let api_ref = DlpackExchangeApiRef::from_object(obj.as_borrowed())?.unwrap();
            assert!(api_ref.current_work_stream(DLDevice::CPU)?.is_null());
            api_ref.with_dltensor_view_no_sync(obj.as_borrowed(), |tensor| {
                assert_eq!(tensor.ndim, 1);
                assert_eq!(tensor.num_elements().unwrap(), 3);
            })?;

            let dlpack = ManagedBox::<DLManagedTensorVersioned>::extract(obj.as_borrowed())?;
            let tensor = dlpack.tensor();
            assert_eq!(tensor.ndim, 1);
            assert_eq!(tensor.shape().unwrap(), &[3]);
            assert_eq!(tensor.cpu_data_slice::<i32>().unwrap(), &[7, 8, 9]);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn exchange_api_lookup_preserves_non_attribute_errors() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class BrokenDescriptor:
    def __get__(self, instance, owner):
        raise RuntimeError("boom")


class MockTensor:
    __dlpack_c_exchange_api__ = BrokenDescriptor()
"#,
                c"broken_exchange.py",
                c"broken_exchange",
            )?;
            let obj = module.getattr("MockTensor")?.call0()?;

            let err = match DlpackExchangeApiRef::from_object(obj.as_borrowed()) {
                Ok(_) => panic!("non-AttributeError exchange API lookup failure must propagate"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyRuntimeError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn exchange_api_exports_without_capsule() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let api = NonNull::new(leak_mock_api()).unwrap();
            let api = DlpackExchangeApiRef { api };
            let data = Box::new(vec![7i32, 8, 9]);
            let data_ptr = data.as_ptr() as *mut c_void;
            let tensor = Builder::new(data, crate::metadata::CopiedArray::new([3i64], [1i64]))
                .data(data_ptr)
                .dtype(DLDataType::of::<i32>())
                .build::<DLManagedTensorVersioned>();

            let object = api.managed_tensor_to_py_object_no_sync(tensor, py)?;

            assert_eq!(object.extract::<i64>()?, 42);
            Ok(())
        })
        .unwrap();
    }
}
