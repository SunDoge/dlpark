use pyo3::conversion::{FromPyObject, IntoPyObject};
use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyDict, PyString};
use pyo3::{Borrowed, Bound, PyAny, PyErr};
use std::ffi::CStr;

use crate::{
    Foreign, Local,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    python::{DlpackStream, device::dlpack_device, exchange::DlpackExchangeApiRef},
};

const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";
const DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
const USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

fn fetch_python_error() -> PyErr {
    // These helpers are only called while PyO3 is extracting or creating Python
    // objects, so the current thread is attached to the Python interpreter.
    unsafe { PyErr::fetch(pyo3::Python::assume_attached()) }
}

unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, USED_DLTENSOR.as_ptr()) == 1 {
            return;
        }

        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr());

        if ptr.is_null() {
            pyo3::ffi::PyErr_WriteUnraisable(capsule);
            return;
        }

        let _ = Foreign::<DLManagedTensor>::from_raw_unchecked(ptr as *mut _);
    }
}

unsafe extern "C" fn dlpack_capsule_deleter_versioned(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, USED_DLTENSOR_VERSIONED.as_ptr()) == 1 {
            return;
        }

        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED.as_ptr());

        if ptr.is_null() {
            pyo3::ffi::PyErr_WriteUnraisable(capsule);
            return;
        }

        let _ = Foreign::<DLManagedTensorVersioned>::from_raw_unchecked(ptr as *mut _);
    }
}

fn capsule_to_raw_dlpack(
    capsule: *mut pyo3::ffi::PyObject,
    name: &CStr,
    used_name: &CStr,
) -> pyo3::PyResult<*mut std::ffi::c_void> {
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, used_name.as_ptr()) == 1 {
            return Err(PyValueError::new_err(
                "DLPack capsule has already been consumed",
            ));
        }
        if pyo3::ffi::PyCapsule_IsValid(capsule, name.as_ptr()) != 1 {
            if pyo3::ffi::PyErr_Occurred().is_null() {
                return Err(PyValueError::new_err(format!(
                    "expected a PyCapsule named {:?}",
                    name
                )));
            }
            return Err(fetch_python_error());
        }

        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr());
        if ptr.is_null() {
            if pyo3::ffi::PyErr_Occurred().is_null() {
                return Err(PyBufferError::new_err(
                    "DLPack capsule contains a null pointer",
                ));
            }
            return Err(fetch_python_error());
        }

        if pyo3::ffi::PyCapsule_SetName(capsule, used_name.as_ptr()) != 0 {
            return Err(fetch_python_error());
        }
        Ok(ptr)
    }
}

fn is_dlpack_capsule<'py>(ob: Borrowed<'_, 'py, PyAny>, name: &CStr, used_name: &CStr) -> bool {
    unsafe {
        pyo3::ffi::PyCapsule_IsValid(ob.as_ptr(), name.as_ptr()) == 1
            || pyo3::ffi::PyCapsule_IsValid(ob.as_ptr(), used_name.as_ptr()) == 1
    }
}

fn call_dlpack<'py>(
    ob: Borrowed<'_, 'py, PyAny>,
    max_version: Option<(u32, u32)>,
    stream: Option<&Bound<'py, PyAny>>,
    copy: Option<bool>,
) -> pyo3::PyResult<Bound<'py, PyAny>> {
    if max_version.is_none() && stream.is_none() && copy.is_none() {
        return ob.call_method0(PyString::intern(ob.py(), "__dlpack__"));
    }

    let py = ob.py();
    let kwargs = PyDict::new(py);
    if let Some(max_version) = max_version {
        kwargs.set_item(PyString::intern(py, "max_version"), max_version)?;
    }
    if let Some(stream) = &stream {
        kwargs.set_item(PyString::intern(py, "stream"), stream)?;
    }
    if let Some(copy) = copy {
        kwargs.set_item(PyString::intern(py, "copy"), copy)?;
    }
    ob.call_method(PyString::intern(py, "__dlpack__"), (), Some(&kwargs))
}

fn raw_dlpack_to_capsule(
    ptr: *mut std::ffi::c_void,
    name: &CStr,
    deleter: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
) -> pyo3::PyResult<*mut pyo3::ffi::PyObject> {
    if ptr.is_null() {
        return Err(PyBufferError::new_err(
            "cannot create a DLPack capsule from a null pointer",
        ));
    }
    let capsule = unsafe { pyo3::ffi::PyCapsule_New(ptr, name.as_ptr(), Some(deleter)) };
    if capsule.is_null() {
        Err(fetch_python_error())
    } else {
        Ok(capsule)
    }
}

impl<'py> FromPyObject<'_, 'py> for Foreign<DLManagedTensor> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let owned_capsule;
        let capsule = if is_dlpack_capsule(ob, DLTENSOR, USED_DLTENSOR) {
            ob.as_ptr()
        } else {
            owned_capsule = call_dlpack(ob, None, None, None)?;
            owned_capsule.as_ptr()
        };
        let ptr = capsule_to_raw_dlpack(capsule, DLTENSOR, USED_DLTENSOR)?;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack capsule pointer is unexpectedly null",
            ));
        }
        unsafe { Ok(Self::from_raw_unchecked(ptr as *mut _)) }
    }
}

impl<'py> IntoPyObject<'py> for Foreign<DLManagedTensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let raw = self.into_raw();
            let capsule =
                match raw_dlpack_to_capsule(raw as *mut _, DLTENSOR, dlpack_capsule_deleter) {
                    Ok(capsule) => capsule,
                    Err(err) => {
                        let _ = Foreign::<DLManagedTensor>::from_raw_unchecked(raw);
                        return Err(err);
                    }
                };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

impl<'py> IntoPyObject<'py> for Local<DLManagedTensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let raw = self.into_raw();
            let capsule = match raw_dlpack_to_capsule(raw.cast(), DLTENSOR, dlpack_capsule_deleter)
            {
                Ok(capsule) => capsule,
                Err(err) => {
                    let _ = Foreign::<DLManagedTensor>::from_raw_unchecked(raw);
                    return Err(err);
                }
            };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for Foreign<DLManagedTensorVersioned> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Some(api) = DlpackExchangeApiRef::from_object(ob)? {
            return api.managed_tensor_from_py_object_no_sync(ob);
        }

        let owned_capsule;
        let capsule = if is_dlpack_capsule(ob, DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED) {
            ob.as_ptr()
        } else {
            owned_capsule = call_dlpack(
                ob,
                Some((
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION,
                )),
                None,
                None,
            )?;
            owned_capsule.as_ptr()
        };
        let ptr = capsule_to_raw_dlpack(capsule, DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED)?;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack capsule pointer is unexpectedly null",
            ));
        }
        unsafe { Ok(Self::from_raw_unchecked(ptr as *mut _)) }
    }
}

impl Foreign<DLManagedTensorVersioned> {
    /// Extracts a versioned DLPack tensor with optional stream and copy
    /// requests.
    pub fn extract_with_options(
        ob: Borrowed<'_, '_, PyAny>,
        stream: Option<&dyn DlpackStream>,
        copy: Option<bool>,
    ) -> pyo3::PyResult<Self> {
        if is_dlpack_capsule(ob, DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED) {
            return Err(PyValueError::new_err(
                "an existing DLPack capsule cannot negotiate stream or copy options",
            ));
        }

        let stream = match stream {
            Some(stream) => {
                let device = dlpack_device(ob)?;
                stream
                    .as_python_arg(ob.py(), device)?
                    .into_python(ob.py())?
            }
            None => None,
        };
        let capsule = call_dlpack(
            ob,
            Some((
                crate::ffi::DLPACK_MAJOR_VERSION,
                crate::ffi::DLPACK_MINOR_VERSION,
            )),
            stream.as_ref(),
            copy,
        )?;
        let ptr = capsule_to_raw_dlpack(
            capsule.as_ptr(),
            DLTENSOR_VERSIONED,
            USED_DLTENSOR_VERSIONED,
        )?;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack capsule pointer is unexpectedly null",
            ));
        }
        unsafe { Ok(Self::from_raw_unchecked(ptr.cast())) }
    }

    /// Extracts a versioned DLPack tensor using an explicit consumer stream.
    ///
    /// This follows the Python DLPack consumer protocol: it queries
    /// `__dlpack_device__()`, maps `stream` for that device, and passes the
    /// result to `__dlpack__(stream=..., max_version=..., copy=...)`.
    /// `copy` maps directly to Python's tri-state copy request.
    pub fn extract_with_stream<'py, S>(
        ob: Borrowed<'_, 'py, PyAny>,
        stream: &S,
        copy: Option<bool>,
    ) -> pyo3::PyResult<Self>
    where
        S: DlpackStream,
    {
        Self::extract_with_options(ob, Some(stream), copy)
    }
}

impl<'py> IntoPyObject<'py> for Foreign<DLManagedTensorVersioned> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let raw = self.into_raw();
            let capsule = match raw_dlpack_to_capsule(
                raw as *mut _,
                DLTENSOR_VERSIONED,
                dlpack_capsule_deleter_versioned,
            ) {
                Ok(capsule) => capsule,
                Err(err) => {
                    let _ = Foreign::<DLManagedTensorVersioned>::from_raw_unchecked(raw);
                    return Err(err);
                }
            };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

impl<'py> IntoPyObject<'py> for Local<DLManagedTensorVersioned> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let raw = self.into_raw();
            let capsule = match raw_dlpack_to_capsule(
                raw.cast(),
                DLTENSOR_VERSIONED,
                dlpack_capsule_deleter_versioned,
            ) {
                Ok(capsule) => capsule,
                Err(err) => {
                    let _ = Foreign::<DLManagedTensorVersioned>::from_raw_unchecked(raw);
                    return Err(err);
                }
            };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DlpackFlags;
    use crate::{
        ffi::{DLDataType, DLDevice, DLDeviceType},
        test_support::{fixed_local, fixed_tensor},
    };
    use pyo3::conversion::IntoPyObject;
    use pyo3::types::PyModule;
    use std::ffi::c_void;

    struct TestStream;

    unsafe impl DlpackStream for TestStream {
        fn as_python_arg(
            &self,
            _py: pyo3::Python<'_>,
            device: DLDevice,
        ) -> pyo3::PyResult<crate::python::StreamArg> {
            assert_eq!(device.device_type, DLDeviceType::CUDA);
            assert_eq!(device.device_id, 3);
            Ok(crate::python::stream::cuda(
                std::ptr::without_provenance_mut(42),
            ))
        }
    }

    fn legacy_tensor() -> Local<DLManagedTensor> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        fixed_local(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice::CPU,
            [3],
            [1],
            DlpackFlags::empty(),
        )
    }

    fn versioned_tensor() -> Local<DLManagedTensorVersioned> {
        let data = Box::new(vec![4i32, 5, 6]);
        let data_ptr = data.as_ptr() as *mut c_void;
        fixed_local(
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
    fn local_versioned_tensor_converts_to_capsule() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let data = Box::new(vec![4i32, 5, 6]);
            let data_ptr = data.as_ptr() as *mut c_void;
            let tensor: Local<DLManagedTensorVersioned> = fixed_tensor(
                data,
                data_ptr,
                DLDataType::of::<i32>(),
                DLDevice::CPU,
                [3],
                [1],
                DlpackFlags::empty(),
            );
            let capsule = tensor.into_pyobject(py)?;
            let tensor = Foreign::<DLManagedTensorVersioned>::extract(capsule.as_borrowed())?;
            assert_eq!(
                unsafe { tensor.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn legacy_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = legacy_tensor().into_pyobject(py)?;

            let dlpack = Foreign::<DLManagedTensor>::extract(capsule.as_borrowed())?;
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[1, 2, 3]
            );

            let err = match Foreign::<DLManagedTensor>::extract(capsule.as_borrowed()) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;

            let dlpack = Foreign::<DLManagedTensorVersioned>::extract(capsule.as_borrowed())?;
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );

            let err = match Foreign::<DLManagedTensorVersioned>::extract(capsule.as_borrowed()) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn legacy_extract_calls_dunder_dlpack_fallback() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = legacy_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self):
        return self.capsule
"#,
                c"producer.py",
                c"producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let dlpack = Foreign::<DLManagedTensor>::extract(producer.as_borrowed())?;
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[1, 2, 3]
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_calls_dunder_dlpack_with_max_version() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_max_version = None

    def __dlpack__(self, *, max_version=None):
        self.seen_max_version = max_version
        return self.capsule
"#,
                c"versioned_producer.py",
                c"versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let dlpack = Foreign::<DLManagedTensorVersioned>::extract(producer.as_borrowed())?;
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );
            assert_eq!(
                producer
                    .getattr("seen_max_version")?
                    .extract::<(u32, u32)>()?,
                (
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION
                )
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_with_stream_maps_device_and_passes_stream() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_stream = None
        self.seen_max_version = None
        self.seen_copy = None

    def __dlpack_device__(self):
        return (2, 3)

    def __dlpack__(self, *, stream=None, max_version=None, copy=None):
        self.seen_stream = stream
        self.seen_max_version = max_version
        self.seen_copy = copy
        return self.capsule
"#,
                c"stream_producer.py",
                c"stream_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let _dlpack = Foreign::<DLManagedTensorVersioned>::extract_with_stream(
                producer.as_borrowed(),
                &TestStream,
                Some(false),
            )?;
            assert_eq!(producer.getattr("seen_stream")?.extract::<usize>()?, 42);
            assert!(!producer.getattr("seen_copy")?.extract::<bool>()?);
            assert_eq!(
                producer
                    .getattr("seen_max_version")?
                    .extract::<(u32, u32)>()?,
                (
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION
                )
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_with_options_passes_copy_without_stream() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_copy = None

    def __dlpack__(self, *, max_version=None, copy=None):
        self.seen_copy = copy
        return self.capsule
"#,
                c"copy_producer.py",
                c"copy_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let _dlpack = Foreign::<DLManagedTensorVersioned>::extract_with_options(
                producer.as_borrowed(),
                None,
                Some(true),
            )?;
            assert!(producer.getattr("seen_copy")?.extract::<bool>()?);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_rejects_producers_without_version_negotiation() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self):
        self.calls = 0

    def __dlpack__(self):
        self.calls += 1
"#,
                c"legacy_producer.py",
                c"legacy_producer",
            )?;
            let producer = module.getattr("Producer")?.call0()?;

            let err = match Foreign::<DLManagedTensorVersioned>::extract(producer.as_borrowed()) {
                Ok(_) => panic!("versioned extraction requires max_version support"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert_eq!(producer.getattr("calls")?.extract::<u32>()?, 0);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_does_not_retry_internal_type_error() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self):
        self.calls = 0

    def __dlpack__(self, *, max_version=None):
        self.calls += 1
        raise TypeError("producer failed")
"#,
                c"broken_versioned_producer.py",
                c"broken_versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call0()?;

            let err = match Foreign::<DLManagedTensorVersioned>::extract(producer.as_borrowed()) {
                Ok(_) => panic!("producer TypeError should propagate"),
                Err(err) => err,
            };

            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert_eq!(producer.getattr("calls")?.extract::<u32>()?, 1);
            Ok(())
        })
        .unwrap();
    }
}
