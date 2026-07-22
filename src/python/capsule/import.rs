use super::{DLTENSOR, DLTENSOR_VERSIONED, USED_DLTENSOR, USED_DLTENSOR_VERSIONED};
use crate::{
    Foreign,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    python::{DlpackStream, device::dlpack_device, exchange::DlpackExchangeApiRef},
};
use pyo3::{
    Borrowed, Bound, PyAny, PyErr,
    conversion::FromPyObject,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    types::{PyAnyMethods, PyDict, PyString},
};
use std::ffi::CStr;

fn fetch_python_error() -> PyErr {
    unsafe { PyErr::fetch(pyo3::Python::assume_attached()) }
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
        unsafe { Self::from_raw(ptr.cast()) }
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
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
        unsafe { Self::from_raw(ptr.cast()) }
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
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
        unsafe { Self::from_raw(ptr.cast()) }
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
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
