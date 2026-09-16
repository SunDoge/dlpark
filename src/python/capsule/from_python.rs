//! Converts Python DLPack capsules into owning Rust tensors.

use super::{DLTENSOR, DLTENSOR_VERSIONED, USED_DLTENSOR, USED_DLTENSOR_VERSIONED};
use crate::{
    DlpackFlags, Managed,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
};
use pyo3::{
    Borrowed, Bound, PyAny, PyErr,
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

pub(crate) fn is_dlpack_capsule<'py>(
    ob: Borrowed<'_, 'py, PyAny>,
    name: &CStr,
    used_name: &CStr,
) -> bool {
    unsafe {
        pyo3::ffi::PyCapsule_IsValid(ob.as_ptr(), name.as_ptr()) == 1
            || pyo3::ffi::PyCapsule_IsValid(ob.as_ptr(), used_name.as_ptr()) == 1
    }
}

pub(crate) fn is_legacy_capsule(ob: Borrowed<'_, '_, PyAny>) -> bool {
    is_dlpack_capsule(ob, DLTENSOR, USED_DLTENSOR)
}

pub(crate) fn is_versioned_capsule(ob: Borrowed<'_, '_, PyAny>) -> bool {
    is_dlpack_capsule(ob, DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED)
}

pub(crate) fn call_dlpack<'py>(
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

pub(crate) fn consume_legacy_capsule(
    capsule: Borrowed<'_, '_, PyAny>,
) -> pyo3::PyResult<Managed<DLManagedTensor>> {
    let ptr = capsule_to_raw_dlpack(capsule.as_ptr(), DLTENSOR, USED_DLTENSOR)?;
    unsafe { Managed::from_raw(ptr.cast()) }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub(crate) fn validate_copy_result(
    tensor: Managed<DLManagedTensorVersioned>,
    requested: Option<bool>,
) -> pyo3::PyResult<Managed<DLManagedTensorVersioned>> {
    let copied = tensor.flags().contains(DlpackFlags::IS_COPIED);
    match (requested, copied) {
        (Some(true), false) => Err(PyBufferError::new_err(
            "DLPack producer did not copy despite copy=True",
        )),
        (Some(false), true) => Err(PyBufferError::new_err(
            "DLPack producer copied despite copy=False",
        )),
        _ => Ok(tensor),
    }
}

pub(crate) fn consume_versioned_capsule(
    capsule: Borrowed<'_, '_, PyAny>,
) -> pyo3::PyResult<Managed<DLManagedTensorVersioned>> {
    let ptr = capsule_to_raw_dlpack(
        capsule.as_ptr(),
        DLTENSOR_VERSIONED,
        USED_DLTENSOR_VERSIONED,
    )?;
    unsafe { Managed::from_raw(ptr.cast()) }
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
