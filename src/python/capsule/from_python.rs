//! Converts Python DLPack capsules into owning Rust tensors.

use super::{
    DLTENSOR, DLTENSOR_VERSIONED, USED_DLTENSOR, USED_DLTENSOR_VERSIONED,
    ffi_error::{checked_fetch_python_error, fetch_python_error},
};
use crate::{
    DlpackFlags, Managed,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
};
use pyo3::{
    Borrowed, Bound, PyAny,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    types::{PyAnyMethods, PyDict, PyString},
};
use std::ffi::CStr;

fn set_capsule_name(capsule: *mut pyo3::ffi::PyObject, name: &CStr) -> pyo3::PyResult<()> {
    if unsafe { pyo3::ffi::PyCapsule_SetName(capsule, name.as_ptr()) } != 0 {
        Err(unsafe { fetch_python_error() })
    } else {
        Ok(())
    }
}

/// Marks an externally consumed DLPack capsule as used.
///
/// This is a recovery helper for a non-conforming consumer that invokes the
/// managed-tensor deleter, then raises without renaming the capsule. It accepts
/// both legacy and versioned DLPack capsules and is idempotent when the capsule
/// already has its corresponding `used_*` name.
///
/// This function does not invoke the managed-tensor deleter. The caller must
/// only use it after establishing that the consumer took ownership or already
/// invoked the deleter. Marking an unconsumed capsule as used prevents its
/// destructor from releasing the tensor and therefore leaks its allocation.
pub fn mark_dlpack_capsule_consumed(capsule: Borrowed<'_, '_, PyAny>) -> pyo3::PyResult<()> {
    let capsule = capsule.as_ptr();
    unsafe {
        for (name, used_name) in [
            (DLTENSOR, USED_DLTENSOR),
            (DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED),
        ] {
            if pyo3::ffi::PyCapsule_IsValid(capsule, used_name.as_ptr()) == 1 {
                return Ok(());
            }
            if pyo3::ffi::PyCapsule_IsValid(capsule, name.as_ptr()) == 1 {
                return set_capsule_name(capsule, used_name);
            }
        }
    }
    Err(PyValueError::new_err(
        "expected a legacy or versioned DLPack capsule",
    ))
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
            // PyCapsule_IsValid never sets a Python exception; any pending
            // error here is stale state from unrelated earlier code, so
            // always report the capsule problem itself.
            return Err(PyValueError::new_err(format!(
                "expected a PyCapsule named {:?}",
                name
            )));
        }

        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr());
        if ptr.is_null() {
            // A capsule may legally hold a NULL pointer, in which case
            // GetPointer returns NULL without setting an exception.
            return Err(checked_fetch_python_error(PyBufferError::new_err(
                "DLPack capsule contains a null pointer",
            )));
        }

        set_capsule_name(capsule, used_name)?;
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
