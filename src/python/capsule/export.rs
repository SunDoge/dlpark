use super::{DLTENSOR, DLTENSOR_VERSIONED, USED_DLTENSOR, USED_DLTENSOR_VERSIONED};
use crate::{
    Foreign, Local,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
};
use pyo3::{Bound, PyAny, PyErr, conversion::IntoPyObject, exceptions::PyBufferError};
use std::ffi::CStr;

fn fetch_python_error() -> PyErr {
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
