use pyo3::{
    Bound, PyAny,
    conversion::{FromPyObject, IntoPyObject},
};

use crate::{SafeManagedTensor, SafeManagedTensorVersioned, ffi};
use std::ffi::CStr;

const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";
const DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
const USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

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

        let _ = SafeManagedTensor::from_raw(ptr as *mut ffi::ManagedTensor);
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

        let _ = SafeManagedTensorVersioned::from_raw(ptr as *mut ffi::ManagedTensorVersioned);
    }
}

fn capsule_to_raw_dlpack(
    capsule: *mut pyo3::ffi::PyObject,
    name: &CStr,
    used_name: &CStr,
) -> *mut std::ffi::c_void {
    unsafe {
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr());
        pyo3::ffi::PyCapsule_SetName(capsule, used_name.as_ptr());
        ptr
    }
}

fn raw_dlpack_to_capsule(
    ptr: *mut std::ffi::c_void,
    name: &CStr,
    deleter: unsafe extern "C" fn(*mut pyo3::ffi::PyObject),
) -> *mut pyo3::ffi::PyObject {
    unsafe { pyo3::ffi::PyCapsule_New(ptr, name.as_ptr(), Some(deleter)) }
}

impl<'py> FromPyObject<'py> for SafeManagedTensor {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let ptr = capsule_to_raw_dlpack(ob.as_ptr(), DLTENSOR, USED_DLTENSOR);
        unsafe { Ok(SafeManagedTensor::from_raw(ptr as *mut _)) }
    }
}

impl<'py> IntoPyObject<'py> for SafeManagedTensor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let capsule =
                raw_dlpack_to_capsule(self.into_raw() as *mut _, DLTENSOR, dlpack_capsule_deleter);
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

impl<'py> FromPyObject<'py> for SafeManagedTensorVersioned {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let ptr = capsule_to_raw_dlpack(ob.as_ptr(), DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED);
        unsafe { Ok(SafeManagedTensorVersioned::from_raw(ptr as *mut _)) }
    }
}

impl<'py> IntoPyObject<'py> for SafeManagedTensorVersioned {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        unsafe {
            let capsule = raw_dlpack_to_capsule(
                self.into_raw() as *mut _,
                DLTENSOR_VERSIONED,
                dlpack_capsule_deleter_versioned,
            );
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}
