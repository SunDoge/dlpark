use std::ffi::{CStr, CString};

use crate::{
    dlpack::DLManagedTensor,
    tensor::{ManagedTensor, TensorWrapper, ToTensor},
};
use pyo3::{
    ffi::{PyCapsule_GetPointer, PyErr_Occurred, PyErr_Restore},
    prelude::*,
};

unsafe impl Send for DLManagedTensor {}

pub struct CapsuleContent<T> {
    pub value: T,
    pub name: CString,
}

impl DLManagedTensor {
    pub fn to_capsule(self) -> *mut pyo3::ffi::PyObject {
        let self_ptr = Box::into_raw(Box::new(self));
        let ptr = unsafe {
            pyo3::ffi::PyCapsule_New(
                self_ptr as *mut _,
                b"dltensor\0".as_ptr() as *const _,
                Some(dlpack_capsule_deleter),
            )
        };
        ptr
    }
}

impl<T: ToTensor> TensorWrapper<T> {
    pub fn to_capsule(self) -> *mut pyo3::ffi::PyObject {
        DLManagedTensor::from(self).to_capsule()
    }
}

unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(capsule, b"used_dltensor\0".as_ptr() as *const _) == 1 {
        return;
    }

    let mut exc_type = std::ptr::null_mut();
    let mut exc_value = std::ptr::null_mut();
    let mut exc_trace = std::ptr::null_mut();
    pyo3::ffi::PyErr_Fetch(&mut exc_type, &mut exc_value, &mut exc_trace);

    let managed =
        PyCapsule_GetPointer(capsule, b"dltensor\0".as_ptr() as *const _) as *mut DLManagedTensor;

    if managed.is_null() {
        pyo3::ffi::PyErr_WriteUnraisable(capsule);
        PyErr_Restore(exc_type, exc_value, exc_trace);
        return;
    }

    (*managed).deleter.map(|del_fn| {
        del_fn(managed);
        assert!(PyErr_Occurred().is_null());
    });

    PyErr_Restore(exc_type, exc_value, exc_trace);
}

#[pyclass]
pub struct PyManagedTensor {
    pub inner: Option<DLManagedTensor>,
}

impl From<DLManagedTensor> for PyManagedTensor {
    fn from(value: DLManagedTensor) -> Self {
        Self { inner: Some(value) }
    }
}

// #[pymethods]
// impl PyManagedTensor {
//     pub fn __dlpack__<'a>(&mut self, py: Python<'a>) -> PyResult<&'a PyCapsule> {
//         // self.inner.unwrap().map(|mt| mt.to_capsule(py)).transpose()
//         self.inner.unwrap().to_capsule(py)
//     }
// }