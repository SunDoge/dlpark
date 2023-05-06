use std::ffi::CString;

use crate::{dlpack::DLManagedTensor, tensor::ManagedTensor};
use pyo3::{ffi::PyCapsule_New, ffi::PyObject, prelude::*, types::PyCapsule};

unsafe impl Send for DLManagedTensor {}

// const DLTENSOR: CString = CString::new("dltensor").unwrap();

impl DLManagedTensor {
    pub fn to_capsule<'a>(self, py: Python<'a>) -> PyResult<&'a PyCapsule> {
        let name = CString::new("dltensor").unwrap();
        PyCapsule::new(py, self, Some(name))
    }
}

#[pyclass]
pub struct PyManagedTensor {
    pub inner: Option<DLManagedTensor>,
}

#[pymethods]
impl PyManagedTensor {
    pub fn __dlpack__<'a>(&mut self, py: Python<'a>) -> PyResult<Option<&'a PyCapsule>> {
        self.inner.take().map(|mt| mt.to_capsule(py)).transpose()
    }
}
