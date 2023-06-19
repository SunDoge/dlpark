use std::ptr::NonNull;

use crate::{
    ffi,
    tensor::traits::{HasByteOffset, HasData, HasDevice, HasDtype, ToDLPack},
    tensor::{ManagedTensor, ManagerCtx},
};
use pyo3::{
    ffi::{PyCapsule_GetPointer, PyCapsule_New, PyCapsule_SetName, PyErr_Occurred, PyErr_Restore},
    prelude::*,
    IntoPy, PyAny, PyResult, Python,
};

/// The producer must set the PyCapsule name to "dltensor" so that it can be inspected by name,
/// and set PyCapsule_Destructor that calls the deleter of the ffi::DLManagedTensor
/// when the "dltensor"-named capsule is no longer needed.
const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";

/// The consumer must transer ownership of the DLManangedTensor from the capsule
/// to its own object. It does so by renaming the capsule to "used_dltensor"
/// to ensure that PyCapsule_Destructor will not get called
/// (ensured if PyCapsule_Destructor calls deleter only for capsules whose name is "dltensor")
const DLPACK_CAPSULE_USED_NAME: &[u8] = b"used_dltensor\0";

impl IntoPyPointer for ffi::DLManagedTensor {
    fn into_ptr(self) -> *mut pyo3::ffi::PyObject {
        let self_ = Box::new(self);
        self_.into_ptr()
    }
}

impl IntoPyPointer for Box<ffi::DLManagedTensor> {
    fn into_ptr(self) -> *mut pyo3::ffi::PyObject {
        unsafe {
            PyCapsule_New(
                Box::into_raw(self).cast(),
                DLPACK_CAPSULE_NAME.as_ptr().cast(),
                Some(dlpack_capsule_deleter),
            )
        }
    }
}

impl IntoPy<PyObject> for ffi::DLManagedTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ptr = self.into_ptr();
        unsafe { PyObject::from_owned_ptr(py, ptr) }
    }
}

pub fn new_py_capsule(dl_managed_tensor: *mut ffi::DLManagedTensor) -> *mut pyo3::ffi::PyObject {
    unsafe {
        PyCapsule_New(
            dl_managed_tensor.cast(),
            DLPACK_CAPSULE_NAME.as_ptr().cast(),
            Some(dlpack_capsule_deleter),
        )
    }
}

/// Refer to [dlpack python_spec](https://dmlc.github.io/dlpack/latest/python_spec.html#implementation)
unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    if pyo3::ffi::PyCapsule_IsValid(capsule, DLPACK_CAPSULE_USED_NAME.as_ptr() as *const _) == 1 {
        return;
    }

    let mut exc_type = std::ptr::null_mut();
    let mut exc_value = std::ptr::null_mut();
    let mut exc_trace = std::ptr::null_mut();
    pyo3::ffi::PyErr_Fetch(&mut exc_type, &mut exc_value, &mut exc_trace);

    let managed = PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const _)
        as *mut ffi::DLManagedTensor;

    if managed.is_null() {
        pyo3::ffi::PyErr_WriteUnraisable(capsule);
        PyErr_Restore(exc_type, exc_value, exc_trace);
        return;
    }

    if let Some(del_fn) = (*managed).deleter {
        del_fn(managed);
        assert!(PyErr_Occurred().is_null());
    }

    PyErr_Restore(exc_type, exc_value, exc_trace);
}

impl<T> IntoPy<PyObject> for ManagerCtx<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let tensor = self.to_dlpack();
        let ptr = new_py_capsule(tensor.as_ptr());
        unsafe { PyObject::from_owned_ptr(py, ptr) }
    }
}

impl ManagedTensor {
    /// Check this [pytorch src](https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L1583)
    /// # Safety
    /// We use pyo3 ffi here.
    pub unsafe fn from_py_ptr(capsule: *mut pyo3::ffi::PyObject) -> Self {
        let dl_managed_tensor =
            PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr().cast()).cast();

        PyCapsule_SetName(capsule, DLPACK_CAPSULE_USED_NAME.as_ptr().cast());

        ManagedTensor::new(NonNull::new_unchecked(dl_managed_tensor))
    }

    pub fn from_py(ob: impl IntoPyPointer) -> Self {
        unsafe { Self::from_py_ptr(ob.into_ptr()) }
    }
}

impl<'source> FromPyObject<'source> for ManagedTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(ManagedTensor::from_py(ob))
    }
}

impl IntoPy<PyObject> for ManagedTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let tensor = self.into_inner();
        let ptr = new_py_capsule(tensor.as_ptr());
        unsafe { PyObject::from_owned_ptr(py, ptr) }
    }
}
