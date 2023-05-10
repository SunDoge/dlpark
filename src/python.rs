use crate::{
    ffi::DLManagedTensor,
    tensor::traits::{HasByteOffset, HasData, HasDevice, HasDtype},
    tensor::{ManagedTensor, ManagerCtx},
};
use pyo3::{
    ffi::{PyCapsule_GetPointer, PyCapsule_New, PyCapsule_SetName, PyErr_Occurred, PyErr_Restore},
    prelude::*,
    IntoPy, PyAny, PyResult, Python,
};

impl DLManagedTensor {
    pub fn to_capsule_ptr(self) -> *mut pyo3::ffi::PyObject {
        let self_ptr = Box::into_raw(Box::new(self));

        unsafe {
            PyCapsule_New(
                self_ptr as *mut _,
                b"dltensor\0".as_ptr() as *const _,
                Some(dlpack_capsule_deleter),
            )
        }
    }

    pub fn to_capsule(self, py: Python<'_>) -> PyResult<&PyAny> {
        let ptr = self.to_capsule_ptr();
        unsafe { py.from_owned_ptr_or_err(ptr) }
    }
}

impl<T> ManagerCtx<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    pub fn to_capsule(self, py: Python<'_>) -> PyResult<&PyAny> {
        DLManagedTensor::from(self).to_capsule(py)
    }

    pub fn to_capsule_ptr(self) -> *mut pyo3::ffi::PyObject {
        DLManagedTensor::from(self).to_capsule_ptr()
    }
}

/// Refer to [dlpack python_spec](https://dmlc.github.io/dlpack/latest/python_spec.html#implementation)
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
        let ptr = self.to_capsule_ptr();
        unsafe { PyObject::from_owned_ptr(py, ptr) }
    }
}

// impl<T> ToPyObject for TensorWrapper<T>
//  where T: HasData + HasDevice + HasDtype + HasByteOffset {
//     fn to_object(&self, py: Python<'_>) -> PyObject {

//     }
// }

// impl<'source, T> FromPyObject<'source> for ManagerCtx<T> {
//     fn extract(ob: &'source PyAny) -> PyResult<Self> {}
// }

/// Check this [pytorch src](https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp#L1583)
impl From<*mut pyo3::ffi::PyObject> for ManagedTensor {
    fn from(capsule: *mut pyo3::ffi::PyObject) -> Self {
        unsafe {
            let dl_managed_tensor =
                PyCapsule_GetPointer(capsule, b"dltensor\0".as_ptr() as *const _)
                    as *mut DLManagedTensor;

            // TODO: we should add a flag for buggy numpy dlpack deleter
            // let deleter_with_gil = move |_| {
            //     if let Some(del_fn) = (*dl_managed_tensor).deleter {
            //         Python::with_gil(move |_py| {
            //             del_fn(dl_managed_tensor);
            //         });
            //     }
            // };

            PyCapsule_SetName(capsule, b"used_dltensor\0".as_ptr() as *const _);

            ManagedTensor {
                inner: dl_managed_tensor,
                deleter: None,
            }
        }
    }
}

impl<'source> FromPyObject<'source> for ManagedTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(ManagedTensor::from(ob.into_ptr()))
    }
}
