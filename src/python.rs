use crate::{
    dlpack::DLManagedTensor,
    tensor::{HasByteOffset, HasData, HasDevice, HasDtype, TensorWrapper},
};
use pyo3::ffi::{PyCapsule_GetPointer, PyCapsule_New, PyErr_Occurred, PyErr_Restore};

impl DLManagedTensor {
    pub fn to_capsule(self) -> *mut pyo3::ffi::PyObject {
        let self_ptr = Box::into_raw(Box::new(self));

        unsafe {
            PyCapsule_New(
                self_ptr as *mut _,
                b"dltensor\0".as_ptr() as *const _,
                Some(dlpack_capsule_deleter),
            )
        }
    }
}

impl<T> TensorWrapper<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    pub fn to_capsule(self) -> *mut pyo3::ffi::PyObject {
        DLManagedTensor::from(self).to_capsule()
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
