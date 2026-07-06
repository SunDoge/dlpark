use pyo3::conversion::{FromPyObject, IntoPyObject};
use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyDict};
use pyo3::{Borrowed, Bound, PyAny, PyErr, Python};
use std::ffi::CStr;

use crate::{
    ManagedBox,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    interop::python_exchange::DlpackExchangeApiRef,
};

const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";
const DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
const USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

fn fetch_python_error() -> PyErr {
    // These helpers are only called while PyO3 is extracting or creating Python
    // objects, so the current thread is attached to the Python interpreter.
    unsafe { PyErr::fetch(pyo3::Python::assume_attached()) }
}

fn attached_python() -> Python<'static> {
    // These helpers run while PyO3 is converting Python objects, so the current
    // thread is attached to the Python interpreter.
    unsafe { pyo3::Python::assume_attached() }
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

        let _ = ManagedBox::<DLManagedTensor>::new_unchecked(ptr as *mut _);
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

        let _ = ManagedBox::<DLManagedTensorVersioned>::new_unchecked(ptr as *mut _);
    }
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
) -> pyo3::PyResult<Bound<'py, PyAny>> {
    let Some(max_version) = max_version else {
        return ob.call_method0("__dlpack__");
    };

    let py = attached_python();
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_version", max_version)?;
    match ob.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(capsule) => Ok(capsule),
        Err(err) if err.is_instance_of::<PyTypeError>(py) => ob.call_method0("__dlpack__"),
        Err(err) => Err(err),
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

impl<'py> FromPyObject<'_, 'py> for ManagedBox<DLManagedTensor> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let owned_capsule;
        let capsule = if is_dlpack_capsule(ob, DLTENSOR, USED_DLTENSOR) {
            ob.as_ptr()
        } else {
            owned_capsule = call_dlpack(ob, None)?;
            owned_capsule.as_ptr()
        };
        let ptr = capsule_to_raw_dlpack(capsule, DLTENSOR, USED_DLTENSOR)?;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack capsule pointer is unexpectedly null",
            ));
        }
        unsafe { Ok(Self::new_unchecked(ptr as *mut _)) }
    }
}

impl<'py> IntoPyObject<'py> for ManagedBox<DLManagedTensor> {
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
                        let _ = ManagedBox::<DLManagedTensor>::new_unchecked(raw);
                        return Err(err);
                    }
                };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for ManagedBox<DLManagedTensorVersioned> {
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
            )?;
            owned_capsule.as_ptr()
        };
        let ptr = capsule_to_raw_dlpack(capsule, DLTENSOR_VERSIONED, USED_DLTENSOR_VERSIONED)?;
        if ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack capsule pointer is unexpectedly null",
            ));
        }
        unsafe { Ok(Self::new_unchecked(ptr as *mut _)) }
    }
}

impl<'py> IntoPyObject<'py> for ManagedBox<DLManagedTensorVersioned> {
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
                    let _ = ManagedBox::<DLManagedTensorVersioned>::new_unchecked(raw);
                    return Err(err);
                }
            };
            Bound::from_owned_ptr_or_err(py, capsule)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DlpackBuilder, ffi::DLDataType};
    use pyo3::conversion::IntoPyObject;
    use pyo3::types::PyModule;
    use std::ffi::c_void;

    fn legacy_tensor() -> ManagedBox<DLManagedTensor> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        DlpackBuilder::<DLManagedTensor, 1>::with_array_layout(data, [3i64], [1i64])
            .data(data_ptr)
            .dtype(DLDataType::of::<i32>())
            .build()
    }

    fn versioned_tensor() -> ManagedBox<DLManagedTensorVersioned> {
        let data = Box::new(vec![4i32, 5, 6]);
        let data_ptr = data.as_ptr() as *mut c_void;
        DlpackBuilder::<DLManagedTensorVersioned, 1>::with_array_layout(data, [3i64], [1i64])
            .data(data_ptr)
            .dtype(DLDataType::of::<i32>())
            .build()
    }

    #[test]
    fn legacy_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = legacy_tensor().into_pyobject(py)?;

            let dlpack = ManagedBox::<DLManagedTensor>::extract(capsule.as_borrowed())?;
            assert_eq!(
                dlpack.dl_tensor().cpu_data_slice::<i32>().unwrap(),
                &[1, 2, 3]
            );

            let err = match ManagedBox::<DLManagedTensor>::extract(capsule.as_borrowed()) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;

            let dlpack = ManagedBox::<DLManagedTensorVersioned>::extract(capsule.as_borrowed())?;
            assert_eq!(
                dlpack.dl_tensor().cpu_data_slice::<i32>().unwrap(),
                &[4, 5, 6]
            );

            let err = match ManagedBox::<DLManagedTensorVersioned>::extract(capsule.as_borrowed()) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn legacy_extract_calls_dunder_dlpack_fallback() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = legacy_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                c"class Producer:\n    def __init__(self, capsule):\n        self.capsule = capsule\n    def __dlpack__(self):\n        return self.capsule\n",
                c"producer.py",
                c"producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let dlpack = ManagedBox::<DLManagedTensor>::extract(producer.as_borrowed())?;
            assert_eq!(dlpack.dl_tensor().cpu_data_slice::<i32>().unwrap(), &[1, 2, 3]);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_calls_dunder_dlpack_with_max_version() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                c"class Producer:\n    def __init__(self, capsule):\n        self.capsule = capsule\n        self.seen_max_version = None\n    def __dlpack__(self, *, max_version=None):\n        self.seen_max_version = max_version\n        return self.capsule\n",
                c"versioned_producer.py",
                c"versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let dlpack = ManagedBox::<DLManagedTensorVersioned>::extract(producer.as_borrowed())?;
            assert_eq!(dlpack.dl_tensor().cpu_data_slice::<i32>().unwrap(), &[4, 5, 6]);
            assert_eq!(
                producer.getattr("seen_max_version")?.extract::<(u32, u32)>()?,
                (
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION
                )
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_extract_falls_back_to_no_arg_dunder_dlpack_on_type_error() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                c"class Producer:\n    def __init__(self, capsule):\n        self.capsule = capsule\n        self.calls = 0\n    def __dlpack__(self):\n        self.calls += 1\n        return self.capsule\n",
                c"old_versioned_producer.py",
                c"old_versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let dlpack = ManagedBox::<DLManagedTensorVersioned>::extract(producer.as_borrowed())?;
            assert_eq!(dlpack.dl_tensor().cpu_data_slice::<i32>().unwrap(), &[4, 5, 6]);
            assert_eq!(producer.getattr("calls")?.extract::<u32>()?, 1);

            Ok(())
        })
        .unwrap();
    }
}
