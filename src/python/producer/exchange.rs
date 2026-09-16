//! DLPack 1.3 C Exchange API support for Python producers.

use crate::{
    Managed,
    ffi::{
        DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPACK_MAJOR_VERSION,
        DLPACK_MINOR_VERSION, DLPackExchangeAPI, DLPackExchangeAPIHeader, DLPackVersion, DLTensor,
    },
};
use pyo3::{
    Bound, Py, PyClass, PyRef, PyResult, Python,
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    types::PyAnyMethods,
};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

const DLPACK_EXCHANGE_API: &CStr = c"dlpack_exchange_api";

/// Rust implementation behind a PyO3 class's DLPack C Exchange API table.
///
/// The callbacks are no-sync by definition.
///
/// # Safety
///
/// When [`Self::HAS_DLTENSOR_VIEW`] is true, `tensor_view_no_sync` must return
/// data, shape, and stride pointers that stay valid and immutable until that
/// callback returns. Managed tensors must obey DLPack's ownership contract,
/// and `current_work_stream` must return a live backend stream for the requested
/// device.
pub unsafe trait DlpackExchangeProducer: PyClass {
    /// Whether to publish the optional borrowed `DLTensor` callback.
    const HAS_DLTENSOR_VIEW: bool = false;

    /// Creates a fresh owning versioned managed tensor without synchronization.
    fn managed_tensor_no_sync(&self, py: Python<'_>)
    -> PyResult<Managed<DLManagedTensorVersioned>>;

    /// Returns a temporary borrowed tensor descriptor without synchronization.
    ///
    /// This is called only when [`Self::HAS_DLTENSOR_VIEW`] is true.
    fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<DLTensor> {
        Err(PyNotImplementedError::new_err(
            "borrowed DLTensor export is not supported",
        ))
    }

    /// Returns the producer's current native work stream for `device`.
    fn current_work_stream(py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void>;

    /// Allocates a new tensor from a prototype descriptor.
    fn allocate(_prototype: &DLTensor) -> Result<Managed<DLManagedTensorVersioned>, String> {
        Err("DLPack managed-tensor allocation is not supported".into())
    }

    /// Converts an incoming versioned managed tensor into this Python class.
    fn from_managed_tensor_no_sync(
        _py: Python<'_>,
        _tensor: Managed<DLManagedTensorVersioned>,
    ) -> PyResult<Py<Self>> {
        Err(PyNotImplementedError::new_err(
            "DLPack managed-tensor import is not supported",
        ))
    }
}

/// Installs a process-lifetime DLPack C Exchange API capsule on a PyO3 class.
///
/// Call this once while initializing the extension module, after registering
/// the class with `module.add_class::<T>()`.
pub fn install_exchange_api<T>(py: Python<'_>) -> PyResult<()>
where
    T: DlpackExchangeProducer,
{
    let api = Box::leak(Box::new(DLPackExchangeAPI {
        header: DLPackExchangeAPIHeader {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            prev_api: ptr::null_mut(),
        },
        managed_tensor_allocator: Some(allocate::<T>),
        managed_tensor_from_py_object_no_sync: Some(managed_from_python::<T>),
        managed_tensor_to_py_object_no_sync: Some(managed_to_python::<T>),
        dltensor_from_py_object_no_sync: T::HAS_DLTENSOR_VIEW.then_some(tensor_view::<T>),
        current_work_stream: Some(current_work_stream::<T>),
    }));
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            (api as *mut DLPackExchangeAPI).cast(),
            DLPACK_EXCHANGE_API.as_ptr(),
            None,
        )
    };
    let capsule = unsafe { Bound::from_owned_ptr_or_err(py, capsule)? };
    T::type_object(py).setattr("__dlpack_c_exchange_api__", capsule)
}

unsafe extern "C" fn managed_from_python<T>(
    object: *mut c_void,
    out: *mut *mut DLManagedTensorVersioned,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if object.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let object = unsafe { Bound::from_borrowed_ptr(py, object.cast()) };
        let producer: PyRef<'_, T> = object.extract()?;
        let tensor = producer.managed_tensor_no_sync(py)?;
        unsafe { out.write(tensor.into_raw()) };
        Ok(())
    })
}

unsafe extern "C" fn tensor_view<T>(object: *mut c_void, out: *mut DLTensor) -> c_int
where
    T: DlpackExchangeProducer,
{
    python_callback(|| {
        if object.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let object = unsafe { Bound::from_borrowed_ptr(py, object.cast()) };
        let producer: PyRef<'_, T> = object.extract()?;
        unsafe { out.write(producer.tensor_view_no_sync(py)?) };
        Ok(())
    })
}

unsafe extern "C" fn managed_to_python<T>(
    raw: *mut DLManagedTensorVersioned,
    out: *mut *mut c_void,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if raw.is_null() || out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API argument"));
        }
        let py = unsafe { Python::assume_attached() };
        let tensor = unsafe { Managed::from_raw(raw) }
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let object = T::from_managed_tensor_no_sync(py, tensor)?;
        unsafe { out.write(object.into_ptr().cast()) };
        Ok(())
    })
}

unsafe extern "C" fn current_work_stream<T>(
    device_type: DLDeviceType,
    device_id: i32,
    out: *mut *mut c_void,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    python_callback(|| {
        if out.is_null() {
            return Err(PyValueError::new_err("null C Exchange API output"));
        }
        let py = unsafe { Python::assume_attached() };
        let stream = T::current_work_stream(
            py,
            DLDevice {
                device_type,
                device_id,
            },
        )?;
        unsafe { out.write(stream) };
        Ok(())
    })
}

unsafe extern "C" fn allocate<T>(
    prototype: *mut DLTensor,
    out: *mut *mut DLManagedTensorVersioned,
    error_ctx: *mut c_void,
    set_error: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char)>,
) -> c_int
where
    T: DlpackExchangeProducer,
{
    if !out.is_null() {
        unsafe { out.write(ptr::null_mut()) };
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let prototype = unsafe { prototype.as_ref() }.ok_or("prototype is null")?;
        let out = unsafe { out.as_mut() }.ok_or("output is null")?;
        *out = T::allocate(prototype)?.into_raw();
        Ok::<(), String>(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => {
            set_exchange_error(error_ctx, set_error, &message);
            -1
        }
        Err(_) => {
            set_exchange_error(error_ctx, set_error, "Rust panic in DLPack allocator");
            -1
        }
    }
}

fn python_callback(callback: impl FnOnce() -> PyResult<()>) -> c_int {
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            error.restore(unsafe { Python::assume_attached() });
            -1
        }
        Err(_) => {
            PyRuntimeError::new_err("Rust panic in DLPack C Exchange API callback")
                .restore(unsafe { Python::assume_attached() });
            -1
        }
    }
}

fn set_exchange_error(
    context: *mut c_void,
    callback: Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char)>,
    message: &str,
) {
    let Some(callback) = callback else {
        return;
    };
    let message = CString::new(message.replace('\0', "\\0"))
        .expect("replacing NUL characters produces a valid CString");
    unsafe { callback(context, c"RuntimeError".as_ptr(), message.as_ptr()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DlpackFlags, allocation::fixed::make_test_tensor, ffi::DLDataType,
        python::exchange::DlpackExchangeApiRef,
    };
    use pyo3::prelude::*;
    use std::ffi::c_void;

    fn tensor() -> Managed<DLManagedTensorVersioned> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        make_test_tensor(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice::CPU,
            [3],
            [1],
            DlpackFlags::empty(),
        )
    }

    #[pyclass]
    struct TestProducer;

    #[pyclass]
    struct TestProducerWithoutView;

    unsafe impl DlpackExchangeProducer for TestProducer {
        const HAS_DLTENSOR_VIEW: bool = true;

        fn managed_tensor_no_sync(
            &self,
            _py: Python<'_>,
        ) -> PyResult<Managed<DLManagedTensorVersioned>> {
            Ok(tensor())
        }

        fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<DLTensor> {
            static DATA: [i32; 3] = [1, 2, 3];
            static SHAPE: [i64; 1] = [3];
            static STRIDES: [i64; 1] = [1];
            Ok(DLTensor {
                data: DATA.as_ptr() as *mut c_void,
                device: DLDevice::CPU,
                ndim: 1,
                dtype: DLDataType::of::<i32>(),
                shape: SHAPE.as_ptr().cast_mut(),
                strides: STRIDES.as_ptr().cast_mut(),
                byte_offset: 0,
            })
        }

        fn current_work_stream(_py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void> {
            assert_eq!(device, DLDevice::CPU);
            Ok(ptr::null_mut())
        }

        fn from_managed_tensor_no_sync(
            py: Python<'_>,
            _tensor: Managed<DLManagedTensorVersioned>,
        ) -> PyResult<Py<Self>> {
            Py::new(py, Self)
        }
    }

    unsafe impl DlpackExchangeProducer for TestProducerWithoutView {
        fn managed_tensor_no_sync(
            &self,
            _py: Python<'_>,
        ) -> PyResult<Managed<DLManagedTensorVersioned>> {
            Ok(tensor())
        }

        fn current_work_stream(_py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void> {
            assert_eq!(device, DLDevice::CPU);
            Ok(ptr::null_mut())
        }
    }

    #[test]
    fn installs_working_exchange_api_on_pyclass_type() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            install_exchange_api::<TestProducer>(py)?;
            let object = Py::new(py, TestProducer)?.into_bound(py);
            let api = DlpackExchangeApiRef::from_object(object.as_any().as_borrowed())?.unwrap();

            api.with_dltensor_view_no_sync(object.as_any().as_borrowed(), |view| {
                assert_eq!(view.device, DLDevice::CPU);
                assert_eq!(view.ndim, 1);
            })?;
            assert!(api.current_work_stream(DLDevice::CPU)?.is_null());
            let managed =
                api.managed_tensor_from_py_object_no_sync(object.as_any().as_borrowed())?;
            assert_eq!(managed.validate().unwrap().shape(), &[3]);

            let converted = api.managed_tensor_to_py_object_no_sync(tensor(), py)?;
            assert!(converted.is_instance_of::<TestProducer>());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn installed_exchange_api_omits_unsupported_tensor_view() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            install_exchange_api::<TestProducerWithoutView>(py)?;
            let object = Py::new(py, TestProducerWithoutView)?.into_bound(py);
            let api = DlpackExchangeApiRef::from_object(object.as_any().as_borrowed())?.unwrap();

            assert!(!api.supports_dltensor_view());
            let managed =
                api.managed_tensor_from_py_object_no_sync(object.as_any().as_borrowed())?;
            assert_eq!(managed.validate().unwrap().shape(), &[3]);
            Ok(())
        })
        .unwrap();
    }
}
