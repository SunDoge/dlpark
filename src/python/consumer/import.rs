//! Ordered import of Python DLPack producers.

use crate::{
    DlpackFlags, Managed,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    python::{
        DlpackStream,
        capsule::{
            call_dlpack, consume_legacy_capsule, consume_versioned_capsule, is_legacy_capsule,
            is_versioned_capsule, validate_copy_result,
        },
        dlpack_device,
        exchange::DlpackExchangeApiRef,
    },
    tensor::{self, TensorRef},
};
use pyo3::{
    Borrowed, PyAny,
    exceptions::{PyAttributeError, PyTypeError, PyValueError},
    types::PyAnyMethods,
};

/// An owning tensor imported through either DLPack managed-tensor ABI.
pub enum ImportedDlpack {
    /// A legacy `DLManagedTensor`.
    Legacy(Managed<DLManagedTensor>),
    /// A versioned `DLManagedTensorVersioned`.
    Versioned(Managed<DLManagedTensorVersioned>),
}

impl ImportedDlpack {
    /// Validates and borrows the imported tensor descriptor.
    pub fn validate(&self) -> Result<TensorRef<'_>, tensor::Error> {
        match self {
            Self::Legacy(tensor) => tensor.validate(),
            Self::Versioned(tensor) => tensor.validate(),
        }
    }

    /// Returns the versioned flags, or an empty set for a legacy tensor.
    pub fn flags(&self) -> DlpackFlags {
        match self {
            Self::Legacy(tensor) => tensor.flags(),
            Self::Versioned(tensor) => tensor.flags(),
        }
    }
}

/// Imports a Python object through the fastest DLPack protocol it supports.
///
/// Existing capsules are consumed directly. Other objects are tried in this
/// order: DLPack 1.3 C Exchange API, the standard Python Array API protocol
/// (`__dlpack_device__` plus `__dlpack__`), then legacy `__dlpack__` without
/// negotiation arguments.
pub fn from_dlpack(
    object: Borrowed<'_, '_, PyAny>,
    stream: Option<&dyn DlpackStream>,
    copy: Option<bool>,
) -> pyo3::PyResult<ImportedDlpack> {
    if is_versioned_capsule(object) || is_legacy_capsule(object) {
        if stream.is_some() || copy.is_some() {
            return Err(PyValueError::new_err(
                "an existing DLPack capsule cannot negotiate stream or copy options",
            ));
        }
        return consume_capsule(object);
    }

    if copy != Some(true)
        && let Some(api) = DlpackExchangeApiRef::from_object(object)?
        && exchange_stream_is_ready(&api, object, stream)?
    {
        let tensor = api.managed_tensor_from_py_object_no_sync(object)?;
        return Ok(ImportedDlpack::Versioned(validate_copy_result(
            tensor, copy,
        )?));
    }

    if has_attribute(object, "__dlpack_device__")? {
        let device = dlpack_device(object)?;
        let stream_arg = match stream {
            Some(stream) => stream
                .as_python_arg(object.py(), device)?
                .into_python(object.py())?,
            None => None,
        };
        let capsule = call_dlpack(
            object,
            Some((
                crate::ffi::DLPACK_MAJOR_VERSION,
                crate::ffi::DLPACK_MINOR_VERSION,
            )),
            stream_arg.as_ref(),
            copy,
        )?;
        let imported = consume_capsule(capsule.as_borrowed())?;
        return validate_copy(imported, copy);
    }

    if !has_attribute(object, "__dlpack__")? {
        return Err(PyTypeError::new_err(
            "object does not support the DLPack protocol",
        ));
    }
    if stream.is_some() || copy.is_some() {
        return Err(PyValueError::new_err(
            "a legacy __dlpack__ producer cannot negotiate stream or copy options",
        ));
    }
    let capsule = call_dlpack(object, None, None, None)?;
    consume_capsule(capsule.as_borrowed())
}

fn exchange_stream_is_ready(
    api: &DlpackExchangeApiRef,
    object: Borrowed<'_, '_, PyAny>,
    stream: Option<&dyn DlpackStream>,
) -> pyo3::PyResult<bool> {
    if !api.supports_dltensor_view() {
        return Ok(false);
    }
    let device = api.with_dltensor_view_no_sync(object, |tensor| tensor.device)?;
    if device.device_id < 0 {
        return Err(PyValueError::new_err(format!(
            "DLPack device ID must be non-negative, got {}",
            device.device_id
        )));
    }
    let producer = api.current_work_stream(device)?;
    match stream {
        Some(stream) => unsafe { stream.wait_for_producer(device, producer) },
        None => Ok(producer.is_null()),
    }
}

fn consume_capsule(object: Borrowed<'_, '_, PyAny>) -> pyo3::PyResult<ImportedDlpack> {
    if is_versioned_capsule(object) {
        return Ok(ImportedDlpack::Versioned(consume_versioned_capsule(
            object,
        )?));
    }
    if is_legacy_capsule(object) {
        return Ok(ImportedDlpack::Legacy(consume_legacy_capsule(object)?));
    }
    Err(PyValueError::new_err(
        "__dlpack__ returned a capsule with neither dltensor nor dltensor_versioned ABI",
    ))
}

fn validate_copy(
    imported: ImportedDlpack,
    requested: Option<bool>,
) -> pyo3::PyResult<ImportedDlpack> {
    match imported {
        ImportedDlpack::Versioned(tensor) => Ok(ImportedDlpack::Versioned(validate_copy_result(
            tensor, requested,
        )?)),
        ImportedDlpack::Legacy(_) if requested.is_some() => Err(PyValueError::new_err(
            "a legacy DLPack tensor cannot report whether a requested copy was made",
        )),
        legacy => Ok(legacy),
    }
}

fn has_attribute(object: Borrowed<'_, '_, PyAny>, name: &str) -> pyo3::PyResult<bool> {
    match object.getattr(name) {
        Ok(_) => Ok(true),
        Err(error) if error.is_instance_of::<PyAttributeError>(object.py()) => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DlpackFlags,
        allocation::fixed::make_test_tensor,
        ffi::{DLDataType, DLDevice},
    };
    use pyo3::{IntoPyObject, Python, types::PyModule};
    use std::ffi::c_void;

    fn versioned_tensor() -> Managed<DLManagedTensorVersioned> {
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

    fn legacy_tensor() -> Managed<DLManagedTensor> {
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

    #[test]
    fn standard_protocol_queries_device_then_negotiates_version() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.calls = []

    def __dlpack_device__(self):
        self.calls.append("device")
        return (1, 0)

    def __dlpack__(self, **kwargs):
        self.calls.append(("dlpack", kwargs))
        self.max_version = kwargs["max_version"]
        capsule, self.capsule = self.capsule, None
        return capsule
"#,
                c"standard_import.py",
                c"standard_import",
            )?;
            let capsule = versioned_tensor().into_pyobject(py)?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let imported = from_dlpack(producer.as_borrowed(), None, None)?;

            assert!(matches!(imported, ImportedDlpack::Versioned(_)));
            let calls = producer.getattr("calls")?;
            assert_eq!(calls.get_item(0)?.extract::<String>()?, "device");
            assert_eq!(
                producer.getattr("max_version")?.extract::<(u32, u32)>()?,
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
    fn falls_back_to_legacy_no_argument_protocol() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.calls = 0

    def __dlpack__(self):
        self.calls += 1
        capsule, self.capsule = self.capsule, None
        return capsule
"#,
                c"legacy_import.py",
                c"legacy_import",
            )?;
            let capsule = legacy_tensor().into_pyobject(py)?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let imported = from_dlpack(producer.as_borrowed(), None, None)?;

            assert!(matches!(imported, ImportedDlpack::Legacy(_)));
            assert_eq!(producer.getattr("calls")?.extract::<usize>()?, 1);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn rejects_objects_without_any_dlpack_protocol() {
        Python::initialize();
        Python::attach(|py| {
            let object = pyo3::types::PyDict::new(py);
            let error = match from_dlpack(object.as_any().as_borrowed(), None, None) {
                Ok(_) => panic!("object without a DLPack protocol was accepted"),
                Err(error) => error,
            };
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }
}
