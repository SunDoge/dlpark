//! Ordered import of Python DLPack producers.

use super::{
    device::{standard_dlpack_device, validate_device},
    exchange::DlpackExchangeApiRef,
    stream::DlpackStream,
};
use crate::{
    AllocationDeleter, DlpackFlags, Managed,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    python::capsule::{
        call_dlpack, consume_legacy_capsule, consume_versioned_capsule, is_legacy_capsule,
        is_versioned_capsule, validate_copy_result,
    },
    tensor::{self, TensorRef},
};
use pyo3::{
    Borrowed, Bound, PyAny, PyErr,
    exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError},
    types::PyAnyMethods,
};

/// A discovered Python DLPack import that can be completed after selecting a
/// device-specific consumer stream.
///
/// Constructing a request queries the producer device once and caches its C
/// Exchange API table, when present. This is useful for CUDA and ROCm callers
/// that must know the device ordinal before they can create the stream passed
/// to [`Self::import`].
///
/// The request owns a Python reference to the input for `'py` and is consumed
/// by [`Self::import`]. It does not consume a standard Python DLPack producer
/// during discovery. If a C Exchange producer provides neither the optional
/// borrowed view nor `__dlpack_device__`, discovery obtains one owning managed
/// tensor and keeps it inside the request for the final import.
///
/// Use [`from_dlpack`] for the common one-step case. It is equivalent to
/// constructing a request and immediately importing it.
pub struct ImportRequest<'py> {
    object: Bound<'py, PyAny>,
    exchange: Option<DlpackExchangeApiRef>,
    exchange_tensor: Option<Managed<DLManagedTensorVersioned>>,
    device: Option<crate::ffi::DLDevice>,
}

impl<'py> ImportRequest<'py> {
    /// Discovers the object's preferred DLPack protocol and device.
    ///
    /// Discovery follows the same priority as [`from_dlpack`]: C Exchange API,
    /// standard `__dlpack_device__`/`__dlpack__`, then legacy no-argument
    /// `__dlpack__`. Existing capsules are recognized but remain unconsumed
    /// until [`Self::import`].
    ///
    /// The returned request keeps `object` alive even if the caller drops its
    /// original Python reference.
    pub fn new(object: Borrowed<'_, 'py, PyAny>) -> pyo3::PyResult<Self> {
        let object = object.to_owned();
        if is_versioned_capsule(object.as_borrowed()) || is_legacy_capsule(object.as_borrowed()) {
            return Ok(Self {
                object,
                exchange: None,
                exchange_tensor: None,
                device: None,
            });
        }

        let exchange = DlpackExchangeApiRef::from_object(object.as_borrowed())?;
        let mut exchange_tensor = None;
        let device = if let Some(api) = &exchange
            && api.supports_dltensor_view()
        {
            Some(validate_device(api.with_dltensor_view_no_sync(
                object.as_borrowed(),
                |tensor| tensor.device,
            )?)?)
        } else if has_attribute(object.as_borrowed(), "__dlpack_device__")? {
            Some(standard_dlpack_device(object.as_borrowed())?)
        } else if let Some(api) = &exchange {
            let tensor = api.managed_tensor_from_py_object_no_sync(object.as_borrowed())?;
            let device = validate_device(imported_device(&tensor)?)?;
            exchange_tensor = Some(tensor);
            Some(device)
        } else {
            None
        };

        Ok(Self {
            object,
            exchange,
            exchange_tensor,
            device,
        })
    }

    /// Returns the producer device discovered while creating this request.
    ///
    /// Existing capsules and legacy producers that only implement
    /// no-argument `__dlpack__` do not expose a device before import.
    /// Calling this method does not invoke the producer again.
    ///
    /// # Errors
    ///
    /// Returns `AttributeError` when the input has no pre-import device query,
    /// such as an existing capsule or a legacy-only producer.
    pub fn device(&self) -> pyo3::PyResult<crate::ffi::DLDevice> {
        self.device.ok_or_else(|| {
            PyAttributeError::new_err(
                "this DLPack input does not expose a device before it is imported",
            )
        })
    }

    /// Completes the import using the already discovered protocol and device.
    ///
    /// `stream` is the consumer stream on which the imported tensor must be
    /// ready. With C Exchange, dlpark asks [`DlpackStream::wait_for_producer`]
    /// to order it after the producer stream. If that implementation declines,
    /// import falls back to standard Python `__dlpack__(stream=...)`.
    ///
    /// `copy` has the Python Array API tri-state meaning:
    ///
    /// - `None` leaves the choice to the producer;
    /// - `Some(false)` requires a zero-copy result;
    /// - `Some(true)` requires a copy and therefore bypasses C Exchange.
    ///
    /// The imported descriptor's device is checked against the device cached
    /// during discovery. This method consumes the request because existing
    /// capsules are single-use and an import may transfer managed ownership.
    ///
    /// Existing capsules accept neither `stream` nor `copy`. Legacy-only
    /// producers also reject those arguments because they cannot negotiate
    /// them.
    pub fn import(
        self,
        stream: Option<&dyn DlpackStream>,
        copy: Option<bool>,
    ) -> pyo3::PyResult<ImportedDlpack> {
        let Self {
            object,
            exchange,
            exchange_tensor,
            device,
        } = self;
        let object = object.as_borrowed();
        if is_versioned_capsule(object) || is_legacy_capsule(object) {
            if stream.is_some() || copy.is_some() {
                return Err(PyValueError::new_err(
                    "an existing DLPack capsule cannot negotiate stream or copy options",
                ));
            }
            return consume_capsule(object);
        }

        if copy != Some(true)
            && let Some(api) = &exchange
            && let Some(tensor) =
                import_from_exchange_api(api, object, exchange_tensor, device, stream)?
        {
            return Ok(ImportedDlpack::Versioned(validate_copy_result(
                tensor, copy,
            )?));
        }

        if let Some(device) = device {
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
            let imported = validate_copy(consume_capsule(capsule.as_borrowed())?, copy)?;
            validate_imported_device(&imported, device)?;
            return Ok(imported);
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
}

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

    /// Erases the imported managed tensor into its exactly-once allocation deleter.
    ///
    /// Containers can store this alongside their own device pointer and
    /// metadata without depending on either managed-tensor ABI.
    pub fn into_deleter(self) -> AllocationDeleter {
        match self {
            Self::Legacy(tensor) => tensor.into_deleter(),
            Self::Versioned(tensor) => tensor.into_deleter(),
        }
    }
}

/// Imports a Python object through the fastest DLPack protocol it supports.
///
/// Existing capsules are consumed directly. Other objects are tried in this
/// order: DLPack 1.3 C Exchange API, the standard Python Array API protocol
/// (`__dlpack_device__` plus `__dlpack__`), then legacy `__dlpack__` without
/// negotiation arguments.
///
/// This is the one-step form of [`ImportRequest`]. Callers that need the
/// producer device before constructing `stream` should use `ImportRequest`
/// directly to avoid repeating protocol discovery.
pub fn from_dlpack(
    object: Borrowed<'_, '_, PyAny>,
    stream: Option<&dyn DlpackStream>,
    copy: Option<bool>,
) -> pyo3::PyResult<ImportedDlpack> {
    ImportRequest::new(object)?.import(stream, copy)
}

fn import_from_exchange_api(
    api: &DlpackExchangeApiRef,
    object: Borrowed<'_, '_, PyAny>,
    cached_tensor: Option<Managed<DLManagedTensorVersioned>>,
    expected_device: Option<crate::ffi::DLDevice>,
    stream: Option<&dyn DlpackStream>,
) -> pyo3::PyResult<Option<Managed<DLManagedTensorVersioned>>> {
    if api.supports_dltensor_view() {
        let device = expected_device.ok_or_else(|| {
            PyRuntimeError::new_err("C Exchange tensor view did not provide a device")
        })?;
        if !exchange_stream_is_ready(api, device, stream)? {
            return Ok(None);
        }
        let tensor = api.managed_tensor_from_py_object_no_sync(object)?;
        let managed_device = imported_device(&tensor)?;
        if managed_device != device {
            return Err(PyValueError::new_err(format!(
                "DLPack C Exchange API device changed between borrowed and managed exports: {:?}:{} became {:?}:{}",
                device.device_type,
                device.device_id,
                managed_device.device_type,
                managed_device.device_id,
            )));
        }
        return Ok(Some(tensor));
    }

    let tensor = match cached_tensor {
        Some(tensor) => tensor,
        None => api.managed_tensor_from_py_object_no_sync(object)?,
    };
    let device = imported_device(&tensor)?;
    if let Some(expected) = expected_device
        && device != expected
    {
        return Err(device_changed_error(expected, device));
    }
    if exchange_stream_is_ready(api, device, stream)? {
        Ok(Some(tensor))
    } else {
        Ok(None)
    }
}

fn validate_imported_device(
    tensor: &ImportedDlpack,
    expected: crate::ffi::DLDevice,
) -> pyo3::PyResult<()> {
    let actual = tensor
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .device();
    if actual != expected {
        return Err(device_changed_error(expected, actual));
    }
    Ok(())
}

fn device_changed_error(expected: crate::ffi::DLDevice, actual: crate::ffi::DLDevice) -> PyErr {
    PyValueError::new_err(format!(
        "DLPack device changed during import: {:?}:{} became {:?}:{}",
        expected.device_type, expected.device_id, actual.device_type, actual.device_id,
    ))
}

fn imported_device(
    tensor: &Managed<DLManagedTensorVersioned>,
) -> pyo3::PyResult<crate::ffi::DLDevice> {
    tensor
        .validate()
        .map(|tensor| tensor.device())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn exchange_stream_is_ready(
    api: &DlpackExchangeApiRef,
    device: crate::ffi::DLDevice,
    stream: Option<&dyn DlpackStream>,
) -> pyo3::PyResult<bool> {
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
        versioned_tensor_on(DLDevice::CPU)
    }

    fn versioned_tensor_on(device: DLDevice) -> Managed<DLManagedTensorVersioned> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        make_test_tensor(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            device,
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

            let request = ImportRequest::new(producer.as_borrowed())?;
            assert_eq!(request.device()?, DLDevice::CPU);
            let imported = request.import(None, None)?;

            assert!(matches!(imported, ImportedDlpack::Versioned(_)));
            let calls = producer.getattr("calls")?;
            assert_eq!(calls.len()?, 2);
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
    fn rejects_a_device_change_between_discovery_and_import() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, **kwargs):
        return self.capsule
"#,
                c"changed_device.py",
                c"changed_device",
            )?;
            let capsule = versioned_tensor_on(DLDevice::cuda(0)).into_pyobject(py)?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let request = ImportRequest::new(producer.as_borrowed())?;
            assert_eq!(request.device()?, DLDevice::CPU);
            let error = match request.import(None, None) {
                Ok(_) => panic!("a changed producer device must be rejected"),
                Err(error) => error,
            };
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("device changed during import"));
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
