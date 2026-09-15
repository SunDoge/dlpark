//! Reusable Python DLPack producer protocol implementation.

use crate::{
    ffi::{DLDevice, DLPACK_MAJOR_VERSION},
    versioned,
};
use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::{PyBufferError, PyValueError},
    prelude::*,
};

type PrepareExport = dyn for<'py> FnMut(Option<&Bound<'py, PyAny>>) -> PyResult<()> + 'static;

/// A reusable, single-consumption Python producer for versioned DLPack tensors.
///
/// Use this as the base of a PyO3 class with
/// `#[pyclass(extends = DlpackProducer)]`. The subclass owns any diagnostic or
/// application-specific state, while this base class implements
/// `__dlpack_device__` and `__dlpack__`, validates the standard negotiation
/// arguments, invokes the backend synchronization callback, and transfers the
/// managed tensor exactly once.
///
/// Consumers must request DLPack 1.x through `max_version`; calls without
/// version negotiation receive `BufferError` rather than an incorrectly named
/// versioned capsule.
///
/// The callback receives Python's optional `stream` argument. It must make the
/// exported tensor safe for work submitted to that stream before returning.
#[pyclass(subclass, unsendable)]
pub struct DlpackProducer {
    tensor: Option<versioned::Dlpack>,
    device: DLDevice,
    prepare_export: Box<PrepareExport>,
}

impl DlpackProducer {
    /// Creates a producer with backend-specific stream synchronization.
    ///
    /// # Safety
    ///
    /// `prepare_export` must make the tensor safe for the consumer represented
    /// by its optional Python stream argument. It must not return success until
    /// the producer's outstanding work is visible to that consumer.
    pub unsafe fn new<F>(
        tensor: versioned::Dlpack,
        prepare_export: F,
    ) -> Result<Self, crate::tensor::Error>
    where
        F: for<'py> FnMut(Option<&Bound<'py, PyAny>>) -> PyResult<()> + 'static,
    {
        let device = tensor.validate()?.device();
        Ok(Self {
            tensor: Some(tensor),
            device,
            prepare_export: Box::new(prepare_export),
        })
    }

    /// Creates a producer for a backend which does not accept stream arguments.
    ///
    /// # Safety
    ///
    /// The tensor must already be ready for consumption without stream
    /// synchronization and remain so until ownership is transferred.
    pub unsafe fn without_stream(tensor: versioned::Dlpack) -> Result<Self, crate::tensor::Error> {
        unsafe {
            Self::new(tensor, |stream| {
                if stream.is_some() {
                    Err(PyValueError::new_err(
                        "this DLPack producer does not accept a stream argument",
                    ))
                } else {
                    Ok(())
                }
            })
        }
    }

    /// Returns whether ownership has already been transferred to a consumer.
    pub fn is_consumed(&self) -> bool {
        self.tensor.is_none()
    }

    /// Returns the device reported through Python's `__dlpack_device__`.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    fn ensure_available(&self) -> PyResult<()> {
        if self.is_consumed() {
            Err(PyBufferError::new_err(
                "this DLPack producer was already consumed",
            ))
        } else {
            Ok(())
        }
    }
}

#[pymethods]
impl DlpackProducer {
    fn __dlpack_device__(&self) -> (u32, i32) {
        (self.device.device_type.0, self.device.device_id)
    }

    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &mut self,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<versioned::Dlpack> {
        self.ensure_available()?;

        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "this DLPack producer only supports zero-copy export",
            ));
        }
        if !max_version.is_some_and(|(major, _)| major >= DLPACK_MAJOR_VERSION) {
            return Err(PyBufferError::new_err(
                "this DLPack producer requires max_version >= (1, 0)",
            ));
        }
        if let Some((device_type, device_id)) = dl_device
            && (device_type != self.device.device_type.0 || device_id != self.device.device_id)
        {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }

        (self.prepare_export)(stream)?;
        Ok(self
            .tensor
            .take()
            .expect("availability was checked before export"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DlpackFlags,
        allocation::fixed::make_test_tensor,
        ffi::{DLDataType, DLDevice, DLDeviceType, DLManagedTensorVersioned},
    };
    use pyo3::{PyClassInitializer, exceptions::PyBufferError, types::PyAnyMethods};
    use std::{cell::Cell, ffi::c_void, rc::Rc};

    #[pyclass(extends = DlpackProducer, unsendable)]
    struct TestProducer;

    fn tensor() -> versioned::Dlpack {
        let data = Box::new(vec![1_i32, 2, 3]);
        let data_ptr = data.as_ptr().cast_mut().cast::<c_void>();
        make_test_tensor::<_, DLManagedTensorVersioned, 1>(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: 2,
            },
            [3],
            [1],
            DlpackFlags::empty(),
        )
    }

    #[test]
    fn inherited_protocol_reports_device_and_consumes_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> PyResult<()> {
            let calls = Rc::new(Cell::new(0));
            let calls_in_callback = Rc::clone(&calls);
            let producer = unsafe {
                DlpackProducer::new(tensor(), move |_| {
                    calls_in_callback.set(calls_in_callback.get() + 1);
                    Ok(())
                })
            }
            .unwrap();
            let producer = Py::new(
                py,
                PyClassInitializer::from(producer).add_subclass(TestProducer),
            )?;

            assert_eq!(
                producer
                    .bind(py)
                    .call_method0("__dlpack_device__")?
                    .extract::<(u32, i32)>()?,
                (DLDeviceType::CUDA.0, 2)
            );
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("max_version", (1, 0))?;
            let capsule = producer
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))?;
            let _managed = versioned::Dlpack::extract(capsule.as_borrowed())?;
            assert_eq!(calls.get(), 1);

            let error = producer
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))
                .unwrap_err();
            assert!(error.is_instance_of::<PyBufferError>(py));
            assert_eq!(calls.get(), 1);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn rejected_request_does_not_consume_tensor() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> PyResult<()> {
            let producer = Py::new(
                py,
                unsafe { DlpackProducer::without_stream(tensor()) }.unwrap(),
            )?;

            let error = producer.bind(py).call_method0("__dlpack__").unwrap_err();
            assert!(error.is_instance_of::<PyBufferError>(py));

            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("max_version", (1, 0))?;
            kwargs.set_item("copy", true)?;
            let error = producer
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))
                .unwrap_err();
            assert!(error.is_instance_of::<PyBufferError>(py));

            kwargs.set_item("copy", false)?;
            let capsule = producer
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))?;
            let _managed = versioned::Dlpack::extract(capsule.as_borrowed())?;
            Ok(())
        })
        .unwrap();
    }
}
