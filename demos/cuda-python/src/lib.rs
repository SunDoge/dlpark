mod cuda;

use cuda::{CudaBuffer, CudaStream};
use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
    },
    metadata::{Copied, Dynamic},
    python::{
        CudaStreamRequest, DlpackExchangeProducer, ExportRequest, ImportedDlpack, from_dlpack,
        install_exchange_api,
    },
};
use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};
use std::{ffi::c_void, sync::Arc};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A reusable CUDA tensor implementing Python's DLPack protocols.
///
/// Tensor metadata is owned directly by this object. `buffer` is a zero-copy
/// CUDA allocation whose custom deleter releases the imported allocation.
#[pyclass(unsendable)]
struct CudaTensor {
    buffer: Arc<CudaBuffer>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: DLDataType,
    flags: DlpackFlags,
    length: usize,
    stream: Arc<CudaStream>,
}

impl CudaTensor {
    fn device(&self) -> DLDevice {
        DLDevice::cuda(self.buffer.device())
    }

    fn export<M>(&self) -> PyResult<Managed<M>>
    where
        M: ManagedTensorBase,
    {
        let prepared = Dynamic::new(Copied(self.shape.clone()), Copied(self.strides.clone()))
            .prepare::<M>()
            .map_err(runtime_error)?;
        let mut initialized = prepared.initialize(Arc::clone(&self.buffer));
        initialized
            .set_data(self.buffer.as_raw())
            .set_device(self.device())
            .set_dtype(self.dtype)
            .set_byte_offset(0);
        initialized.set_flags(self.flags).map_err(runtime_error)?;
        // SAFETY: the fresh header owns copied metadata and an Arc<CudaBuffer>.
        // CudaBuffer retains the original custom deleter until the last
        // exported managed tensor is released.
        Ok(unsafe { initialized.finish() })
    }

    fn tensor_view(&self) -> DLTensor {
        DLTensor {
            data: self.buffer.as_raw(),
            device: self.device(),
            ndim: i32::try_from(self.shape.len())
                .expect("validated DLPack rank continues to fit i32"),
            dtype: self.dtype,
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
            byte_offset: 0,
        }
    }

    fn synchronize(&self, stream: CudaStreamRequest) -> PyResult<()> {
        match stream {
            CudaStreamRequest::Unspecified => {
                eprintln!(
                    "[dlpark/cuda] destination supplied no stream; synchronizing relay stream on the host"
                );
                return self.stream.synchronize().map_err(runtime_error);
            }
            CudaStreamRequest::NoSync => eprintln!(
                "[dlpark/cuda] destination requested stream=-1; no synchronization inserted"
            ),
            stream => {
                let raw = stream
                    .python_value()
                    .expect("a concrete CUDA stream has a Python value");
                let consumer = stream
                    .as_raw()
                    .expect("a concrete CUDA stream has a native representation");
                unsafe { self.stream.hand_off_to_raw(consumer) }.map_err(runtime_error)?;
                eprintln!(
                    "[dlpark/cuda] event handoff relay_stream={:p} destination_stream_arg={raw:#x} cuda_stream={consumer:p}",
                    self.stream.as_raw(),
                );
            }
        }
        Ok(())
    }
}

fn imported_deleter(tensor: ImportedDlpack) -> Box<dyn FnOnce() + Send> {
    match tensor {
        ImportedDlpack::Legacy(tensor) => {
            let raw = tensor.into_raw() as usize;
            Box::new(move || unsafe {
                <DLManagedTensor as ManagedTensorBase>::drop_raw(raw as *mut DLManagedTensor);
            })
        }
        ImportedDlpack::Versioned(tensor) => {
            let raw = tensor.into_raw() as usize;
            Box::new(move || unsafe {
                <DLManagedTensorVersioned as ManagedTensorBase>::drop_raw(
                    raw as *mut DLManagedTensorVersioned,
                );
            })
        }
    }
}

#[pymethods]
impl CudaTensor {
    #[classmethod]
    fn from_dlpack(_class: &Bound<'_, PyType>, tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = dlpark::python::dlpack_device(tensor.as_borrowed())?;
        if device.device_type != DLDeviceType::CUDA {
            return Err(PyValueError::new_err(format!(
                "expected a CUDA tensor, got {:?}",
                device.device_type
            )));
        }

        let stream = CudaStream::for_device(device.device_id).map_err(runtime_error)?;
        let managed = from_dlpack(tensor.as_borrowed(), Some(stream.as_ref()), None)?;
        let abi = match &managed {
            ImportedDlpack::Legacy(_) => "legacy".to_owned(),
            ImportedDlpack::Versioned(tensor) => {
                let version = tensor.version();
                format!("{}.{}", version.major, version.minor)
            }
        };
        let source_flags = managed.flags();

        let (
            shape,
            strides,
            dtype,
            length,
            byte_len,
            byte_offset,
            base_address,
            data_address,
            compact,
        ) = {
            let descriptor = managed.validate().map_err(runtime_error)?;
            let compact = descriptor.is_compact().map_err(runtime_error)?;
            if !descriptor.dtype().is::<f32>() {
                return Err(PyValueError::new_err(format!(
                    "expected float32, got {:?}",
                    descriptor.dtype()
                )));
            }
            if !compact {
                return Err(PyValueError::new_err("expected a compact CUDA tensor"));
            }

            let length = descriptor.num_elements();
            if length != 0 && descriptor.data_ptr().is_null() {
                return Err(PyValueError::new_err(
                    "non-empty CUDA tensor has a null data pointer",
                ));
            }
            let byte_offset = usize::try_from(descriptor.byte_offset())
                .map_err(|_| PyValueError::new_err("byte offset does not fit usize"))?;
            let base_address = descriptor.data_ptr() as usize;
            let data_address = if descriptor.data_ptr().is_null() {
                0
            } else {
                base_address
                    .checked_add(byte_offset)
                    .ok_or_else(|| PyValueError::new_err("CUDA data pointer overflows usize"))?
            };
            (
                descriptor.shape().to_vec(),
                descriptor
                    .strides_or_compact()
                    .map_err(runtime_error)?
                    .into_owned(),
                descriptor.dtype(),
                length,
                descriptor.num_bytes(),
                byte_offset,
                base_address,
                data_address,
                compact,
            )
        };

        let flags = source_flags.difference(DlpackFlags::IS_COPIED);
        let deleter = imported_deleter(managed);
        // SAFETY: the custom deleter now owns the imported managed tensor's raw
        // pointer. The address and byte length came from that tensor's validated
        // compact descriptor after applying byte_offset.
        let buffer = Arc::new(unsafe {
            CudaBuffer::from_external(data_address, byte_len, device.device_id, deleter)
        });

        eprintln!("[dlpark/cuda] imported zero-copy CUDA buffer");
        eprintln!(
            "[dlpark/cuda] abi={abi} flags={source_flags:?} read_only={} is_copied={}",
            source_flags.contains(DlpackFlags::READ_ONLY),
            source_flags.contains(DlpackFlags::IS_COPIED)
        );
        eprintln!(
            "[dlpark/cuda] device=CUDA({}):{} dtype=float32(code={}, bits={}, lanes={})",
            device.device_type.0, device.device_id, dtype.code.0, dtype.bits, dtype.lanes
        );
        eprintln!("[dlpark/cuda] shape={shape:?} strides={strides:?} compact={compact}");
        eprintln!(
            "[dlpark/cuda] elements={length} bytes={} byte_offset={byte_offset} base_pointer=0x{base_address:x} buffer_pointer=0x{data_address:x}",
            buffer.byte_len(),
        );
        eprintln!("[dlpark/cuda] source_wait_stream={:p}", stream.as_raw());

        Ok(Self {
            buffer,
            shape,
            strides,
            dtype,
            flags,
            length,
            stream,
        })
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        let device = self.device();
        (device.device_type.0, device.device_id)
    }

    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let request = ExportRequest::parse(stream, max_version, dl_device, copy)?;
        self.synchronize(request.cuda_stream()?)?;
        request.export_zero_copy(
            py,
            self.device(),
            self.flags,
            || self.export::<DLManagedTensor>(),
            || self.export::<DLManagedTensorVersioned>(),
        )
    }

    #[getter]
    fn device_id(&self) -> usize {
        usize::try_from(self.buffer.device()).expect("validated CUDA device ID is non-negative")
    }

    #[getter]
    fn device_pointer(&self) -> u64 {
        self.buffer.as_raw() as usize as u64
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }
}

// SAFETY: tensor_view returns pointers into immutable Vec metadata owned by the
// PyClass and a CudaBuffer retained by self. Managed exports retain their own
// Arc<CudaBuffer>. The process-wide per-device stream stays live in cuda.rs.
unsafe impl DlpackExchangeProducer for CudaTensor {
    fn managed_tensor_no_sync(
        &self,
        _py: Python<'_>,
    ) -> PyResult<Managed<DLManagedTensorVersioned>> {
        self.export()
    }

    fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<DLTensor> {
        Ok(self.tensor_view())
    }

    fn current_work_stream(
        _py: Python<'_>,
        device: DLDevice,
    ) -> PyResult<*mut c_void> {
        if device.device_type != DLDeviceType::CUDA {
            return Err(PyValueError::new_err(format!(
                "expected a CUDA device, got {:?}",
                device.device_type
            )));
        }
        if device.device_id < 0 {
            return Err(PyValueError::new_err(format!(
                "CUDA device ID must be non-negative, got {}",
                device.device_id
            )));
        }
        Ok(CudaStream::for_device(device.device_id)
            .map_err(runtime_error)?
            .as_raw())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CudaTensor>()?;
    install_exchange_api::<CudaTensor>(module.py())?;
    Ok(())
}
