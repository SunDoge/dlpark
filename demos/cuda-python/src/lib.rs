mod cuda;

use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{DLDeviceType, DLManagedTensor, DLManagedTensorVersioned},
    metadata::{Copied, Dynamic},
    python::{CudaStreamRequest, ExportRequest, ImportedDlpack, from_dlpack},
};
use cuda::CudaStream;
use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{ffi::c_void, sync::Arc};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A reusable CUDA tensor object implementing Python's DLPack protocol.
#[pyclass(unsendable)]
struct CudaTensor {
    dlpack: Arc<ImportedDlpack>,
    stream: CudaStream,
}

impl CudaTensor {
    fn export<M>(&self) -> PyResult<Managed<M>>
    where
        M: ManagedTensorBase,
    {
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let shape = descriptor.shape().to_vec();
        let strides = descriptor.strides().unwrap_or(&[]).to_vec();
        let prepared = Dynamic::new(Copied(shape), Copied(strides))
            .prepare::<M>()
            .map_err(runtime_error)?;
        let mut initialized = prepared.initialize(Arc::clone(&self.dlpack));
        initialized
            .set_data(descriptor.data_ptr().cast_mut())
            .set_device(descriptor.device())
            .set_dtype(descriptor.dtype())
            .set_byte_offset(descriptor.byte_offset());
        let flags = self.dlpack.flags().difference(DlpackFlags::IS_COPIED);
        initialized.set_flags(flags).map_err(runtime_error)?;
        // SAFETY: this fresh header copies the source descriptor and retains
        // `Arc<ImportedDlpack>` in its manager context.
        Ok(unsafe { initialized.finish() })
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

#[pymethods]
impl CudaTensor {
    #[new]
    fn new(tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = dlpark::python::dlpack_device(tensor.as_borrowed())?;
        if device.device_type != DLDeviceType::CUDA {
            return Err(PyValueError::new_err(format!(
                "expected a CUDA tensor, got {:?}",
                device.device_type
            )));
        }

        let stream = CudaStream::new(device.device_id).map_err(runtime_error)?;
        let managed = from_dlpack(tensor.as_borrowed(), Some(&stream), None)?;
        let descriptor = managed.validate().map_err(runtime_error)?;
        let abi = match &managed {
            ImportedDlpack::Legacy(_) => "legacy".to_owned(),
            ImportedDlpack::Versioned(tensor) => {
                let version = tensor.version();
                format!("{}.{}", version.major, version.minor)
            }
        };
        let source_flags = managed.flags();
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
        let data = if descriptor.data_ptr().is_null() {
            std::ptr::null_mut()
        } else {
            (descriptor.data_ptr() as usize)
                .checked_add(byte_offset)
                .ok_or_else(|| PyValueError::new_err("CUDA data pointer overflows usize"))?
                as *mut c_void
        };
        let device_pointer = data as usize as u64;
        let shape = descriptor.shape();
        let strides = descriptor.strides().unwrap_or(&[]);

        eprintln!("[dlpark/cuda] received DLPack tensor");
        eprintln!(
            "[dlpark/cuda] abi={abi} flags={source_flags:?} read_only={} is_copied={}",
            source_flags.contains(DlpackFlags::READ_ONLY),
            source_flags.contains(DlpackFlags::IS_COPIED)
        );
        eprintln!(
            "[dlpark/cuda] device=CUDA({}):{} dtype=float32(code={}, bits={}, lanes={})",
            device.device_type.0,
            device.device_id,
            descriptor.dtype().code.0,
            descriptor.dtype().bits,
            descriptor.dtype().lanes
        );
        eprintln!("[dlpark/cuda] shape={shape:?} strides={strides:?} compact={compact}");
        eprintln!(
            "[dlpark/cuda] elements={length} bytes={} byte_offset={} base_pointer={:p} data_pointer=0x{device_pointer:x}",
            descriptor.num_bytes(),
            descriptor.byte_offset(),
            descriptor.data_ptr()
        );
        eprintln!("[dlpark/cuda] source_wait_stream={:p}", stream.as_raw());

        Ok(Self {
            dlpack: Arc::new(managed),
            stream,
        })
    }

    fn __dlpack_device__(&self) -> PyResult<(u32, i32)> {
        let device = self.dlpack.validate().map_err(runtime_error)?.device();
        Ok((device.device_type.0, device.device_id))
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
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let device = descriptor.device();
        let flags = self.dlpack.flags();
        self.synchronize(request.cuda_stream()?)?;
        request.export_zero_copy(
            py,
            device,
            flags,
            || self.export::<DLManagedTensor>(),
            || self.export::<DLManagedTensorVersioned>(),
        )
    }

    #[getter]
    fn device_id(&self) -> PyResult<usize> {
        let device_id = self
            .dlpack
            .validate()
            .map_err(runtime_error)?
            .device()
            .device_id;
        usize::try_from(device_id)
            .map_err(|_| PyValueError::new_err("CUDA device ID must be non-negative"))
    }

    #[getter]
    fn device_pointer(&self) -> PyResult<u64> {
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let byte_offset = usize::try_from(descriptor.byte_offset())
            .map_err(|_| PyValueError::new_err("byte offset does not fit usize"))?;
        if descriptor.data_ptr().is_null() {
            Ok(0)
        } else {
            (descriptor.data_ptr() as usize)
                .checked_add(byte_offset)
                .map(|pointer| pointer as u64)
                .ok_or_else(|| PyValueError::new_err("CUDA data pointer overflows usize"))
        }
    }

    #[getter]
    fn length(&self) -> PyResult<usize> {
        Ok(self
            .dlpack
            .validate()
            .map_err(runtime_error)?
            .num_elements())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CudaTensor>()?;
    Ok(())
}
