use dlpark::{DlpackFlags, ffi::DLDeviceType, runtime::cuda::CudaStream, versioned};
use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A zero-copy relay over a Python-produced contiguous CUDA `float32` tensor.
#[pyclass(unsendable)]
struct CudaTensorF32 {
    inner: Option<versioned::Dlpack>,
    stream: CudaStream,
    device_id: usize,
    device_pointer: u64,
    length: usize,
}

#[pymethods]
impl CudaTensorF32 {
    #[new]
    fn new(tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let device = dlpark::python::dlpack_device(tensor.as_borrowed())?;
        if device.device_type != DLDeviceType::CUDA {
            return Err(PyValueError::new_err(format!(
                "expected a CUDA tensor, got {:?}",
                device.device_type
            )));
        }
        let device_id = usize::try_from(device.device_id)
            .map_err(|_| PyValueError::new_err("CUDA device ID must be non-negative"))?;

        let negotiation_stream = CudaStream::new(device.device_id).map_err(runtime_error)?;
        let managed = versioned::Dlpack::extract_with_stream(
            tensor.as_borrowed(),
            &negotiation_stream,
            None,
        )?;

        let descriptor = managed.validate().map_err(runtime_error)?;
        let version = managed.version();
        let flags = managed.flags();
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
        let device_pointer = if descriptor.data_ptr().is_null() {
            0
        } else {
            (descriptor.data_ptr() as usize)
                .checked_add(byte_offset)
                .ok_or_else(|| PyValueError::new_err("CUDA data pointer overflows usize"))?
                as u64
        };
        eprintln!("[dlpark/cuda] received DLPack tensor");
        eprintln!(
            "[dlpark/cuda] version={}.{} flags={flags:?} read_only={} is_copied={}",
            version.major,
            version.minor,
            flags.contains(DlpackFlags::READ_ONLY),
            flags.contains(DlpackFlags::IS_COPIED)
        );
        eprintln!(
            "[dlpark/cuda] device=CUDA({}):{} dtype=float32(code={}, bits={}, lanes={})",
            descriptor.device().device_type.0,
            descriptor.device().device_id,
            descriptor.dtype().code.0,
            descriptor.dtype().bits,
            descriptor.dtype().lanes
        );
        eprintln!(
            "[dlpark/cuda] shape={:?} strides={:?} compact={compact}",
            descriptor.shape(),
            descriptor.strides()
        );
        eprintln!(
            "[dlpark/cuda] elements={} bytes={} byte_offset={} base_pointer={:p} data_pointer=0x{device_pointer:x}",
            descriptor.num_elements(),
            descriptor.num_bytes(),
            descriptor.byte_offset(),
            descriptor.data_ptr()
        );
        eprintln!(
            "[dlpark/cuda] source_wait_stream={:p}",
            negotiation_stream.as_raw()
        );
        eprintln!(
            "[dlpark/cuda] ownership=managed DLPack capsule; no CUDA allocation wrapper created"
        );

        Ok(Self {
            inner: Some(managed),
            stream: negotiation_stream,
            device_id,
            device_pointer,
            length,
        })
    }

    #[getter]
    fn device_id(&self) -> usize {
        self.device_id
    }

    #[getter]
    fn device_pointer(&self) -> u64 {
        self.device_pointer
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    fn __dlpack_device__(&self) -> (u32, usize) {
        (DLDeviceType::CUDA.0, self.device_id)
    }

    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &mut self,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, usize)>,
        copy: Option<bool>,
    ) -> PyResult<versioned::Dlpack> {
        if self.inner.is_none() {
            return Err(PyBufferError::new_err("CudaTensorF32 was already consumed"));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "CudaTensorF32 only supports zero-copy export",
            ));
        }
        if matches!(max_version, Some((0, _))) {
            return Err(PyBufferError::new_err(
                "CudaTensorF32 exports the versioned DLPack ABI",
            ));
        }
        if let Some((device_type, device_id)) = dl_device
            && (device_type != DLDeviceType::CUDA.0 || device_id != self.device_id)
        {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }

        match stream {
            None => {
                eprintln!(
                    "[dlpark/cuda] destination supplied no stream; synchronizing relay stream on the host"
                );
                self.stream.synchronize().map_err(runtime_error)?;
            }
            Some(value) => {
                let raw = value
                    .extract::<isize>()
                    .map_err(|_| PyValueError::new_err("CUDA DLPack stream must be an integer"))?;
                match raw {
                    -1 => eprintln!(
                        "[dlpark/cuda] destination requested stream=-1; no synchronization inserted"
                    ),
                    0 => {
                        return Err(PyValueError::new_err(
                            "CUDA DLPack stream 0 is ambiguous; use sentinel 1 for the legacy default stream",
                        ));
                    }
                    raw if raw > 0 => {
                        let consumer = match raw {
                            1 => std::ptr::null_mut(),
                            _ => raw as usize as *mut std::ffi::c_void,
                        };
                        unsafe { self.stream.hand_off_to_raw(consumer) }.map_err(runtime_error)?;
                        eprintln!(
                            "[dlpark/cuda] event handoff relay_stream={:p} destination_stream_arg={raw:#x} cuda_stream={consumer:p}",
                            self.stream.as_raw(),
                        );
                    }
                    _ => {
                        return Err(PyValueError::new_err(
                            "CUDA DLPack stream must be -1, 1, 2, or a positive stream pointer",
                        ));
                    }
                }
            }
        }

        let managed = self
            .inner
            .take()
            .ok_or_else(|| PyBufferError::new_err("CudaTensorF32 was already consumed"))?;
        eprintln!("[dlpark/cuda] exporting the original allocation to the next DLPack consumer");
        Ok(managed)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CudaTensorF32>()?;
    Ok(())
}
