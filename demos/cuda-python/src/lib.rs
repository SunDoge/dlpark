use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned,
        DLPACK_MAJOR_VERSION,
    },
    metadata::{Copied, Dynamic},
    runtime::cuda::CudaStream,
    versioned,
};
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{ffi::c_void, sync::Arc};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

struct SourceLease {
    _managed: versioned::Dlpack,
}

// DLPack requires a producer's managed-tensor deleter to be callable from an
// arbitrary thread. The imported managed-tensor header is kept immutable, and
// Rust never dereferences its CUDA device pointer.
unsafe impl Send for SourceLease {}
unsafe impl Sync for SourceLease {}

/// A reusable CUDA tensor object implementing Python's DLPack protocol.
#[pyclass(unsendable)]
struct CudaTensor {
    source: Arc<SourceLease>,
    data: *mut c_void,
    device: DLDevice,
    shape: Vec<i64>,
    strides: Vec<i64>,
    flags: DlpackFlags,
    stream: CudaStream,
    device_pointer: u64,
    length: usize,
}

impl CudaTensor {
    fn export<M>(&self) -> PyResult<Managed<M>>
    where
        M: ManagedTensorBase,
    {
        let prepared = Dynamic::new(
            Copied(self.shape.as_slice()),
            Copied(self.strides.as_slice()),
        )
        .prepare::<M>()
        .map_err(runtime_error)?;
        let mut initialized = prepared.initialize(Arc::clone(&self.source));
        initialized
            .set_data(self.data)
            .set_device(self.device)
            .set_dtype(DLDataType::F32);
        initialized.set_flags(self.flags).map_err(runtime_error)?;
        // SAFETY: the descriptor points into storage retained by the imported
        // managed tensor inside this export's `Arc<SourceLease>` context.
        Ok(unsafe { initialized.finish() })
    }

    fn synchronize(&self, stream: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let Some(value) = stream else {
            eprintln!(
                "[dlpark/cuda] destination supplied no stream; synchronizing relay stream on the host"
            );
            return self.stream.synchronize().map_err(runtime_error);
        };

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
                    _ => raw as usize as *mut c_void,
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
        let managed = versioned::Dlpack::extract_with_stream(tensor.as_borrowed(), &stream, None)?;
        let descriptor = managed.validate().map_err(runtime_error)?;
        let version = managed.version();
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
        let shape = descriptor.shape().to_vec();
        let strides = descriptor.strides().unwrap_or(&[]).to_vec();

        eprintln!("[dlpark/cuda] received DLPack tensor");
        eprintln!(
            "[dlpark/cuda] version={}.{} flags={source_flags:?} read_only={} is_copied={}",
            version.major,
            version.minor,
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

        // A relay aliases the source allocation and cannot repeat the source
        // producer's claim that this export is a consumer-owned copy.
        let flags = source_flags.difference(DlpackFlags::IS_COPIED);
        Ok(Self {
            source: Arc::new(SourceLease { _managed: managed }),
            data,
            device,
            shape,
            strides,
            flags,
            stream,
            device_pointer,
            length,
        })
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        let device = self.device;
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
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "CudaTensor only supports zero-copy export",
            ));
        }
        if let Some(requested) = dl_device
            && requested != (self.device.device_type.0, self.device.device_id)
        {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }

        let versioned = max_version.is_some_and(|(major, _)| major >= DLPACK_MAJOR_VERSION);
        if !versioned && self.flags.contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED) {
            return Err(PyBufferError::new_err(
                "the legacy DLPack ABI cannot describe padded sub-byte elements",
            ));
        }
        self.synchronize(stream)?;

        if versioned {
            Ok(self
                .export::<DLManagedTensorVersioned>()?
                .into_pyobject(py)?
                .unbind())
        } else {
            Ok(self
                .export::<DLManagedTensor>()?
                .into_pyobject(py)?
                .unbind())
        }
    }

    #[getter]
    fn device_id(&self) -> usize {
        self.device.device_id as usize
    }

    #[getter]
    fn device_pointer(&self) -> u64 {
        self.device_pointer
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CudaTensor>()?;
    Ok(())
}
