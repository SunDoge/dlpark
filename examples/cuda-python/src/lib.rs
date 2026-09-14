use cudarc::driver::{CudaContext, DevicePtr};
use dlpark::{
    DlpackFlags, TryFromDlpack,
    ffi::{DLDeviceType, DLManagedTensorVersioned},
    interop::cudarc::BorrowedCudaSlice,
    versioned,
};
use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    prelude::*,
};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// An owning cudarc view over a Python-produced contiguous CUDA `float32` tensor.
#[pyclass(unsendable)]
struct CudarcTensorF32 {
    inner: Option<BorrowedCudaSlice<DLManagedTensorVersioned, f32>>,
    device_id: usize,
    device_pointer: u64,
    length: usize,
}

#[pymethods]
impl CudarcTensorF32 {
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

        let context = CudaContext::new(device_id).map_err(runtime_error)?;
        let negotiation_stream = context.new_stream().map_err(runtime_error)?;
        let managed = versioned::Dlpack::extract_with_stream(
            tensor.as_borrowed(),
            &negotiation_stream,
            None,
        )?;

        let descriptor = managed.validate().map_err(runtime_error)?;
        let version = managed.version();
        let flags = managed.flags();
        let compact = descriptor.is_compact().map_err(runtime_error)?;
        let dlpack_pointer =
            unsafe { descriptor.offset_data_ptr::<f32>() }.map_err(runtime_error)? as usize as u64;
        eprintln!("[dlpark/cudarc] received DLPack tensor");
        eprintln!(
            "[dlpark/cudarc] version={}.{} flags={flags:?} read_only={} is_copied={}",
            version.major,
            version.minor,
            flags.contains(DlpackFlags::READ_ONLY),
            flags.contains(DlpackFlags::IS_COPIED)
        );
        eprintln!(
            "[dlpark/cudarc] device=CUDA({}):{} dtype=float32(code={}, bits={}, lanes={})",
            descriptor.device().device_type.0,
            descriptor.device().device_id,
            descriptor.dtype().code.0,
            descriptor.dtype().bits,
            descriptor.dtype().lanes
        );
        eprintln!(
            "[dlpark/cudarc] shape={:?} strides={:?} compact={compact}",
            descriptor.shape(),
            descriptor.strides()
        );
        eprintln!(
            "[dlpark/cudarc] elements={} bytes={} byte_offset={} base_pointer={:p}",
            descriptor.num_elements(),
            descriptor.num_bytes(),
            descriptor.byte_offset(),
            descriptor.data_ptr()
        );
        eprintln!(
            "[dlpark/cudarc] negotiation_stream={:p}",
            negotiation_stream.cu_stream()
        );

        // SAFETY: the Python producer returned a DLPack tensor for the CUDA device
        // discovered above and synchronized it for `negotiation_stream`.
        let borrowed: BorrowedCudaSlice<DLManagedTensorVersioned, f32> =
            unsafe { TryFromDlpack::try_from_dlpack(managed, negotiation_stream.clone()) }
                .map_err(runtime_error)?;

        let stream = borrowed.stream().clone();
        let (device_pointer, fence) = borrowed.device_ptr(&stream);
        drop(fence);
        let length = borrowed.len();
        eprintln!(
            "[dlpark/cudarc] view_device={} view_stream={:p} view_pointer=0x{device_pointer:x} view_length={length}",
            stream.context().ordinal(),
            stream.cu_stream()
        );
        eprintln!(
            "[dlpark/cudarc] pointer_match={}",
            dlpack_pointer == device_pointer
        );
        stream.synchronize().map_err(runtime_error)?;

        Ok(Self {
            inner: Some(borrowed),
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
        let _ = (stream, max_version, dl_device);
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "CudarcTensorF32 only supports zero-copy export",
            ));
        }
        let borrowed = self
            .inner
            .take()
            .ok_or_else(|| PyBufferError::new_err("CudarcTensorF32 was already consumed"))?;
        eprintln!("[dlpark/cudarc] exporting the cudarc view to the next DLPack consumer");
        Ok(borrowed.into_dlpack())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CudarcTensorF32>()?;
    Ok(())
}
