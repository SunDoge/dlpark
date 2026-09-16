use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned,
    },
    metadata::{Copied, Dynamic},
    python::{DlpackExchangeProducer, ExportRequest, install_exchange_api},
    versioned,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer as RawMTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use pyo3::{
    Bound, Py, PyAny, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Demo-owned shared Metal storage. dlpark only owns its resulting DLPack
/// manager context; allocating device resources belongs to the application.
struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn RawMTLBuffer>>,
    contents: NonNull<c_void>,
    nbytes: usize,
}

// Apple documents MTLBuffer as thread-safe. Mutable host access requires
// `&mut self` below.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    fn allocate(nbytes: usize) -> PyResult<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| PyRuntimeError::new_err("no system-default Metal device"))?;
        let buffer = device
            .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Metal buffer allocation failed for {nbytes} bytes"
                ))
            })?;
        let contents = buffer.contents();
        Ok(Self {
            buffer,
            contents,
            nbytes,
        })
    }

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.contents.cast().as_ptr(), self.nbytes) }
    }

    fn contents_ptr(&self) -> *mut c_void {
        self.contents.as_ptr()
    }

    fn as_metal_id(&self) -> *mut c_void {
        (&*self.buffer as *const ProtocolObject<dyn RawMTLBuffer>)
            .cast_mut()
            .cast()
    }
}

/// A reusable Metal tensor object implementing Python's DLPack protocol.
#[pyclass(unsendable)]
struct MetalTensor {
    dlpack: Arc<versioned::Dlpack>,
    contents_pointer: u64,
}

impl MetalTensor {
    fn export<M>(&self) -> PyResult<Managed<M>>
    where
        M: ManagedTensorBase,
    {
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let prepared = Dynamic::new(
            Copied(descriptor.shape().to_vec()),
            Copied(descriptor.strides().unwrap_or(&[]).to_vec()),
        )
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
        // `Arc<versioned::Dlpack>` in its manager context.
        Ok(unsafe { initialized.finish() })
    }
}

#[pymethods]
impl MetalTensor {
    #[new]
    fn new(values: Vec<f32>, rows: usize, columns: usize) -> PyResult<Self> {
        let length = rows
            .checked_mul(columns)
            .ok_or_else(|| PyValueError::new_err("shape element count overflows usize"))?;
        if values.len() != length {
            return Err(PyValueError::new_err(format!(
                "shape [{rows}, {columns}] contains {length} elements, got {} values",
                values.len()
            )));
        }
        let nbytes = length
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("buffer byte length overflows usize"))?;
        let rows =
            i64::try_from(rows).map_err(|_| PyValueError::new_err("row count does not fit i64"))?;
        let columns = i64::try_from(columns)
            .map_err(|_| PyValueError::new_err("column count does not fit i64"))?;

        let mut buffer = MetalBuffer::allocate(nbytes).map_err(runtime_error)?;
        let bytes = buffer.as_mut_bytes();
        for (index, value) in values.into_iter().enumerate() {
            let start = index * size_of::<f32>();
            bytes[start..start + size_of::<f32>()].copy_from_slice(&value.to_ne_bytes());
        }
        let metal_buffer = buffer.as_metal_id() as usize as u64;
        let contents_pointer = buffer.contents_ptr() as usize as u64;
        let prepared = Dynamic::new(Copied(vec![rows, columns]), Copied(vec![columns, 1]))
            .prepare::<DLManagedTensorVersioned>()
            .map_err(runtime_error)?;
        let mut initialized = prepared.initialize(Box::new(buffer));
        initialized
            .set_data(metal_buffer as usize as *mut std::ffi::c_void)
            .set_device(DLDevice::metal(0))
            .set_dtype(DLDataType::F32);
        // SAFETY: the descriptor points at the MTLBuffer owned by its boxed
        // manager context, and its shape and strides are copied into the
        // managed allocation.
        let dlpack = unsafe { initialized.finish() };

        eprintln!("[dlpark/metal] created shared MTLBuffer");
        eprintln!(
            "[dlpark/metal] device=Metal({}):0 shape=[{rows}, {columns}] dtype=float32 bytes={nbytes}",
            DLDeviceType::METAL.0
        );
        eprintln!(
            "[dlpark/metal] metal_buffer=0x{metal_buffer:x} contents_pointer=0x{contents_pointer:x} storage=shared"
        );

        Ok(Self {
            dlpack: Arc::new(dlpack),
            contents_pointer,
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
        if request.stream().is_some() {
            return Err(PyValueError::new_err(
                "MetalTensor does not accept a stream argument",
            ));
        }
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let device = descriptor.device();
        request.export_zero_copy(
            py,
            device,
            self.dlpack.flags(),
            || self.export::<DLManagedTensor>(),
            || self.export::<DLManagedTensorVersioned>(),
        )
    }

    #[getter]
    fn metal_buffer(&self) -> PyResult<u64> {
        Ok(self.dlpack.validate().map_err(runtime_error)?.data_ptr() as usize as u64)
    }

    #[getter]
    fn contents_pointer(&self) -> u64 {
        self.contents_pointer
    }
}

unsafe impl DlpackExchangeProducer for MetalTensor {
    fn managed_tensor_no_sync(
        &self,
        _py: Python<'_>,
    ) -> PyResult<Managed<DLManagedTensorVersioned>> {
        self.export()
    }

    fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<dlpark::ffi::DLTensor> {
        Ok(*unsafe { self.dlpack.tensor() })
    }

    fn current_work_stream(
        _py: Python<'_>,
        device: DLDevice,
    ) -> PyResult<*mut c_void> {
        if device.device_type != DLDeviceType::METAL || device.device_id != 0 {
            return Err(PyValueError::new_err(format!(
                "MetalTensor uses Metal device 0, requested {:?}:{}",
                device.device_type, device.device_id
            )));
        }
        // The demo only exports host-filled shared buffers and has no pending
        // Metal command queue work.
        Ok(std::ptr::null_mut())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MetalTensor>()?;
    install_exchange_api::<MetalTensor>(module.py())?;
    Ok(())
}
