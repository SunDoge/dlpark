use dlpark::{
    AllocationDeleter, DlpackFlags, Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned, DLTensor,
    },
    metadata::{Dynamic},
    python::{
        DlpackExchangeProducer, DlpackExporter, ExportRequest, ImportedDlpack, export_dlpack,
        from_dlpack, install_exchange_api,
    },
};
use objc2::{
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_metal::{
    MTLBuffer as RawMTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResource,
    MTLResourceOptions, MTLStorageMode,
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

/// A Metal allocation backed by an `id<MTLBuffer>` and an exactly-once deleter.
///
/// Locally allocated buffers release an Objective-C retain. Imported buffers
/// carry the original DLPack allocation deleter.
struct MetalBuffer {
    metal_buffer: usize,
    contents: Option<usize>,
    byte_len: usize,
    device: i32,
    _deleter: AllocationDeleter,
}

impl MetalBuffer {
    fn allocate(byte_len: usize, device_id: i32) -> PyResult<Self> {
        if device_id != 0 {
            return Err(PyValueError::new_err(format!(
                "the system-default Metal device has ID 0, got {device_id}"
            )));
        }
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| PyRuntimeError::new_err("no system-default Metal device"))?;
        let buffer = device
            .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Metal buffer allocation failed for {byte_len} bytes"
                ))
            })?;
        let metal_buffer = (&*buffer as *const ProtocolObject<dyn RawMTLBuffer>) as usize;
        let contents = buffer.contents().as_ptr() as usize;
        let retained: Retained<AnyObject> = buffer.into();
        let retained = Retained::into_raw(retained) as usize;
        let deleter = AllocationDeleter::new(move || unsafe {
            let retained = Retained::<AnyObject>::from_raw(retained as *mut _);
            drop(retained);
        });
        Ok(Self {
            metal_buffer,
            contents: Some(contents),
            byte_len,
            device: device_id,
            _deleter: deleter,
        })
    }

    /// Adopts an externally owned `id<MTLBuffer>` without retaining it.
    ///
    /// # Safety
    ///
    /// `metal_buffer` must be a live object implementing `MTLBuffer`, kept
    /// alive until `deleter` runs. The deleter must own that lifetime.
    unsafe fn from_external(metal_buffer: usize, device: i32, deleter: AllocationDeleter) -> Self {
        debug_assert_ne!(metal_buffer, 0);
        let buffer = unsafe { &*(metal_buffer as *const ProtocolObject<dyn RawMTLBuffer>) };
        let contents = match buffer.storageMode() {
            MTLStorageMode::Shared | MTLStorageMode::Managed => {
                Some(buffer.contents().as_ptr() as usize)
            }
            _ => None,
        };
        Self {
            metal_buffer,
            contents,
            byte_len: buffer.length(),
            device,
            _deleter: deleter,
        }
    }

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        let contents = self.contents.expect("locally allocated buffer is shared");
        unsafe { std::slice::from_raw_parts_mut(contents as *mut u8, self.byte_len) }
    }

    fn as_metal_id(&self) -> *mut c_void {
        self.metal_buffer as *mut c_void
    }

    fn contents_at(&self, byte_offset: usize) -> Option<*mut c_void> {
        self.contents?
            .checked_add(byte_offset)
            .map(|address| address as *mut c_void)
    }
}

/// A reusable Metal tensor implementing Python's DLPack protocols.
#[pyclass(unsendable)]
struct MetalTensor {
    buffer: Arc<MetalBuffer>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: DLDataType,
    flags: DlpackFlags,
    length: usize,
    byte_offset: usize,
}

impl MetalTensor {
    fn device(&self) -> DLDevice {
        DLDevice::metal(self.buffer.device)
    }

    fn export<M>(&self) -> PyResult<Managed<M>>
    where
        M: ManagedTensorBase,
    {
        let prepared = Dynamic::new(self.shape.clone(), self.strides.clone())
            .prepare::<M>()
            .map_err(runtime_error)?;
        let mut initialized = prepared.initialize(Arc::clone(&self.buffer));
        initialized
            .set_data(self.buffer.as_metal_id())
            .set_device(self.device())
            .set_dtype(self.dtype)
            .set_byte_offset(self.byte_offset as u64);
        initialized.set_flags(self.flags).map_err(runtime_error)?;
        // SAFETY: metadata is copied, and Arc<MetalBuffer> retains the MTLBuffer
        // custom deleter until the final managed export is released.
        Ok(unsafe { initialized.finish() })
    }

    fn tensor_view(&self) -> DLTensor {
        DLTensor {
            data: self.buffer.as_metal_id(),
            device: self.device(),
            ndim: i32::try_from(self.shape.len())
                .expect("validated DLPack rank continues to fit i32"),
            dtype: self.dtype,
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
            byte_offset: self.byte_offset as u64,
        }
    }

    fn compact_metadata(shape: Vec<usize>) -> PyResult<(Vec<i64>, Vec<i64>, usize)> {
        let length = shape
            .iter()
            .try_fold(1_usize, |length, &dimension| length.checked_mul(dimension));
        let length =
            length.ok_or_else(|| PyValueError::new_err("shape element count overflows"))?;
        let shape = shape
            .into_iter()
            .map(|dimension| {
                i64::try_from(dimension)
                    .map_err(|_| PyValueError::new_err("shape dimension does not fit i64"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut strides = vec![0_i64; shape.len()];
        let mut stride = 1_i64;
        for (dimension, output) in shape.iter().zip(&mut strides).rev() {
            *output = stride;
            stride = stride
                .checked_mul(*dimension)
                .ok_or_else(|| PyValueError::new_err("compact stride overflows i64"))?;
        }
        Ok((shape, strides, length))
    }

    fn empty_inner(shape: Vec<usize>, device_id: i32) -> PyResult<Self> {
        let (shape, strides, length) = Self::compact_metadata(shape)?;
        let byte_len = length
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| PyValueError::new_err("buffer byte length overflows"))?;
        let buffer = Arc::new(MetalBuffer::allocate(byte_len, device_id)?);

        eprintln!("[dlpark/metal] allocated shared MTLBuffer");
        eprintln!(
            "[dlpark/metal] device=Metal({}):{device_id} shape={shape:?} strides={strides:?} dtype=float32 bytes={byte_len}",
            DLDeviceType::METAL.0
        );
        eprintln!(
            "[dlpark/metal] metal_buffer={:p} contents_pointer={:p} storage=shared",
            buffer.as_metal_id(),
            buffer
                .contents_at(0)
                .expect("shared buffer has host contents")
        );

        Ok(Self {
            buffer,
            shape,
            strides,
            dtype: DLDataType::F32,
            flags: DlpackFlags::empty(),
            length,
            byte_offset: 0,
        })
    }
}

#[pymethods]
impl MetalTensor {
    #[classmethod]
    #[pyo3(signature = (shape, device_id=0))]
    fn empty(_class: &Bound<'_, PyType>, shape: Vec<usize>, device_id: i32) -> PyResult<Self> {
        Self::empty_inner(shape, device_id)
    }

    #[classmethod]
    fn from_values(
        _class: &Bound<'_, PyType>,
        values: Vec<f32>,
        rows: usize,
        columns: usize,
    ) -> PyResult<Self> {
        let mut tensor = Self::empty_inner(vec![rows, columns], 0)?;
        if values.len() != tensor.length {
            return Err(PyValueError::new_err(format!(
                "shape [{rows}, {columns}] contains {} elements, got {} values",
                tensor.length,
                values.len()
            )));
        }
        let buffer =
            Arc::get_mut(&mut tensor.buffer).expect("a newly allocated Metal buffer has one owner");
        let bytes = buffer.as_mut_bytes();
        for (index, value) in values.into_iter().enumerate() {
            let start = index * size_of::<f32>();
            bytes[start..start + size_of::<f32>()].copy_from_slice(&value.to_ne_bytes());
        }
        Ok(tensor)
    }

    #[classmethod]
    fn from_dlpack(_class: &Bound<'_, PyType>, tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        let managed = from_dlpack(tensor.as_borrowed(), None, None)?;
        let abi = match &managed {
            ImportedDlpack::Legacy(_) => "legacy".to_owned(),
            ImportedDlpack::Versioned(tensor) => {
                let version = tensor.version();
                format!("{}.{}", version.major, version.minor)
            }
        };
        let source_flags = managed.flags();
        let (device, shape, strides, dtype, length, byte_len, byte_offset, metal_buffer, compact) = {
            let descriptor = managed.validate().map_err(runtime_error)?;
            let device = descriptor.device();
            if device.device_type != DLDeviceType::METAL {
                return Err(PyValueError::new_err(format!(
                    "expected a Metal tensor, got {:?}",
                    device.device_type
                )));
            }
            if device.device_id != 0 {
                return Err(PyValueError::new_err(format!(
                    "expected Metal device 0, got {}",
                    device.device_id
                )));
            }
            let compact = descriptor.is_compact().map_err(runtime_error)?;
            if !descriptor.dtype().is::<f32>() {
                return Err(PyValueError::new_err(format!(
                    "expected float32, got {:?}",
                    descriptor.dtype()
                )));
            }
            if !compact {
                return Err(PyValueError::new_err("expected a compact Metal tensor"));
            }
            if descriptor.data_ptr().is_null() {
                return Err(PyValueError::new_err(
                    "Metal DLPack data must contain an MTLBuffer object",
                ));
            }
            (
                device,
                descriptor.shape().to_vec(),
                descriptor
                    .strides_or_compact()
                    .map_err(runtime_error)?
                    .into_owned(),
                descriptor.dtype(),
                descriptor.num_elements(),
                descriptor.num_bytes(),
                usize::try_from(descriptor.byte_offset())
                    .map_err(|_| PyValueError::new_err("byte offset does not fit usize"))?,
                descriptor.data_ptr() as usize,
                compact,
            )
        };

        let deleter = managed.into_deleter();
        let buffer = Arc::new(unsafe {
            MetalBuffer::from_external(metal_buffer, device.device_id, deleter)
        });
        let end = byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| PyValueError::new_err("Metal buffer range overflows usize"))?;
        if end > buffer.byte_len {
            return Err(PyValueError::new_err(format!(
                "tensor byte range ends at {end}, but MTLBuffer length is {}",
                buffer.byte_len
            )));
        }
        let flags = source_flags.difference(DlpackFlags::IS_COPIED);

        eprintln!("[dlpark/metal] imported zero-copy Metal buffer");
        eprintln!(
            "[dlpark/metal] abi={abi} flags={source_flags:?} device=Metal({}):{} dtype=float32 shape={shape:?} strides={strides:?} compact={compact}",
            device.device_type.0, device.device_id
        );
        eprintln!(
            "[dlpark/metal] elements={length} bytes={byte_len} byte_offset={byte_offset} metal_buffer=0x{metal_buffer:x} contents_pointer={:?}",
            buffer.contents_at(byte_offset)
        );

        Ok(Self {
            buffer,
            shape,
            strides,
            dtype,
            flags,
            length,
            byte_offset,
        })
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        self.device().into()
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
        export_dlpack(self, py, request)
    }

    #[getter]
    fn metal_buffer(&self) -> u64 {
        self.buffer.metal_buffer as u64
    }

    #[getter]
    fn contents_pointer(&self) -> Option<u64> {
        self.buffer
            .contents_at(self.byte_offset)
            .map(|pointer| pointer as usize as u64)
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }
}

impl DlpackExporter for MetalTensor {
    fn device(&self) -> DLDevice {
        self.device()
    }

    fn flags(&self) -> DlpackFlags {
        self.flags
    }

    fn prepare_export(&self, _py: Python<'_>, request: &ExportRequest<'_>) -> PyResult<()> {
        if request.stream().is_some() {
            return Err(PyValueError::new_err(
                "MetalTensor does not accept a stream argument",
            ));
        }
        Ok(())
    }

    fn export_legacy(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensor>> {
        self.export()
    }

    fn export_versioned(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensorVersioned>> {
        self.export()
    }
}

// SAFETY: tensor_view points into immutable metadata and a live MTLBuffer owned
// by self. Managed exports retain Arc<MetalBuffer>. This demo submits no Metal
// command-buffer work, so its type-wide current work stream is null.
unsafe impl DlpackExchangeProducer for MetalTensor {
    const HAS_DLTENSOR_VIEW: bool = true;

    fn managed_tensor_no_sync(
        &self,
        _py: Python<'_>,
    ) -> PyResult<Managed<DLManagedTensorVersioned>> {
        self.export()
    }

    fn tensor_view_no_sync(&self, _py: Python<'_>) -> PyResult<DLTensor> {
        Ok(self.tensor_view())
    }

    fn current_work_stream(_py: Python<'_>, device: DLDevice) -> PyResult<*mut c_void> {
        if device.device_type != DLDeviceType::METAL || device.device_id != 0 {
            return Err(PyValueError::new_err(format!(
                "MetalTensor uses Metal device 0, requested {:?}:{}",
                device.device_type, device.device_id
            )));
        }
        Ok(std::ptr::null_mut())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MetalTensor>()?;
    install_exchange_api::<MetalTensor>(module.py())?;
    Ok(())
}
