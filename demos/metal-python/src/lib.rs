use dlpark::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned,
        DLPACK_MAJOR_VERSION,
    },
    metadata::{Copied, Dynamic},
    runtime::metal::MetalBuffer,
    versioned,
};
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::sync::Arc;

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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
        if stream.is_some() {
            return Err(PyValueError::new_err(
                "MetalTensor does not accept a stream argument",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "MetalTensor only supports zero-copy export",
            ));
        }
        let descriptor = self.dlpack.validate().map_err(runtime_error)?;
        let device = descriptor.device();
        if let Some(requested) = dl_device
            && requested != (device.device_type.0, device.device_id)
        {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }

        let versioned = max_version.is_some_and(|(major, _)| major >= DLPACK_MAJOR_VERSION);
        if !versioned
            && self
                .dlpack
                .flags()
                .contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED)
        {
            return Err(PyBufferError::new_err(
                "the legacy DLPack ABI cannot describe padded sub-byte elements",
            ));
        }

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
    fn metal_buffer(&self) -> PyResult<u64> {
        Ok(self.dlpack.validate().map_err(runtime_error)?.data_ptr() as usize as u64)
    }

    #[getter]
    fn contents_pointer(&self) -> u64 {
        self.contents_pointer
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MetalTensor>()?;
    Ok(())
}
