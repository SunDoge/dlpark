use dlpark::{
    Managed, ManagedTensorBase,
    ffi::{
        DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned,
        DLPACK_MAJOR_VERSION,
    },
    metadata::{Copied, Dynamic},
    runtime::metal::MetalBuffer,
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
    buffer: Arc<MetalBuffer>,
    shape: [i64; 2],
    strides: [i64; 2],
    metal_buffer: u64,
    contents_pointer: u64,
}

impl MetalTensor {
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
        let mut initialized = prepared.initialize(Arc::clone(&self.buffer));
        initialized
            .set_data(self.buffer.as_metal_id())
            .set_device(DLDevice::metal(0))
            .set_dtype(DLDataType::F32);
        // SAFETY: the descriptor points at the shared MTLBuffer retained by
        // the managed tensor's `Arc` context.
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

        eprintln!("[dlpark/metal] created shared MTLBuffer");
        eprintln!(
            "[dlpark/metal] device=Metal({}):0 shape=[{rows}, {columns}] dtype=float32 bytes={nbytes}",
            DLDeviceType::METAL.0
        );
        eprintln!(
            "[dlpark/metal] metal_buffer=0x{metal_buffer:x} contents_pointer=0x{contents_pointer:x} storage=shared"
        );

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: [rows, columns],
            strides: [columns, 1],
            metal_buffer,
            contents_pointer,
        })
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        (DLDeviceType::METAL.0, 0)
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
        if let Some(requested) = dl_device
            && requested != (DLDeviceType::METAL.0, 0)
        {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }

        if max_version.is_some_and(|(major, _)| major >= DLPACK_MAJOR_VERSION) {
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
    fn metal_buffer(&self) -> u64 {
        self.metal_buffer
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
