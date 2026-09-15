use dlpark::{
    allocation::device::from_device_allocation,
    ffi::{DLDeviceType, DLManagedTensorVersioned},
    runtime::metal::MetalBuffer,
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

/// A host-filled shared Metal buffer exported through DLPack.
#[pyclass(unsendable)]
struct MetalTensorF32 {
    inner: Option<versioned::Dlpack>,
    metal_buffer: u64,
    contents_pointer: u64,
}

#[pymethods]
impl MetalTensorF32 {
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
        let initialized = from_device_allocation::<f32, DLManagedTensorVersioned, _>(
            Box::new(buffer),
            &[rows, columns],
            &[columns, 1],
        )
        .map_err(runtime_error)?;
        let inner = unsafe { initialized.finish() };

        eprintln!("[dlpark/metal] created shared MTLBuffer");
        eprintln!(
            "[dlpark/metal] device=Metal({}):0 shape=[{rows}, {columns}] dtype=float32 bytes={nbytes}",
            DLDeviceType::METAL.0
        );
        eprintln!(
            "[dlpark/metal] metal_buffer=0x{metal_buffer:x} contents_pointer=0x{contents_pointer:x} storage=shared"
        );

        Ok(Self {
            inner: Some(inner),
            metal_buffer,
            contents_pointer,
        })
    }

    #[getter]
    fn metal_buffer(&self) -> u64 {
        self.metal_buffer
    }

    #[getter]
    fn contents_pointer(&self) -> u64 {
        self.contents_pointer
    }

    fn __dlpack_device__(&self) -> (u32, i32) {
        (DLDeviceType::METAL.0, 0)
    }

    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &mut self,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(u32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<versioned::Dlpack> {
        if self.inner.is_none() {
            return Err(PyBufferError::new_err(
                "MetalTensorF32 was already consumed",
            ));
        }
        if stream.is_some() {
            return Err(PyValueError::new_err(
                "MetalTensorF32 does not accept a stream argument",
            ));
        }
        if matches!(max_version, Some((0, _))) {
            return Err(PyBufferError::new_err(
                "MetalTensorF32 exports the versioned DLPack ABI",
            ));
        }
        if dl_device.is_some_and(|device| device != (DLDeviceType::METAL.0, 0)) {
            return Err(PyBufferError::new_err(
                "cross-device copies are not supported",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "MetalTensorF32 only supports zero-copy export",
            ));
        }

        eprintln!("[dlpark/metal] exporting shared MTLBuffer to MLX");
        self.inner
            .take()
            .ok_or_else(|| PyBufferError::new_err("MetalTensorF32 was already consumed"))
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MetalTensorF32>()?;
    Ok(())
}
