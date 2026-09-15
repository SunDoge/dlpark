use dlpark::{
    allocation::device::from_device_allocation,
    ffi::{DLDeviceType, DLManagedTensorVersioned},
    python::DlpackProducer,
    runtime::metal::MetalBuffer,
};
use pyo3::{
    Bound, PyClassInitializer, PyResult,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A host-filled shared Metal buffer exported through DLPack.
#[pyclass(extends = DlpackProducer, unsendable)]
struct MetalTensorF32 {
    metal_buffer: u64,
    contents_pointer: u64,
}

#[pymethods]
impl MetalTensorF32 {
    #[new]
    fn new(values: Vec<f32>, rows: usize, columns: usize) -> PyResult<PyClassInitializer<Self>> {
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
        // The shared buffer was filled by the host and no Metal command queue
        // has outstanding work against it.
        let producer = unsafe { DlpackProducer::without_stream(inner) }.map_err(runtime_error)?;

        eprintln!("[dlpark/metal] created shared MTLBuffer");
        eprintln!(
            "[dlpark/metal] device=Metal({}):0 shape=[{rows}, {columns}] dtype=float32 bytes={nbytes}",
            DLDeviceType::METAL.0
        );
        eprintln!(
            "[dlpark/metal] metal_buffer=0x{metal_buffer:x} contents_pointer=0x{contents_pointer:x} storage=shared"
        );

        Ok(PyClassInitializer::from(producer).add_subclass(Self {
            metal_buffer,
            contents_pointer,
        }))
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
    module.add_class::<DlpackProducer>()?;
    module.add_class::<MetalTensorF32>()?;
    Ok(())
}
