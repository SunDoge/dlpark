use dlpark::{allocation::fixed, ffi::DLManagedTensorVersioned, Foreign, Local};
use image::{ImageBuffer, Rgb};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn read_image(filename: &str) -> PyResult<Local<DLManagedTensorVersioned>> {
    let img = image::open(filename).map_err(|err| PyIOError::new_err(err.to_string()))?;
    let rgb_img = img.to_rgb8();
    let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> = Box::new(rgb_img)
        .try_into()
        .map_err(|err: dlpark::metadata::Error| PyValueError::new_err(err.to_string()))?;
    Ok(unsafe { initialized.finish() })
}

#[pyfunction]
fn write_image(filename: &str, tensor: Foreign<DLManagedTensorVersioned>) -> PyResult<()> {
    // SAFETY: this extension accepts tensors through the Python DLPack
    // protocol and relies on the producer to provide a valid descriptor.
    let tensor = unsafe { tensor.assume_valid() };
    let rgb_img: ImageBuffer<Rgb<u8>, _> = (&tensor)
        .try_into()
        .map_err(|err: dlpark::interop::image::Error| PyValueError::new_err(err.to_string()))?;
    rgb_img
        .save(filename)
        .map_err(|err| PyIOError::new_err(err.to_string()))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    Ok(())
}
