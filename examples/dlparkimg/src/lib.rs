use dlpark::{
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    Dlpack,
};
use image::{ImageBuffer, Rgb};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn read_image(filename: &str) -> PyResult<Dlpack<DLManagedTensor>> {
    let img = image::open(filename).map_err(|err| PyIOError::new_err(err.to_string()))?;
    let rgb_img = img.to_rgb8();
    Ok(Dlpack::from(rgb_img))
}

#[pyfunction]
fn read_image_versioned(filename: &str) -> PyResult<Dlpack<DLManagedTensorVersioned>> {
    let img = image::open(filename).map_err(|err| PyIOError::new_err(err.to_string()))?;
    let rgb_img = img.to_rgb8();
    Ok(Dlpack::from(rgb_img))
}

#[pyfunction]
fn write_image(filename: &str, tensor: Dlpack<DLManagedTensor>) -> PyResult<()> {
    let rgb_img: ImageBuffer<Rgb<u8>, _> = (&tensor)
        .try_into()
        .map_err(|err: dlpark::interop::image::Error| PyValueError::new_err(err.to_string()))?;
    rgb_img
        .save(filename)
        .map_err(|err| PyIOError::new_err(err.to_string()))
}

#[pyfunction]
fn write_image_versioned(filename: &str, tensor: Dlpack<DLManagedTensorVersioned>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(read_image_versioned, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_versioned, m)?)?;
    Ok(())
}
