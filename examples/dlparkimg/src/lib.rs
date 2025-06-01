use dlpark::prelude::*;
use image::{ImageBuffer, Rgb};
use pyo3::prelude::*;

#[pyfunction]
fn read_image(filename: &str) -> SafeManagedTensorVersioned {
    let img = image::open(filename).unwrap();
    let rgb_img = img.to_rgb8();
    SafeManagedTensorVersioned::new(rgb_img)
}

#[pyfunction]
fn write_image(filename: &str, tensor: SafeManagedTensorVersioned) {
    let rgb_img: ImageBuffer<Rgb<u8>, _> = tensor.as_ref().try_into().unwrap();
    rgb_img.save(filename).unwrap();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dlparkimg(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    Ok(())
}
