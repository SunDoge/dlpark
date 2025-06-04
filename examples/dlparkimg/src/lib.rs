use dlpark::prelude::*;
use image::{ImageBuffer, Rgb};
use pyo3::prelude::*;

#[pyfunction]
fn read_image(filename: &str) -> SafeManagedTensor {
    let img = image::open(filename).unwrap();
    let rgb_img = img.to_rgb8();
    SafeManagedTensor::new(rgb_img).unwrap()
}

#[pyfunction]
fn write_image(filename: &str, tensor: SafeManagedTensor) {
    let rgb_img: ImageBuffer<Rgb<u8>, _> = tensor.as_ref().try_into().unwrap();
    rgb_img.save(filename).unwrap();
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
