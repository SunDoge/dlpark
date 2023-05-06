use dlpark::tensor::TensorWrapper;
use pyo3::prelude::*;

#[pyfunction]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[pyfunction]
pub fn arange(n: usize, py: Python<'_>) -> PyResult<&PyAny> {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tensor = TensorWrapper::from(v);
    let ptr = tensor.to_capsule();
    unsafe { py.from_owned_ptr_or_err(ptr) }
}

#[pymodule]
fn mylib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
