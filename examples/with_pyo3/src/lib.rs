use std::ffi::CString;

use dlpark::{
    dlpack::DLManagedTensor,
    python::PyManagedTensor,
    tensor::{AsDLTensor, ManagedTensor, TensorBuilder, TensorWrapper},
};
use pyo3::{prelude::*, types::PyCapsule};

#[pyfunction]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[pyfunction]
pub fn arange(n: usize, py: Python<'_>) -> PyResult<&PyCapsule> {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tensor: TensorWrapper<_> = v.into();
    let mt: ManagedTensor<_> = tensor.into();
    let dlmt = mt.into_inner();
    dlmt.to_capsule(py)
}

#[pymodule]
fn mylib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
