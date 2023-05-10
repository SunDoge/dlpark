use dlpark::tensor::ManagerCtx;
use pyo3::{prelude::*, types::PyDict};

#[pyfunction]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[pyfunction]
pub fn arange(n: usize) -> ManagerCtx<Vec<f32>> {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tensor = ManagerCtx::from(v);
    tensor
}

#[pyfunction]
pub fn tensordict(py: Python<'_>) -> PyResult<&PyDict> {
    let dic = PyDict::new(py);
    let v1: Vec<f32> = vec![1.0; 10];
    let v2: Vec<u8> = vec![2; 10];
    dic.set_item("v1", ManagerCtx::from(v1).to_capsule(py)?)?;
    dic.set_item("v2", ManagerCtx::from(v2).to_capsule(py)?)?;
    Ok(dic)
}

#[pymodule]
fn mylib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(tensordict, m)?)?;
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
