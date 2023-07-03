use dlpark::prelude::*;
use pyo3::{prelude::*, types::PyDict};

#[pyfunction]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[pyfunction]
pub fn arange(n: usize) -> ManagerCtx<Vec<f32>> {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let tensor = ManagerCtx::new(v);
    tensor
}

#[pyfunction]
pub fn tensordict(py: Python<'_>) -> PyResult<&PyDict> {
    let dic = PyDict::new(py);
    let v1: Vec<f32> = vec![1.0; 10];
    let v2: Vec<u8> = vec![2; 10];
    dic.set_item("v1", ManagerCtx::new(v1).into_py(py))?;
    dic.set_item("v2", ManagerCtx::new(v2).into_py(py))?;
    Ok(dic)
}

#[pyfunction]
pub fn print_tensor(tensor: ManagedTensor) {
    dbg!(tensor.shape(), tensor.strides(), tensor.dtype(), tensor.device());
    assert!(tensor.dtype() == DataType::F32);
    dbg!(tensor.as_slice::<f32>());
}

#[pymodule]
fn mylib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(tensordict, m)?)?;
    m.add_function(wrap_pyfunction!(print_tensor, m)?)?;
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
