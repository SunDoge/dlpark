use std::ffi::CString;

use dlpark::{
    python::PyManagedTensor,
    tensor::{ManagedTensor, TensorBuilder},
};
use pyo3::{prelude::*, types::PyCapsule};

#[pyfunction]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub struct Vector {
    pub v: Vec<f32>,
    pub shape: Vec<i64>,
}

impl Drop for Vector {
    fn drop(&mut self) {
        println!("call drop for Vector");
        drop(self);
    }
}

#[pyfunction]
pub fn arange(n: usize, py: Python<'_>) -> PyResult<&PyCapsule> {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let shape = vec![v.len() as i64];
    let mut ve = Vector { v, shape };
    // let strides = vec![1];
    let ptr = ve.v.as_mut_ptr();
    dbg!(ptr);
    let tensor = TensorBuilder::new(ptr as *mut _, ve.shape.as_mut_ptr())
        // .strides(Some(&strides))
        .build();
    dbg!(&tensor, tensor.shape());
    let mt = ManagedTensor::new(ve, tensor.into());
    let dlmt = mt.into_inner();
    let name = CString::new("dltensor").unwrap();
    PyCapsule::new(py, dlmt, Some(name))
}

#[pyfunction]
pub fn arange2(n: usize) -> PyManagedTensor {
    let v: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let shape = vec![v.len() as i64];
    let mut ve = Vector { v, shape };
    // let strides = vec![1];
    let ptr = ve.v.as_mut_ptr();
    let tensor = TensorBuilder::new(ptr as *mut _, ve.shape.as_mut_ptr())
        // .strides(Some(&strides))
        .build();
    let mt = ManagedTensor::new(ve, tensor.into());

    PyManagedTensor {
        inner: Some(mt.into_inner()),
    }
}

#[pymodule]
fn mylib(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(arange2, m)?)?;
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
