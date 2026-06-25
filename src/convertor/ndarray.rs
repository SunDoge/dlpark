use crate::traits::{InferDataType, TensorLike, TensorView};
use crate::utils::make_row_major_strides;
use crate::{Result, ffi};
use crate::{SafeManagedTensor, SafeManagedTensorVersioned};

use ndarray::{ArrayBase, ArrayViewD, Dimension, RawData, ShapeBuilder};

impl<S, D> TensorLike for ArrayBase<S, D>
where
    S: RawData,
    S::Elem: InferDataType,
    D: Dimension,
{
    type Error = crate::Error;

    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut S::Elem as *mut std::ffi::c_void
    }

    fn shape(&self) -> Vec<i64> {
        self.shape().iter().map(|&x| x as i64).collect()
    }

    fn strides(&self) -> Option<Vec<i64>> {
        Some(self.strides().iter().map(|&x| x as i64).collect())
    }

    fn device(&self) -> Result<ffi::Device> {
        Ok(ffi::Device::CPU)
    }

    fn data_type(&self) -> Result<ffi::DataType> {
        Ok(S::Elem::data_type())
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<'a, A> TryFrom<&'a SafeManagedTensorVersioned> for ArrayViewD<'a, A> {
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensorVersioned) -> Result<Self> {
        let shape: Vec<usize> = value.shape().iter().map(|x| *x as usize).collect();
        let shape = match value.strides() {
            Some(s) => {
                let strides: Vec<usize> = s.iter().map(|x| *x as usize).collect();
                shape.strides(strides)
            }
            None => {
                let strides = make_row_major_strides(value.shape())
                    .into_iter()
                    .map(|x| x as usize)
                    .collect();
                shape.strides(strides)
            }
        };
        unsafe {
            Ok(ArrayViewD::from_shape_ptr(
                shape,
                value.as_slice::<A>()?.as_ptr(),
            ))
        }
    }
}

impl<'a, A> TryFrom<&'a SafeManagedTensor> for ArrayViewD<'a, A> {
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensor) -> Result<Self> {
        let shape: Vec<usize> = value.shape().iter().map(|x| *x as usize).collect();
        let shape = match value.strides() {
            Some(s) => {
                let strides: Vec<usize> = s.iter().map(|x| *x as usize).collect();
                shape.strides(strides)
            }
            None => {
                let strides = make_row_major_strides(value.shape())
                    .into_iter()
                    .map(|x| x as usize)
                    .collect();
                shape.strides(strides)
            }
        };
        unsafe {
            Ok(ArrayViewD::from_shape_ptr(
                shape,
                value.as_slice::<A>()?.as_ptr(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use super::*;

    #[test]
    fn test_dlpack() {
        let arr =
            ndarray::ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let arr2 = arr.clone();
        let mt = SafeManagedTensor::new(arr).unwrap();
        let view = ArrayViewD::<i32>::try_from(&mt).unwrap();
        assert_eq!(view.shape(), arr2.shape());
        assert_eq!(view.strides(), arr2.strides());
        assert_eq!(view.as_slice().unwrap(), arr2.as_slice().unwrap());
    }

    #[test]
    fn test_dlpack_versioned() {
        let arr =
            ndarray::ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let arr2 = arr.clone();
        let mt = SafeManagedTensorVersioned::new(arr).unwrap();
        let view = ArrayViewD::<i32>::try_from(&mt).unwrap();
        assert_eq!(view.shape(), arr2.shape());
        assert_eq!(view.strides(), arr2.strides());
        assert_eq!(view.as_slice().unwrap(), arr2.as_slice().unwrap());
    }
}
