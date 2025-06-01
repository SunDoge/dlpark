// use crate::error::Error;

// use crate::safe_managed_tensor::SafeManagedTensorVersioned;
// use crate::{
//     data_type::{DataType, InferDataType},
//     device::Device,
//     manager_context::TensorLike,
//     memory_layout::StridedLayout,
//     utils::make_contiguous_strides,
// };

use crate::ffi;
use crate::traits::{InferDataType, StridedLayout, TensorLike, TensorView};
use crate::utils::make_row_major_strides;
use crate::{SafeManagedTensor, SafeManagedTensorVersioned};

use ndarray::{ArrayBase, ArrayViewD, Dimension, RawData, ShapeBuilder};

impl<S, D> TensorLike<StridedLayout> for ArrayBase<S, D>
where
    S: RawData,
    S::Elem: InferDataType,
    D: Dimension,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut S::Elem as *mut std::ffi::c_void
    }

    fn memory_layout(&self) -> StridedLayout {
        let mut layout = StridedLayout::new_with_ndim(self.ndim());
        for i in 0..self.ndim() {
            layout.shape_mut()[i] = self.shape()[i] as i64;
            layout.strides_mut()[i] = self.strides()[i] as i64;
        }
        layout
    }

    fn device(&self) -> ffi::Device {
        ffi::Device::CPU
    }

    fn data_type(&self) -> ffi::DataType {
        S::Elem::data_type()
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<'a, A> TryFrom<&'a SafeManagedTensorVersioned> for ArrayViewD<'a, A> {
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensorVersioned) -> Result<Self, Self::Error> {
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

    fn try_from(value: &'a SafeManagedTensor) -> Result<Self, Self::Error> {
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
