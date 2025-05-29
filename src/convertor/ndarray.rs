use crate::{
    data_type::{DataType, InferDataType},
    device::Device,
    manager_context::TensorLike,
    memory_layout::StridedLayout,
    owned_tensor::OwnedTensor,
    utils::make_contiguous_strides,
};

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

    fn device(&self) -> Device {
        Device::CPU
    }

    fn data_type(&self) -> DataType {
        S::Elem::infer_dtype()
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<'a, A> From<&'a OwnedTensor> for ArrayViewD<'a, A> {
    // TODO: The data type is not checked, it's unsafe.
    fn from(tensor: &'a OwnedTensor) -> Self {
        let shape: Vec<usize> = tensor.shape().iter().map(|x| *x as usize).collect();
        let shape = match tensor.strides() {
            Some(s) => {
                let strides: Vec<usize> = s.iter().map(|x| *x as usize).collect();
                shape.strides(strides)
            }
            None => {
                let strides = make_contiguous_strides(tensor.shape())
                    .into_iter()
                    .map(|x| x as usize)
                    .collect();
                shape.strides(strides)
            }
        };
        unsafe { ArrayViewD::from_shape_ptr(shape, tensor.as_ptr()) }
    }
}
