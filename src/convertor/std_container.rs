use snafu::ensure;

use crate::error::{Error, NonContiguousSnafu};
use crate::{ffi::InferDataType, traits::ContiguousLayout, traits::TensorLike};

use crate::ffi;
use crate::ffi::TensorView;
use crate::{SafeManagedTensor, SafeManagedTensorVersioned};

impl<A> TensorLike<ContiguousLayout> for Vec<A>
where
    A: InferDataType,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut A as *mut _
    }

    fn data_type(&self) -> ffi::DataType {
        A::infer_dtype()
    }

    fn memory_layout(&self) -> ContiguousLayout {
        ContiguousLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> ffi::Device {
        ffi::Device::CPU
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_convert() {
        let v = vec![1.0f32, 2., 3.];
        let t = SafeManagedTensorVersioned::new(v);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.strides(), None);

        let s: &[f32] = t.as_slice_contiguous().expect("fuck");
        assert_eq!(s, &[1.0f32, 2., 3.]);
    }
}
