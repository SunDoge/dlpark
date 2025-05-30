use snafu::ensure;

use crate::error::{Error, NonContiguousSnafu};
use crate::{ffi::InferDataType, traits::ContiguousLayout, traits::TensorLike};

use crate::ffi;
use crate::versioned;

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

impl<'a, A> TryFrom<&'a versioned::SafeManagedTensor> for &'a [A] {
    type Error = Error;
    fn try_from(value: &'a versioned::SafeManagedTensor) -> Result<Self, Self::Error> {
        ensure!(
            value.is_contiguous(),
            NonContiguousSnafu {
                shape: value.shape(),
                strides: value.strides().expect("fuck")
            }
        );
        unsafe { value.as_slice::<A>() }
    }
}

#[cfg(test)]
mod tests {
    use crate::versioned::SafeManagedTensor;

    use super::*;

    #[test]
    fn test_convert() {
        let v = vec![1.0f32, 2., 3.];
        let t = SafeManagedTensor::new(v);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.strides(), None);

        let s: &[f32] = (&t).try_into().expect("fuck");
        assert_eq!(s, &[1.0f32, 2., 3.]);
    }
}
