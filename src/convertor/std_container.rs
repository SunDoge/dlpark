use crate::Result;
use crate::ffi;
use crate::traits::{InferDataType, RowMajorCompactLayout, TensorLike};

impl<A> TensorLike<RowMajorCompactLayout> for Vec<A>
where
    A: InferDataType,
{
    type Error = crate::Error;
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut A as *mut _
    }

    fn data_type(&self) -> Result<ffi::DataType> {
        Ok(A::data_type())
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> Result<ffi::Device> {
        Ok(ffi::Device::CPU)
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<A> TensorLike<RowMajorCompactLayout> for Box<[A]>
where
    A: InferDataType,
{
    type Error = crate::Error;
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut A as *mut _
    }

    fn data_type(&self) -> Result<ffi::DataType> {
        Ok(A::data_type())
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> Result<ffi::Device> {
        Ok(ffi::Device::CPU)
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_vec() {
        let v = vec![1.0f32, 2., 3.];
        let t = SafeManagedTensorVersioned::new(v).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.strides(), None);

        let s: &[f32] = t.as_slice_contiguous().unwrap();
        assert_eq!(s, &[1.0f32, 2., 3.]);
    }

    #[test]
    fn test_boxed_slice() {
        let v: Box<[f32]> = vec![1.0f32, 2., 3.].into_boxed_slice();
        let t = SafeManagedTensorVersioned::new(v).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.strides(), None);

        let s: &[f32] = t.as_slice_contiguous().unwrap();
        assert_eq!(s, &[1.0f32, 2., 3.]);
    }
}
