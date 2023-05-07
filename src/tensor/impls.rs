use super::{
    traits::{HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides},
    Shape,
};
use crate::dlpack::{DataType, Device};
use std::ffi::c_void;

impl<T> HasDevice for Vec<T> {
    fn device(&self) -> Device {
        Device::CPU
    }
}

impl<T> HasShape for Vec<T> {
    fn shape(&self) -> Shape {
        Shape::Owned(vec![self.len() as i64])
    }
}

impl<T> HasData for Vec<T> {
    fn data(&self) -> *mut c_void {
        self.as_ptr() as *const c_void as *mut c_void
    }
}

impl HasDtype for Vec<f32> {
    fn dtype(&self) -> DataType {
        DataType::F32
    }
}

impl HasDtype for Vec<u8> {
    fn dtype(&self) -> DataType {
        DataType::U8
    }
}

impl HasDtype for Vec<i64> {
    fn dtype(&self) -> DataType {
        DataType::I64
    }
}

impl<T> HasStrides for Vec<T> {}
impl<T> HasByteOffset for Vec<T> {
    fn byte_offset(&self) -> u64 {
        0
    }
}
