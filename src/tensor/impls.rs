use super::{
    traits::{HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides, InferDtype},
    Shape,
};
use crate::ffi::{DataType, Device};
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

impl<T> HasDtype for Vec<T>
where
    T: InferDtype,
{
    fn dtype(&self) -> DataType {
        T::infer_dtype()
    }
}
impl<T> HasStrides for Vec<T> {}
impl<T> HasByteOffset for Vec<T> {
    fn byte_offset(&self) -> u64 {
        0
    }
}

impl InferDtype for f32 {
    fn infer_dtype() -> DataType {
        DataType::F32
    }
}

impl InferDtype for f64 {
    fn infer_dtype() -> DataType {
        DataType::F64
    }
}

impl InferDtype for i64 {
    fn infer_dtype() -> DataType {
        DataType::I64
    }
}

impl InferDtype for i32 {
    fn infer_dtype() -> DataType {
        DataType::I32
    }
}

impl InferDtype for u8 {
    fn infer_dtype() -> DataType {
        DataType::U8
    }
}

impl InferDtype for bool {
    fn infer_dtype() -> DataType {
        DataType::BOOL
    }
}
