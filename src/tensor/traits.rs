use std::ffi::c_void;

use crate::ffi::{DataType, Device};

use super::{Shape, Strides};

pub trait HasData {
    fn data(&self) -> *mut c_void;
}

pub trait HasShape {
    fn shape(&self) -> Shape;
}

pub trait HasStrides {
    fn strides(&self) -> Option<Strides> {
        None
    }
}

pub trait HasByteOffset {
    fn byte_offset(&self) -> u64;
}

pub trait HasDevice {
    fn device(&self) -> Device;
}

pub trait HasDtype {
    fn dtype(&self) -> DataType;
}

pub trait InferDtype {
    fn infer_dtype() -> DataType;
}

pub trait AsTensor {
    fn data<T>(&self) -> *const T;
    fn shape(&self) -> &[i64];
    fn strides(&self) -> Option<&[i64]>;
    fn ndim(&self) -> usize;
    fn device(&self) -> Device;
    fn dtype(&self) -> DataType;
    fn byte_offset(&self) -> u64;
}
