use std::ffi::c_void;

use crate::dlpack::{DLManagedTensor, DLTensor, DataType, Device};

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut DLManagedTensor) {
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    // dbg!(ctx);
    drop(unsafe { Box::from_raw(ctx) });
}

#[derive(Debug)]
pub enum Shape {
    Borrowed(*mut i64, usize),
    Owned(Vec<i64>),
}

impl Shape {
    pub fn as_ptr(&self) -> *mut i64 {
        match self {
            Self::Borrowed(ref ptr, _) => *ptr,
            Self::Owned(ref v) => v.as_ptr() as *mut i64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(_, len) => *len,
            Self::Owned(ref v) => v.len(),
        }
    }

    pub fn ndim(&self) -> i32 {
        self.len() as i32
    }
}

#[derive(Debug)]
pub enum Strides {
    Borrowed(*mut i64),
    Owned(Vec<i64>),
}

impl Strides {
    pub fn as_ptr(&self) -> *mut i64 {
        match self {
            Self::Borrowed(ref ptr) => *ptr,
            Self::Owned(ref v) => v.as_ptr() as *mut i64,
        }
    }
}

pub struct TensorWrapper<T> {
    inner: T,
    shape: Shape,
    strides: Option<Strides>,
}

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

impl<T> HasStrides for Vec<T> {}
impl<T> HasByteOffset for Vec<T> {
    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<T> From<T> for TensorWrapper<T>
where
    T: HasShape + HasStrides,
{
    fn from(value: T) -> Self {
        // let bv = Box::new(value);
        // let bv = value;
        let shape: Shape = value.shape();
        let strides = value.strides();

        Self {
            inner: value,
            shape,
            strides,
        }
    }
}

impl<T> From<&Box<TensorWrapper<T>>> for DLTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: &Box<TensorWrapper<T>>) -> Self {
        Self {
            data: value.inner.data(),
            device: value.inner.device(),
            ndim: value.shape.ndim(),
            shape: value.shape.as_ptr(),
            dtype: value.inner.dtype(),
            strides: match value.strides {
                Some(ref strides) => strides.as_ptr(),
                None => std::ptr::null_mut(),
            },
            byte_offset: value.inner.byte_offset(),
        }
    }
}

impl<T> From<TensorWrapper<T>> for DLManagedTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: TensorWrapper<T>) -> Self {
        let bv = Box::new(value);
        let dl_tensor = DLTensor::from(&bv);
        let ctx = Box::into_raw(bv);
        // dbg!(ctx);
        Self {
            dl_tensor,
            manager_ctx: ctx as *mut _,
            deleter: Some(deleter_fn::<TensorWrapper<T>>),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_f32() {
        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = TensorWrapper::from(v);
        dbg!(&tensor.shape, &tensor.strides);
        assert_eq!(tensor.shape.len(), 1);
    }
}
