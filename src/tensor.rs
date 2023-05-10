pub mod impls;
pub mod traits;

use std::{
    ffi::c_void,
    marker::{PhantomData, PhantomPinned},
};

use traits::{HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides};

use crate::ffi::{self, DataType, Device};

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut ffi::DLManagedTensor) {
    // Reconstruct pointer and destroy it.
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

impl<T> From<T> for TensorWrapper<T>
where
    T: HasShape + HasStrides,
{
    fn from(value: T) -> Self {
        let shape: Shape = value.shape();
        let strides = value.strides();

        Self {
            inner: value,
            shape,
            strides,
        }
    }
}

impl<T> From<&Box<TensorWrapper<T>>> for ffi::DLTensor
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

impl<T> From<TensorWrapper<T>> for ffi::DLManagedTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: TensorWrapper<T>) -> Self {
        let bv = Box::new(value);
        let dl_tensor = ffi::DLTensor::from(&bv);
        let ctx = Box::into_raw(bv);
        // dbg!(ctx);
        Self {
            dl_tensor,
            manager_ctx: ctx as *mut _,
            deleter: Some(deleter_fn::<TensorWrapper<T>>),
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct TensorRef<'a> {
    pub inner: ffi::DLTensor,
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

impl<'a> From<ffi::DLTensor> for TensorRef<'a> {
    fn from(value: ffi::DLTensor) -> Self {
        TensorRef {
            inner: value,
            _marker: PhantomData,
        }
    }
}

impl<'a> TensorRef<'a> {
    pub fn new(
        data: *mut c_void,
        device: Device,
        ndim: i32,
        dtype: DataType,
        shape: *mut i64,
        strides: *mut i64,
        byte_offset: u64,
    ) -> Self {
        let inner = ffi::DLTensor {
            data,
            device,
            ndim,
            dtype,
            shape,
            strides,
            byte_offset,
        };
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> ffi::DLTensor {
        self.inner
    }

    pub fn data(&self) -> *mut c_void {
        self.inner.data
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim as usize
    }

    pub fn dtype(&self) -> DataType {
        self.inner.dtype
    }

    pub fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.inner.shape, self.ndim()) }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        if self.inner.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.inner.strides, self.ndim()) })
        }
    }

    pub fn byte_offset(&self) -> u64 {
        self.inner.byte_offset
    }
}

pub struct ManagerCtx<T> {
    inner: T,
    shape: Shape,
    strides: Option<Strides>,
}

pub struct ManagedTensor<T> {
    pub tensor: ffi::DLTensor,
    pub manager_ctx: Box<T>,
    pub deleter: Option<fn(&mut Self)>,
}

impl Drop for ffi::DLManagedTensor {
    fn drop(&mut self) {
        self.deleter.map(|del_fn| unsafe {
            del_fn(self as *mut ffi::DLManagedTensor);
        });
    }
}

impl Drop for ffi::DLManagedTensorVersioned {
    fn drop(&mut self) {
        self.deleter.map(|del_fn| unsafe {
            del_fn(self as *mut ffi::DLManagedTensorVersioned);
        });
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
