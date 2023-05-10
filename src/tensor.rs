pub mod impls;
pub mod traits;

use std::{
    ffi::c_void,
    marker::{PhantomData, PhantomPinned},
};

use traits::{HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides};

use crate::ffi::{self, DataType, Device};

use self::traits::AsTensor;

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

pub struct ManagerCtx<T> {
    inner: T,
    shape: Shape,
    strides: Option<Strides>,
}

impl<T> From<T> for ManagerCtx<T>
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

impl<T> From<&Box<ManagerCtx<T>>> for ffi::DLTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: &Box<ManagerCtx<T>>) -> Self {
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

impl<T> From<ManagerCtx<T>> for ffi::DLManagedTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: ManagerCtx<T>) -> Self {
        let bv = Box::new(value);
        let dl_tensor = ffi::DLTensor::from(&bv);
        let ctx = Box::into_raw(bv);
        // dbg!(ctx);
        Self {
            dl_tensor,
            manager_ctx: ctx as *mut _,
            deleter: Some(deleter_fn::<ManagerCtx<T>>),
        }
    }
}

impl Drop for ffi::DLManagedTensor {
    fn drop(&mut self) {
        if let Some(del_fn) = self.deleter {
            unsafe {
                del_fn(self as *mut ffi::DLManagedTensor);
            }
        }
    }
}

impl Drop for ffi::DLManagedTensorVersioned {
    fn drop(&mut self) {
        if let Some(del_fn) = self.deleter {
            unsafe {
                del_fn(self as *mut ffi::DLManagedTensorVersioned);
            }
        }
    }
}

impl AsTensor for ffi::DLTensor {
    fn data<T>(&self) -> *const T {
        self.data as *const c_void as *const T
    }

    fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.shape, self.ndim()) }
    }

    fn strides(&self) -> Option<&[i64]> {
        if self.strides.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.strides, self.ndim()) })
        }
    }

    fn ndim(&self) -> usize {
        self.ndim as usize
    }

    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn byte_offset(&self) -> u64 {
        self.byte_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_f32() {
        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = ManagerCtx::from(v);
        dbg!(&tensor.shape, &tensor.strides);
        assert_eq!(tensor.shape.len(), 1);
    }
}
