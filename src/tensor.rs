pub mod impls;
pub mod traits;

use std::ptr::NonNull;

use crate::ffi;

use self::traits::{TensorView, ToDLPack, ToTensor};

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut ffi::DLManagedTensor) {
    // Reconstruct pointer and destroy it.
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    drop(unsafe { Box::from_raw(ctx) });
    // ctx.drop_in_place();
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

    pub fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    pub fn num_elements(&self) -> usize {
        self.as_slice().iter().fold(1, |acc, x| acc * (*x as usize))
    }
}

/// If it is borrowed, the length should be `tensor.ndim()`
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

    pub fn as_slice(&self, len: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), len) }
    }
}

pub struct ManagerCtx<T> {
    inner: T,
    shape: Shape,
    strides: Option<Strides>,
    // The ctx should hold DLManagedTensor, so that the tensor can be freed.
    // tensor: Option<ffi::DLManagedTensor>,
    tensor: ffi::DLManagedTensor,
}

// impl<T> ManagerCtx<T> {
//     pub fn set_tensor(&mut self, tensor: ffi::DLManagedTensor) {
//         self.tensor = Some(tensor);
//     }

//     pub fn get_tensor_ptr(&self) -> NonNull<ffi::DLManagedTensor> {
//         NonNull::from(self.tensor.as_ref().unwrap())
//     }
// }

impl<T> ManagerCtx<T>
where
    T: ToTensor,
{
    pub fn new(inner: T) -> Self {
        let shape: Shape = inner.shape();
        let strides = inner.strides();
        dbg!(&shape, &strides);
        Self {
            inner,
            shape,
            strides,
            tensor: Default::default(),
        }
    }

    fn update_tensor(&mut self) {
        self.tensor.dl_tensor.data = self.inner.data_ptr();
        self.tensor.dl_tensor.device = self.inner.device();
        self.tensor.dl_tensor.ndim = self.shape.ndim();
        self.tensor.dl_tensor.shape = self.shape.as_ptr();
        self.tensor.dl_tensor.strides = match self.strides {
            Some(ref strides) => strides.as_ptr(),
            None => std::ptr::null_mut(),
        };
        self.tensor.dl_tensor.dtype = self.inner.dtype();
        self.tensor.dl_tensor.byte_offset = self.inner.byte_offset();

        self.tensor.manager_ctx = self as *const Self as *mut std::ffi::c_void;
        self.tensor.deleter = Some(deleter_fn::<Self>);
    }

    pub fn into_dl_managed_tensor(self) -> NonNull<ffi::DLManagedTensor> {
        // Move self to heap and get it's pointer.
        let ctx = Box::leak(Box::new(self));
        ctx.update_tensor();
        NonNull::from(&ctx.tensor)
    }
}

impl<T> Drop for ManagerCtx<T> {
    fn drop(&mut self) {
        dbg!(self.tensor.deleter.is_some());
        if let Some(delete_fn) = self.tensor.deleter {
            unsafe { delete_fn(&mut self.tensor as *mut _) };
        }
    }
}

impl<T> ManagerCtx<T> where T: ToTensor {}

impl<T> From<T> for ManagerCtx<T>
where
    T: ToTensor,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

/// Safe wrapper for DLManagedTensor
/// Will call deleter when dropped.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct ManagedTensor(NonNull<ffi::DLManagedTensor>);

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        // TODO: we should add a flag for buggy numpy dlpack deleter
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl ManagedTensor {
    pub fn new(src: NonNull<ffi::DLManagedTensor>) -> Self {
        Self(src)
    }

    pub fn as_slice<A>(&self) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.data_ptr().cast(), self.num_elements()) }
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        self.0.as_ptr()
    }

    /// Get DLPack ptr.
    pub fn into_inner(self) -> NonNull<ffi::DLManagedTensor> {
        self.0
    }

    pub fn dl_tensor(&self) -> &ffi::DLTensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl<T> From<ManagerCtx<T>> for ManagedTensor
where
    T: ToTensor,
{
    fn from(value: ManagerCtx<T>) -> Self {
        Self(value.to_dlpack())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn from_vec_f32() {
        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = ManagerCtx::new(v);
        assert_eq!(tensor.shape(), &[10]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.device(), Device::CPU);
        assert_eq!(tensor.strides(), None);
        assert_eq!(tensor.byte_offset(), 0);
        assert_eq!(tensor.dtype(), DataType::F32);
    }
}
