use std::borrow::Cow;
use std::ptr::NonNull;

use crate::tensor::traits::TensorView;
use crate::{ffi, prelude::ToTensor};

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut ffi::DLManagedTensor) {
    // Reconstruct pointer and destroy it.
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    drop(unsafe { Box::from_raw(ctx) });
    // ctx.drop_in_place();
}

// #[derive(Debug)]
// pub enum Shape {
//     Borrowed(*mut i64, usize),
//     Owned(Vec<i64>),
// }

#[derive(Debug)]
pub struct CowIntArray(Cow<'static, [i64]>);

impl CowIntArray {
    pub fn from_owned(v: Vec<i64>) -> Self {
        Self(Cow::Owned(v))
    }

    pub fn from_borrowed(v: &'static [i64]) -> Self {
        Self(Cow::Borrowed(v))
    }

    pub fn as_ptr(&self) -> *mut i64 {
        match self.0 {
            Cow::Borrowed(v) => v.as_ptr() as *mut i64,
            Cow::Owned(ref v) => v.as_ptr() as *mut i64,
        }
    }

    fn len(&self) -> usize {
        match self.0 {
            Cow::Borrowed(v) => v.len(),
            Cow::Owned(ref v) => v.len(),
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
// pub enum Strides {
//     Borrowed(*mut i64),
//     Owned(Vec<i64>),
// }

pub struct ManagerCtx<T> {
    inner: T,
    shape: CowIntArray,
    strides: Option<CowIntArray>,
    // The ctx should hold DLManagedTensor, so that the tensor can be freed.
    tensor: ffi::DLManagedTensor,
}

impl<T> ManagerCtx<T>
where
    T: ToTensor,
{
    pub fn new(inner: T) -> Self {
        let shape: CowIntArray = inner.shape();
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

    pub fn as_ref(&self) -> &T {
        &self.inner
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

impl<T> TensorView for ManagerCtx<T>
where
    T: ToTensor,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.data_ptr()
    }

    fn device(&self) -> ffi::Device {
        self.inner.device()
    }

    fn byte_offset(&self) -> u64 {
        self.inner.byte_offset()
    }

    fn shape(&self) -> &[i64] {
        self.shape.as_slice()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.strides.as_ref().map(|s| s.as_slice())
    }

    fn ndim(&self) -> usize {
        self.shape.ndim() as usize
    }

    fn dtype(&self) -> ffi::DataType {
        self.inner.dtype()
    }
}
