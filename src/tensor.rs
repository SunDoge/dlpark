pub mod impls;
pub mod traits;

use std::ptr::NonNull;

use traits::{HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides};

use crate::ffi::{self, DataType, Device};

use self::traits::{AsTensor, HasTensor};

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
}

pub struct ManagerCtx<T> {
    inner: T,
    shape: Shape,
    strides: Option<Strides>,
    // The ctx should hold DLManagedTensor, so that the tensor can be freed.
    tensor: Option<ffi::DLManagedTensor>,
}

impl<T> ManagerCtx<T> {
    pub fn set_tensor(&mut self, tensor: ffi::DLManagedTensor) {
        self.tensor = Some(tensor);
    }

    pub fn get_tensor_ptr(&self) -> NonNull<ffi::DLManagedTensor> {
        NonNull::from(self.tensor.as_ref().unwrap())
    }
}

impl<T> ManagerCtx<T>
where
    T: HasShape + HasStrides,
{
    pub fn new(inner: T) -> Self {
        let shape: Shape = inner.shape();
        let strides = inner.strides();

        Self {
            inner,
            shape,
            strides,
            tensor: None,
        }
    }

    pub fn new_boxed(inner: T) -> Box<Self> {
        Box::new(Self::new(inner))
    }
}

impl<T> ManagerCtx<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    pub fn to_dl_tensor(&self) -> ffi::DLTensor {
        ffi::DLTensor {
            data: self.inner.data(),
            device: self.inner.device(),
            ndim: self.shape.ndim(),
            shape: self.shape.as_ptr(),
            dtype: self.inner.dtype(),
            strides: match self.strides {
                Some(ref strides) => strides.as_ptr(),
                None => std::ptr::null_mut(),
            },
            byte_offset: self.inner.byte_offset(),
        }
    }

    pub fn into_dl_managed_tensor(self) -> NonNull<ffi::DLManagedTensor> {
        // Move self to heap and get it's pointer.
        let ctx_ref = Box::leak(Box::new(self));
        let dl_tensor = ctx_ref.to_dl_tensor();
        let tensor = ffi::DLManagedTensor {
            dl_tensor,
            manager_ctx: ctx_ref as *const Self as *mut std::ffi::c_void,
            deleter: Some(deleter_fn::<ManagerCtx<T>>),
        };
        // Make a self-reference struct so DLManagedTensor can be correctly dropped.
        ctx_ref.set_tensor(tensor);

        // Return the DLManagedTensor's pointer.
        ctx_ref.get_tensor_ptr()
    }
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
            tensor: None,
        }
    }
}

impl<T> From<&mut ManagerCtx<T>> for ffi::DLTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: &mut ManagerCtx<T>) -> Self {
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

impl<T> From<ManagerCtx<T>> for NonNull<ffi::DLManagedTensor>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: ManagerCtx<T>) -> Self {
        value.into_dl_managed_tensor()
    }
}

impl AsTensor for ffi::DLTensor {
    fn data<T>(&self) -> *const T {
        self.data as *const T
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

/// Safe wrapper for DLManagedTensor
/// Will call deleter when dropped.
#[repr(transparent)]
pub struct ManagedTensor(NonNull<ffi::DLManagedTensor>);

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl ManagedTensor {
    pub fn new(ptr: NonNull<ffi::DLManagedTensor>) -> Self {
        Self(ptr)
    }

    pub fn as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data(), self.num_elements()) }
    }

    pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        self.0.as_ptr()
    }

    pub fn into_inner(self) -> NonNull<ffi::DLManagedTensor> {
        self.0
    }
}

impl<T> AsTensor for T
where
    T: HasTensor<ffi::DLTensor>,
{
    fn data<D>(&self) -> *const D {
        self.tensor().data()
    }

    fn device(&self) -> Device {
        self.tensor().device()
    }

    fn dtype(&self) -> DataType {
        self.tensor().dtype()
    }

    fn ndim(&self) -> usize {
        self.tensor().ndim()
    }

    fn byte_offset(&self) -> u64 {
        self.tensor().byte_offset()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.tensor().strides()
    }

    fn shape(&self) -> &[i64] {
        self.tensor().shape()
    }
}

impl HasTensor<ffi::DLTensor> for ManagedTensor {
    fn tensor(&self) -> &ffi::DLTensor {
        // unsafe { &(*self.inner).dl_tensor }
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl HasTensor<ffi::DLTensor> for ffi::DLManagedTensor {
    fn tensor(&self) -> &ffi::DLTensor {
        &self.dl_tensor
    }
}

impl<T> From<ManagerCtx<T>> for ManagedTensor
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn from(value: ManagerCtx<T>) -> Self {
        Self(value.into_dl_managed_tensor())
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
