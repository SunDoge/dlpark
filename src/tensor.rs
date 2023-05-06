use std::{ffi::c_void, marker::PhantomData, slice};

use crate::dlpack::{DLManagedTensor, DLTensor, DataType, Device};

#[derive(Debug)]
pub struct Tensor<'a> {
    pub inner: DLTensor,
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

impl<'a> From<DLTensor> for Tensor<'a> {
    fn from(value: DLTensor) -> Self {
        Self {
            inner: value,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<Tensor<'a>> for DLTensor {
    fn from(value: Tensor<'a>) -> Self {
        value.inner
    }
}

impl<'a> Tensor<'a> {
    pub fn data(&self) -> *mut c_void {
        self.inner.data
    }

    pub fn shape(&self) -> &[i64] {
        unsafe { slice::from_raw_parts(self.inner.shape, self.ndim()) }
    }

    pub fn strides(&self) -> &[i64] {
        unsafe { slice::from_raw_parts(self.inner.strides, self.ndim()) }
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim as usize
    }

    pub fn byte_offset(&self) -> u64 {
        self.inner.byte_offset
    }

    pub fn into_inner(self) -> DLTensor {
        self.inner
    }

    pub fn into_ptr(self) -> *const DLTensor {
        &self.inner as *const _
    }

    pub unsafe fn from_raw(ptr: *mut DLTensor) -> Self {
        debug_assert!(!ptr.is_null());
        Self {
            inner: *ptr,
            _marker: PhantomData,
        }
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }
}

pub struct TensorBuilder {
    data: *mut c_void,
    shape: *mut i64,
    strides: *mut i64,
    dtype: DataType,
    device: Device,
    byte_offset: u64,
    ndim: i32,
}

impl TensorBuilder {
    pub fn new(data: *mut c_void, shape: *mut i64) -> Self {
        Self {
            data,
            shape,
            strides: std::ptr::null_mut(),
            dtype: DataType::default(),
            device: Device::default(),
            byte_offset: 0,
            ndim: 1,
        }
    }

    pub fn strides(&mut self, strides: *mut i64) -> &mut Self {
        self.strides = strides;
        self
    }

    pub fn build(self) -> Tensor<'static> {
        let inner = DLTensor {
            data: self.data,
            device: self.device,
            ndim: self.ndim,
            dtype: self.dtype,
            shape: self.shape,
            strides: self.strides,
            byte_offset: self.byte_offset,
        };

        Tensor {
            inner,
            _marker: PhantomData,
        }
    }
}

pub struct ManagedTensor<C> {
    pub inner: DLManagedTensor,
    _marker: PhantomData<C>,
}

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut DLManagedTensor) {
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    // dbg!(ctx);
    drop(unsafe { Box::from_raw(ctx) });
}

impl<C> ManagedTensor<C> {
    pub fn new(ctx: C, dl_tensor: DLTensor) -> Self {
        let dl_managed_tensor = DLManagedTensor {
            dl_tensor,
            manager_ctx: Box::into_raw(Box::new(ctx)) as *mut _,
            deleter: Some(deleter_fn::<C>),
        };
        Self {
            inner: dl_managed_tensor,
            _marker: PhantomData,
        }
    }

    pub fn into_inner(self) -> DLManagedTensor {
        self.inner
    }
}

pub trait AsDLTensor {
    fn as_dl_tensor(&self) -> DLTensor;
}

pub enum Shape {
    Borrowed(*mut i64, usize),
    Owned(Vec<i64>),
}

// impl From<&IntArrayRef> for *mut i64 {
//     fn from(value: &IntArrayRef) -> Self {
//         match value {
//             IntArrayRef::Borrowed(ptr) => ptr,
//             IntArrayRef::Owned(ref v) => v.as_ptr() as *mut i64,
//         }
//     }
// }

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
        let mut v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let mut shape = vec![v.len() as i64];
        let tensor = TensorBuilder::new(v.as_mut_ptr() as *mut _, shape.as_mut_ptr()).build();
        dbg!(&tensor);
        let v2: &[f32] =
            unsafe { slice::from_raw_parts(tensor.data() as *const _, tensor.shape()[0] as usize) };
        dbg!(v2);
    }

    #[test]
    fn from_vec_f32_managed() {
        struct V(Vec<f32>);

        impl Drop for V {
            fn drop(&mut self) {
                println!("drop v");
                drop(self);
            }
        }

        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let mut v = V(v);
        let mut shape = vec![v.0.len() as i64];
        let tensor = TensorBuilder::new(v.0.as_mut_ptr() as *mut _, shape.as_mut_ptr()).build();
        dbg!(&tensor);
        let managed_tensor = ManagedTensor::new(v, tensor.into());
        managed_tensor.inner.deleter.map(|del_fn| unsafe {
            del_fn(&managed_tensor.inner as *const _ as *mut _);
        });
    }
}
