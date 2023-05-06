use pin_project::{pin_project, pinned_drop};
use std::{
    ffi::c_void,
    marker::{PhantomData, PhantomPinned},
    mem::transmute,
    pin::Pin,
    ptr::{self, NonNull},
    slice,
};

use crate::dlpack::{DLManagedTensor, DLTensor, DataType, DataTypeCode, Device, DeviceType};

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

    // drop(ctx);
    drop(unsafe { Box::from_raw(ctx) });
    // println!("drop ctx: {:?}", ctx);
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
