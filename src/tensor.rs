use std::{ffi::c_void, marker::PhantomData, slice};

use crate::dlpack::{DLManagedTensor, DLTensor, Device};

#[derive(Debug)]
pub struct TensorRef<'a> {
    pub inner: DLTensor,
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

impl<'a> From<DLTensor> for TensorRef<'a> {
    fn from(value: DLTensor) -> Self {
        Self {
            inner: value,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<TensorRef<'a>> for DLTensor {
    fn from(value: TensorRef<'a>) -> Self {
        value.inner
    }
}

impl<'a> TensorRef<'a> {
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

// pub struct TensorMaker {

//     shape: Vec<i64>,
//     strides: Option<Vec<i64>>,
//     dtype: DataType,
// }

// impl TensorMaker {
//     pub fn new(data: *mut c_void, shape: &[i64]) -> Self {
//         Self {
//             shape: shape.to_vec(),
//             strides: None,
//             dtype: DataType::F32,
//         }
//     }

//     pub fn shape(&mut self) -> &mut Self {}
// }

pub struct ManagedTensor {
    pub inner: DLManagedTensor,
}

impl ManagedTensor {}

pub struct Managed<T> {
    pub inner: T,
    pub dl_tensor: DLTensor,
}

// pub struct TensorRefMut<'a> {
//     pub inner: DLTensor,
//     _marker: PhantomData<&'a mut ()>,
// }

// impl<'a> From<DLTensor> for TensorRefMut<'a> {
//     fn from(value: DLTensor) -> Self {
//         Self {
//             inner: value,
//             _marker: PhantomData,
//         }
//     }
// }

// impl<'a> From<TensorRefMut<'a>> for DLTensor {
//     fn from(value: TensorRefMut<'a>) -> Self {
//         value.inner
//     }
// }
