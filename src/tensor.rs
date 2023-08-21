pub mod impls;
pub mod traits;

use std::ptr::NonNull;

use self::traits::{FromDLPack, IntoDLPack, TensorView, ToTensor};
use crate::{ffi, manager_ctx::ManagerCtx};

/// Safe wrapper for DLManagedTensor.
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

    /// Access inner data as 1d array.
    pub fn as_slice<A>(&self) -> &[A] {
        assert_eq!(
            std::mem::size_of::<A>(),
            self.dtype().size(),
            "dtype and A size mismatch"
        );
        unsafe {
            let ptr = self.data_ptr().add(self.byte_offset() as usize);
            std::slice::from_raw_parts(ptr.cast(), self.num_elements())
        }
    }

    /// Get raw pointer.
    /// Please note that consume raw pointer multiple times may lead to double
    /// free error.
    pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor {
        self.0.as_ptr()
    }

    /// Get DLPack ptr.
    pub fn into_inner(self) -> NonNull<ffi::DLManagedTensor> {
        self.0
    }

    pub(crate) fn dl_tensor(&self) -> &ffi::DLTensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl TensorView for ManagedTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.dl_tensor().data_ptr()
    }

    fn byte_offset(&self) -> u64 {
        self.dl_tensor().byte_offset()
    }

    fn device(&self) -> ffi::Device {
        self.dl_tensor().device()
    }

    fn dtype(&self) -> ffi::DataType {
        self.dl_tensor().dtype()
    }

    fn shape(&self) -> &[i64] {
        self.dl_tensor().shape()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.dl_tensor().strides()
    }

    fn ndim(&self) -> usize {
        self.dl_tensor().ndim()
    }
}

impl<T> From<ManagerCtx<T>> for ManagedTensor
where
    T: ToTensor,
{
    fn from(value: ManagerCtx<T>) -> Self {
        Self(value.into_dlpack())
    }
}

impl FromDLPack for ManagedTensor {
    fn from_dlpack(src: NonNull<ffi::DLManagedTensor>) -> Self {
        Self(src)
    }
}

impl IntoDLPack for ManagedTensor {
    fn into_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{prelude::*, utils::make_contiguous_strides};

    #[test]
    fn from_vec_f32() {
        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = ManagedTensor::from_dlpack(v.into_dlpack());
        assert_eq!(tensor.shape(), &[10]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.device(), Device::CPU);
        assert_eq!(tensor.byte_offset(), 0);
        assert_eq!(tensor.dtype(), DataType::F32);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn from_arc_slice_f32() {
        let v: Arc<[f32]> = (0..10).map(|x| x as f32).collect::<Vec<_>>().into();
        let t1 = ManagedTensor::from_dlpack(v.clone().into_dlpack());
        let t2 = ManagedTensor::from_dlpack(v.clone().into_dlpack());
        assert_eq!(t1.data_ptr(), t2.data_ptr());
        assert_eq!(v.data_ptr(), t1.data_ptr());
    }

    #[test]
    fn contiguous_strides() {
        let shape = [1, 2, 3];
        let strides = make_contiguous_strides(&shape);
        assert_eq!(&strides, &[6, 3, 1]);
    }

    #[test]
    fn test_as_slice() {
        let v: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let tensor = ManagedTensor::from_dlpack(v.clone().into_dlpack());
        assert_eq!(tensor.as_slice::<f32>(), &v[..]);
    }
}
