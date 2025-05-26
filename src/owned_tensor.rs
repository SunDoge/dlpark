use crate::{
    data_type::DataType,
    device::Device,
    managed_tensor::{FromDlpack, IntoDlpack, ManagedTensor},
};

use std::ptr::NonNull;

pub struct OwnedTensor(NonNull<ManagedTensor>);

impl Drop for OwnedTensor {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl OwnedTensor {
    pub fn new(managed_tensor: NonNull<ManagedTensor>) -> Self {
        Self(managed_tensor)
    }

    pub fn shape(&self) -> &[i64] {
        unsafe { self.0.as_ref().dl_tensor.get_shape() }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        unsafe { self.0.as_ref().dl_tensor.get_strides() }
    }

    pub fn num_dimensions(&self) -> usize {
        unsafe { self.0.as_ref().dl_tensor.num_dimensions() }
    }

    pub fn data_type(&self) -> &DataType {
        unsafe { &self.0.as_ref().dl_tensor.dtype }
    }

    pub fn device(&self) -> &Device {
        unsafe { &self.0.as_ref().dl_tensor.device }
    }

    pub fn num_elements(&self) -> usize {
        unsafe { self.0.as_ref().dl_tensor.num_elements() }
    }

    pub fn as_slice_untyped(&self) -> &[u8] {
        unsafe { self.0.as_ref().dl_tensor.as_slice_untyped() }
    }

    pub fn as_slice<A>(&self) -> &[A] {
        unsafe { self.0.as_ref().dl_tensor.as_slice::<A>() }
    }

    pub fn byte_offset(&self) -> usize {
        unsafe { self.0.as_ref().dl_tensor.byte_offset as usize }
    }

    pub fn as_ptr<A>(&self) -> *const A {
        unsafe { self.0.as_ref().dl_tensor.data.cast::<A>() }
    }

    pub fn as_ptr_untyped(&self) -> *const u8 {
        unsafe { self.0.as_ref().dl_tensor.data.cast::<u8>() }
    }
}

impl FromDlpack for OwnedTensor {
    fn from_dlpack(managed_tensor: NonNull<ManagedTensor>) -> Self {
        Self::new(managed_tensor)
    }
}

impl IntoDlpack for OwnedTensor {
    fn into_dlpack(self) -> NonNull<ManagedTensor> {
        self.0
    }
}
