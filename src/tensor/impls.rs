use super::{
    ffi,
    traits::{
        FromDLPack, HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides, HasTensor,
        InferDtype, ToDLPack,
    },
    ManagedTensor, ManagerCtx, Shape,
};
use crate::{
    ffi::{DataType, Device},
    prelude::AsTensor,
};
use std::{ffi::c_void, ptr::NonNull};

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

impl<T> HasDtype for Vec<T>
where
    T: InferDtype,
{
    fn dtype(&self) -> DataType {
        T::infer_dtype()
    }
}
impl<T> HasStrides for Vec<T> {}
impl<T> HasByteOffset for Vec<T> {
    fn byte_offset(&self) -> u64 {
        0
    }
}

impl InferDtype for f32 {
    fn infer_dtype() -> DataType {
        DataType::F32
    }
}

impl InferDtype for f64 {
    fn infer_dtype() -> DataType {
        DataType::F64
    }
}

impl InferDtype for i64 {
    fn infer_dtype() -> DataType {
        DataType::I64
    }
}

impl InferDtype for i32 {
    fn infer_dtype() -> DataType {
        DataType::I32
    }
}

impl InferDtype for u8 {
    fn infer_dtype() -> DataType {
        DataType::U8
    }
}

impl InferDtype for bool {
    fn infer_dtype() -> DataType {
        DataType::BOOL
    }
}

impl AsTensor for ffi::DLTensor {
    fn data(&self) -> *mut std::ffi::c_void {
        self.data
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

impl<T> AsTensor for T
where
    T: HasTensor<ffi::DLTensor>,
{
    fn data(&self) -> *mut std::ffi::c_void {
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

impl<T> AsTensor for ManagerCtx<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn data(&self) -> *mut std::ffi::c_void {
        self.inner.data()
    }

    fn device(&self) -> Device {
        self.inner.device()
    }

    fn byte_offset(&self) -> u64 {
        self.inner.byte_offset()
    }

    fn shape(&self) -> &[i64] {
        self.shape.as_slice()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.strides.as_ref().map(|s| s.as_slice(self.ndim()))
    }

    fn ndim(&self) -> usize {
        self.shape.ndim() as usize
    }

    fn dtype(&self) -> DataType {
        self.inner.dtype()
    }
}

// It's hard and unsafe to recover T from dlpack ptr.
// ManagerCtx should only be a DLManagedTensor builder.
impl<T> ToDLPack for ManagerCtx<T>
where
    T: HasData + HasDevice + HasDtype + HasByteOffset,
{
    fn to_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        self.into_dl_managed_tensor()
    }
}

impl FromDLPack for ManagedTensor {
    fn from_dlpack(src: NonNull<ffi::DLManagedTensor>) -> Self {
        Self(src)
    }
}

impl<T> ToDLPack for T
where
    T: HasData + HasDevice + HasShape + HasStrides + HasDtype + HasByteOffset,
{
    fn to_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        let ctx = ManagerCtx::new(self);
        ctx.into_dl_managed_tensor()
    }
}
