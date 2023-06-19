use super::{
    ffi,
    traits::{
        FromDLPack, HasByteOffset, HasData, HasDevice, HasDtype, HasShape, HasStrides, InferDtype,
        ToDLPack,
    },
    ManagedTensor, ManagerCtx, Shape,
};
use crate::ffi::{DataType, Device};
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

impl ToDLPack for ManagedTensor {
    fn to_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        self.0
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
