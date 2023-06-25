use super::{
    ffi,
    traits::{InferDtype, ToDLPack, ToTensor},
};
use crate::ffi::{DataType, Device};
use crate::manager_ctx::{CowIntArray, ManagerCtx};
use std::{ptr::NonNull, sync::Arc};

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

impl<T> ToTensor for Vec<T>
where
    T: InferDtype,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut T as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        T::infer_dtype()
    }

    fn shape(&self) -> CowIntArray {
        CowIntArray::from_owned(vec![self.len() as i64])
    }

    fn strides(&self) -> Option<CowIntArray> {
        None
    }
}

impl<T> ToTensor for Box<[T]>
where
    T: InferDtype,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut T as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        T::infer_dtype()
    }

    fn shape(&self) -> CowIntArray {
        CowIntArray::from_owned(vec![self.len() as i64])
    }

    fn strides(&self) -> Option<CowIntArray> {
        None
    }
}

impl<T> ToTensor for Arc<[T]>
where
    T: InferDtype,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut T as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        T::infer_dtype()
    }

    fn shape(&self) -> CowIntArray {
        CowIntArray::from_owned(vec![self.len() as i64])
    }

    fn strides(&self) -> Option<CowIntArray> {
        None
    }
}

impl<T> ToDLPack for T
where
    T: ToTensor,
{
    fn to_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        let ctx = ManagerCtx::new(self);
        ctx.into_dl_managed_tensor()
    }
}
