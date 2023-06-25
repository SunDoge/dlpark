use super::{
    ffi,
    traits::{FromDLPack, InferDtype, TensorView, ToDLPack, ToTensor},
    ManagedTensor,
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

impl TensorView for ffi::DLTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
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

impl TensorView for ManagedTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.dl_tensor().data_ptr()
    }

    fn byte_offset(&self) -> u64 {
        self.dl_tensor().byte_offset()
    }

    fn device(&self) -> Device {
        self.dl_tensor().device()
    }

    fn dtype(&self) -> DataType {
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

// It's hard and unsafe to recover T from dlpack ptr.
// ManagerCtx should only be a DLManagedTensor builder.
impl<T> ToDLPack for ManagerCtx<T>
where
    T: ToTensor,
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
    T: ToTensor,
{
    fn to_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        let ctx = ManagerCtx::new(self);
        ctx.into_dl_managed_tensor()
    }
}
