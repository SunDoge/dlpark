use super::{
    ffi,
    traits::{InferDtype, ToDLPack, ToTensor},
};
use crate::ffi::{DataType, Device};
use crate::manager_ctx::{CowIntArray, ManagerCtx};
use std::{ptr::NonNull, sync::Arc};

macro_rules! impl_infer_dtype {
    ($rust_type:ty, $dtype:expr) => {
        impl InferDtype for $rust_type {
            fn infer_dtype() -> DataType {
                $dtype
            }
        }
    };
}

impl_infer_dtype!(f32, DataType::F32);
impl_infer_dtype!(f64, DataType::F64);

impl_infer_dtype!(u8, DataType::U8);
impl_infer_dtype!(u16, DataType::U16);
impl_infer_dtype!(u32, DataType::U32);
impl_infer_dtype!(u64, DataType::U64);

impl_infer_dtype!(i8, DataType::I8);
impl_infer_dtype!(i16, DataType::I16);
impl_infer_dtype!(i32, DataType::I32);
impl_infer_dtype!(i64, DataType::I64);

impl_infer_dtype!(bool, DataType::BOOL);

#[cfg(feature = "half")]
impl_infer_dtype!(half::f16, DataType::F16);
#[cfg(feature = "half")]
impl_infer_dtype!(half::bf16, DataType::BF16);

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
