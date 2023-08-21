use super::{
    ffi,
    traits::{InferDtype, IntoDLPack, ToTensor},
};
use crate::ffi::{DataType, Device};
use crate::manager_ctx::ManagerCtx;
use crate::ShapeAndStrides;
use std::{ptr::NonNull, sync::Arc};

macro_rules! impl_for_rust_type {
    ($rust_type:ty, $dtype:expr) => {
        impl InferDtype for $rust_type {
            fn infer_dtype() -> DataType {
                $dtype
            }
        }

        impl ToTensor for $rust_type {
            fn data_ptr(&self) -> *mut std::ffi::c_void {
                self as *const Self as *mut std::ffi::c_void
            }

            fn byte_offset(&self) -> u64 {
                0
            }

            fn device(&self) -> Device {
                Device::CPU
            }

            fn dtype(&self) -> DataType {
                $dtype
            }

            fn shape_and_strides(&self) -> ShapeAndStrides {
                ShapeAndStrides::new_contiguous(&[])
            }
        }
    };
}

impl_for_rust_type!(f32, DataType::F32);
impl_for_rust_type!(f64, DataType::F64);

impl_for_rust_type!(u8, DataType::U8);
impl_for_rust_type!(u16, DataType::U16);
impl_for_rust_type!(u32, DataType::U32);
impl_for_rust_type!(u64, DataType::U64);
impl_for_rust_type!(u128, DataType::U128);

impl_for_rust_type!(i8, DataType::I8);
impl_for_rust_type!(i16, DataType::I16);
impl_for_rust_type!(i32, DataType::I32);
impl_for_rust_type!(i64, DataType::I64);
impl_for_rust_type!(i128, DataType::I128);

impl_for_rust_type!(bool, DataType::BOOL);

#[cfg(feature = "half")]
impl_for_rust_type!(half::f16, DataType::F16);
#[cfg(feature = "half")]
impl_for_rust_type!(half::bf16, DataType::BF16);

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

    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_with_strides(&[self.len() as i64], &[1])
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

    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_with_strides(&[self.len() as i64], &[1])
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

    fn shape_and_strides(&self) -> ShapeAndStrides {
        ShapeAndStrides::new_with_strides(&[self.len() as i64], &[1])
    }
}

impl<T> IntoDLPack for T
where
    T: ToTensor,
{
    fn into_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        let ctx = ManagerCtx::new(self);
        ctx.into_dl_managed_tensor()
    }
}
