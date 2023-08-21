use std::ptr::NonNull;

use crate::tensor::traits::{IntoDLPack, TensorView};
use crate::ShapeAndStrides;
use crate::{ffi, prelude::ToTensor};

unsafe extern "C" fn deleter_fn<T>(dl_managed_tensor: *mut ffi::DLManagedTensor) {
    // Reconstruct pointer and destroy it.
    let ctx = (*dl_managed_tensor).manager_ctx as *mut T;
    // https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_raw
    // Use from_raw to clean it.
    unsafe { Box::from_raw(ctx) };
}

// TODO: should be ManagerCtx<T, M> where M is one of DLManagedTensor and DLManagedTensorVersioned
/// The ManagerCtx holds the Tensor and its metadata.
pub struct ManagerCtx<T> {
    inner: T,
    shape_and_strides: ShapeAndStrides,
    // The ctx should hold DLManagedTensor, so that the tensor can be freed.
    tensor: Option<ffi::DLManagedTensor>,
}

impl<T> ManagerCtx<T>
where
    T: ToTensor,
{
    pub fn new(inner: T) -> Self {
        let shape_and_strides = inner.shape_and_strides();
        Self {
            inner,
            shape_and_strides,
            tensor: None,
        }
    }

    pub(crate) fn into_dl_managed_tensor(self) -> NonNull<ffi::DLManagedTensor> {
        // Move self to heap and get it's pointer.
        // We leak the data here and let deleter handle its memmory.
        let ctx = Box::leak(Box::new(self));
        let tensor: ffi::DLManagedTensor = ffi::DLManagedTensor {
            dl_tensor: ctx.make_dl_tensor(),
            manager_ctx: ctx as *mut Self as *mut std::ffi::c_void,
            deleter: Some(deleter_fn::<Self>),
        };
        // Hold the data so it can be dropped when ctx dropped.
        ctx.tensor = Some(tensor);
        // Take the address of DLManagedTensor
        NonNull::from(ctx.tensor.as_ref().unwrap())
    }

    fn make_dl_tensor(&self) -> ffi::DLTensor {
        ffi::DLTensor {
            data: self.inner.data_ptr(),
            device: self.inner.device(),
            ndim: self.shape_and_strides.ndim(),
            dtype: self.inner.dtype(),
            shape: self.shape_and_strides.shape_ptr(),
            strides: self.shape_and_strides.strides_ptr(),
            byte_offset: self.inner.byte_offset(),
        }
    }
}

impl<T> ManagerCtx<T> where T: ToTensor {}

impl<T> From<T> for ManagerCtx<T>
where
    T: ToTensor,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> TensorView for ManagerCtx<T>
where
    T: ToTensor,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.inner.data_ptr()
    }

    fn device(&self) -> ffi::Device {
        self.inner.device()
    }

    fn byte_offset(&self) -> u64 {
        self.inner.byte_offset()
    }

    fn shape(&self) -> &[i64] {
        self.shape_and_strides.shape()
    }

    fn strides(&self) -> Option<&[i64]> {
        self.shape_and_strides.strides()
    }

    fn ndim(&self) -> usize {
        self.shape_and_strides.len()
    }

    fn dtype(&self) -> ffi::DataType {
        self.inner.dtype()
    }
}

// It's hard and unsafe to recover T from dlpack ptr.
// ManagerCtx should only be a DLManagedTensor builder.
impl<T> IntoDLPack for ManagerCtx<T>
where
    T: ToTensor,
{
    fn into_dlpack(self) -> NonNull<ffi::DLManagedTensor> {
        self.into_dl_managed_tensor()
    }
}
