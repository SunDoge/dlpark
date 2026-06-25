use std::ptr::NonNull;

use crate::ffi::{self, DataType, Device, Dlpack};
use crate::traits::{TensorLike, manager_context::{ManagerContext, auto_deleter_legacy}};

pub struct TensorLikeContext<T> {
    inner: T,
    shape: Box<[i64]>,
    strides: Option<Box<[i64]>>,
    dtype: DataType,
    device: Device,
    byte_offset: u64,
    pub(crate) managed_tensor: ffi::ManagedTensor,
}

impl<T: TensorLike> TensorLikeContext<T> {
    pub fn new(tensor: T) -> Result<Box<Self>, T::Error> {
        let shape = tensor.shape().into_boxed_slice();
        let strides = tensor.strides().map(|s| s.into_boxed_slice());
        let dtype = tensor.data_type()?;
        let device = tensor.device()?;
        let byte_offset = tensor.byte_offset();
        Ok(Box::new(Self { inner: tensor, shape, strides, dtype, device, byte_offset, managed_tensor: ffi::ManagedTensor::default() }))
    }
}

unsafe impl<T: TensorLike> ManagerContext for TensorLikeContext<T> {
    fn data_ptr(&self) -> *mut std::ffi::c_void { self.inner.data_ptr() }
    fn ndim(&self) -> i32 { self.shape.len() as i32 }
    fn shape_ptr(&self) -> NonNull<i64> { if self.shape.is_empty() { NonNull::dangling() } else { unsafe { NonNull::new_unchecked(self.shape.as_ptr() as *mut i64) } } }
    fn strides_ptr(&self) -> Option<NonNull<i64>> { self.strides.as_ref().map(|s| unsafe { NonNull::new_unchecked(s.as_ptr() as *mut i64) }) }
    fn dtype(&self) -> DataType { self.dtype }
    fn device(&self) -> Device { self.device }
    fn byte_offset(&self) -> u64 { self.byte_offset }
}

pub(crate) fn into_dlpack<T: TensorLike>(mut ctx: Box<TensorLikeContext<T>>) -> Dlpack {
    let data=ctx.data_ptr(); let device=ctx.device(); let dtype=ctx.dtype(); let offset=ctx.byte_offset(); let ndim=ctx.ndim(); let sp=ctx.shape_ptr().as_ptr(); let stp=ctx.strides_ptr().map(|p|p.as_ptr()).unwrap_or(std::ptr::null_mut());
    { let mt=&mut ctx.managed_tensor; mt.dl_tensor.data=data; mt.dl_tensor.device=device; mt.dl_tensor.dtype=dtype; mt.dl_tensor.byte_offset=offset; mt.dl_tensor.ndim=ndim; mt.dl_tensor.shape=sp; mt.dl_tensor.strides=stp; mt.deleter=Some(auto_deleter_legacy::<TensorLikeContext<T>>); }
    let ctx_ptr=Box::into_raw(ctx);
    unsafe { let mt_ptr=&mut (*ctx_ptr).managed_tensor as *mut ffi::ManagedTensor; (*mt_ptr).manager_ctx=ctx_ptr as *mut _; NonNull::new_unchecked(mt_ptr) }
}