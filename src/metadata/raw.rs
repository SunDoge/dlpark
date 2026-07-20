use super::{
    Error, MismatchedLengthSnafu, NdimOverflowSnafu,
    allocation::{copied_array_layout, copied_slice_layout},
};
use crate::{OpaqueContext, ffi::DLTensor, managed_tensor::ManagedTensorBase};
use snafu::{ResultExt, ensure};
use std::{alloc::Layout, ptr::NonNull};

#[inline]
pub(super) unsafe fn copy_i64_metadata(src: &[i64], dst: *mut i64) {
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

#[inline]
pub(super) unsafe fn copy_i64_metadata_n(src: &[i64], dst: *mut i64, len: usize) {
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len) };
}

#[inline]
pub(super) unsafe fn try_copy_generic_metadata<T: Copy + TryInto<i64>>(
    src: &[T],
    dst: *mut i64,
) -> Result<(), usize> {
    for (axis, &value) in src.iter().enumerate() {
        let value = value.try_into().map_err(|_| axis)?;
        unsafe { dst.add(axis).write(value) };
    }
    Ok(())
}

#[inline]
pub(super) unsafe fn copy_generic_metadata_unchecked<T: Copy + TryInto<i64>>(
    src: &[T],
    dst: *mut i64,
) {
    for (index, &value) in src.iter().enumerate() {
        let value = match value.try_into() {
            Ok(value) => value,
            Err(_) => unsafe { std::hint::unreachable_unchecked() },
        };
        unsafe { dst.add(index).write(value) };
    }
}

#[inline]
pub(super) unsafe fn initialize<C, M>(
    managed_tensor: NonNull<M>,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
    ctx: C,
    deleter: unsafe extern "C" fn(*mut M),
) -> NonNull<M>
where
    C: OpaqueContext,
    M: ManagedTensorBase,
{
    unsafe {
        managed_tensor.as_ptr().write(M::from_parts(
            DLTensor::from_parts(shape, strides, ndim),
            ctx.into_raw(),
            Some(deleter),
        ));
    }
    managed_tensor
}

#[inline]
pub(super) fn checked_ndim(shape_len: usize, strides_len: usize) -> Result<i32, Error> {
    ensure!(
        shape_len == strides_len,
        MismatchedLengthSnafu {
            shape_len,
            strides_len
        }
    );
    shape_len
        .try_into()
        .context(NdimOverflowSnafu { ndim: shape_len })
}

pub(super) unsafe extern "C" fn drop_copied_array<
    C: OpaqueContext,
    M: ManagedTensorBase,
    const N: usize,
>(
    managed: *mut M,
) {
    if managed.is_null() {
        return;
    }
    unsafe {
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), copied_array_layout::<M, N>().0);
    }
}

pub(super) unsafe extern "C" fn drop_copied_slice<C: OpaqueContext, M: ManagedTensorBase>(
    managed: *mut M,
) {
    if managed.is_null() {
        return;
    }
    unsafe {
        let allocation_ndim_offset = copied_slice_layout::<M>(0).1;
        let allocation_ndim = managed
            .cast::<u8>()
            .add(allocation_ndim_offset)
            .cast::<usize>()
            .read();
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), copied_slice_layout::<M>(allocation_ndim).0);
    }
}

pub(super) unsafe extern "C" fn drop_borrowed<C: OpaqueContext, M: ManagedTensorBase>(
    managed: *mut M,
) {
    if managed.is_null() {
        return;
    }
    unsafe {
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), Layout::new::<M>());
    }
}
