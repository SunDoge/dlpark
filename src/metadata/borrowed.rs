use super::{
    Error,
    allocation::allocate_borrowed,
    raw::{checked_ndim, drop_borrowed, initialize},
};
use crate::{OpaqueContext, managed_tensor::ManagedTensorBase};
use std::ptr::NonNull;

/// Borrows fixed-rank metadata and allocates only the managed tensor.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedArray<'a, const N: usize> {
    shape: &'a [i64; N],
    strides: &'a [i64; N],
}

impl<'a, const N: usize> BorrowedArray<'a, N> {
    /// Creates fixed-rank metadata that points to caller-owned arrays.
    #[inline]
    pub fn new(shape: &'a [i64; N], strides: &'a [i64; N]) -> Self {
        Self { shape, strides }
    }

    /// Allocates a managed tensor that borrows the shape and strides arrays.
    ///
    /// # Safety
    ///
    /// The borrowed arrays must outlive the resulting managed tensor. They
    /// must not be mutated through the DLPack `shape`/`strides` pointers while
    /// the managed tensor is alive; this API starts from shared Rust
    /// references and exposes them through DLPack's mutable pointer fields.
    #[inline]
    pub unsafe fn allocate<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        const { assert!(N <= i32::MAX as usize, "N must fit in i32") };
        unsafe {
            initialize(
                allocate_borrowed(),
                self.shape.as_ptr().cast_mut(),
                self.strides.as_ptr().cast_mut(),
                N as i32,
                ctx,
                drop_borrowed::<C, M>,
            )
        }
    }
}

/// Borrows runtime-rank metadata and allocates only the managed tensor.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedSlice<'a> {
    shape: &'a [i64],
    strides: &'a [i64],
}

impl<'a> BorrowedSlice<'a> {
    /// Creates runtime-rank metadata that points to caller-owned slices.
    #[inline]
    pub fn new(shape: &'a [i64], strides: &'a [i64]) -> Self {
        Self { shape, strides }
    }

    /// Allocates a managed tensor that borrows the shape and strides slices.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the resulting managed tensor. They
    /// must not be mutated through the DLPack `shape`/`strides` pointers while
    /// the managed tensor is alive; this API starts from shared Rust
    /// references and exposes them through DLPack's mutable pointer fields.
    #[inline]
    pub unsafe fn allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        let ndim = checked_ndim(self.shape.len(), self.strides.len())?;
        Ok(unsafe {
            initialize(
                allocate_borrowed(),
                self.shape.as_ptr().cast_mut(),
                self.strides.as_ptr().cast_mut(),
                ndim,
                ctx,
                drop_borrowed::<C, M>,
            )
        })
    }

    /// Allocates borrowed metadata storage without checking runtime invariants.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the resulting managed tensor, and
    /// shape and strides must have the same length with `ndim` fitting in
    /// `i32`. They must not be mutated through the DLPack `shape`/`strides`
    /// pointers while the managed tensor is alive; this API starts from
    /// shared Rust references and exposes them through DLPack's mutable
    /// pointer fields.
    #[inline]
    pub unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        let ndim = self.shape.len();
        debug_assert_eq!(self.shape.len(), self.strides.len());
        debug_assert!(ndim <= i32::MAX as usize);

        unsafe {
            initialize(
                allocate_borrowed(),
                self.shape.as_ptr().cast_mut(),
                self.strides.as_ptr().cast_mut(),
                ndim as i32,
                ctx,
                drop_borrowed::<C, M>,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::DLManagedTensor;

    #[test]
    fn borrowed_array_allocates_header_only_and_reuses_pointers() {
        let shape = [2i64, 3];
        let strides = [3i64, 1];
        let managed = unsafe {
            BorrowedArray::new(&shape, &strides).allocate::<_, DLManagedTensor>(Box::new(()))
        };
        let tensor = unsafe { managed.as_ref().tensor() };

        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.shape, shape.as_ptr().cast_mut());
        assert_eq!(tensor.strides, strides.as_ptr().cast_mut());
        unsafe { DLManagedTensor::drop_raw(managed.as_ptr()) };
    }
}
