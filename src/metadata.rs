use crate::{OpaqueContext, ffi::DLTensor, managed_tensor::ManagedTensorBase};
use snafu::{ResultExt, Snafu, ensure};
use std::{alloc::Layout, borrow::Borrow, convert::Infallible, ptr::NonNull};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Mismatched length of shape ({shape_len}) and strides ({strides_len})"))]
    MismatchedLength {
        shape_len: usize,
        strides_len: usize,
    },

    #[snafu(display("Dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        ndim: usize,
        source: std::num::TryFromIntError,
    },
}

pub trait Metadata {
    type Error;

    fn try_allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Self::Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;

    /// Allocates metadata storage without checking runtime invariants.
    ///
    /// # Safety
    ///
    /// The caller must uphold the metadata type's runtime invariants. For
    /// dynamic metadata this includes matching shape/strides lengths,
    /// `ndim <= i32::MAX`, and readable shape/strides storage for `ndim`
    /// `i64` values. Violating these requirements may cause out-of-bounds
    /// reads or an invalid `DLTensor`.
    unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;
}

pub trait InfallibleMetadata: Metadata<Error = Infallible> {
    fn allocate<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;
}

#[inline]
fn copied_array_layout<M, const N: usize>() -> (Layout, usize) {
    let metadata = Layout::array::<i64>(
        N.checked_mul(2)
            .expect("DLPack metadata length should fit in usize"),
    )
    .expect("DLPack metadata allocation layout should fit in memory");

    extend_metadata_layout::<M>(metadata)
}

#[inline]
fn copied_slice_layout<M>(ndim: usize) -> (Layout, usize, usize) {
    let allocation_ndim = Layout::new::<usize>();
    let metadata = Layout::array::<i64>(
        ndim.checked_mul(2)
            .expect("DLPack metadata length should fit in usize"),
    )
    .expect("DLPack metadata allocation layout should fit in memory");

    let (layout, allocation_ndim_offset) = Layout::new::<M>()
        .extend(allocation_ndim)
        .expect("DLPack storage allocation layout should fit in memory");
    let (layout, metadata_offset) = layout
        .extend(metadata)
        .expect("DLPack storage allocation layout should fit in memory");

    (
        layout.pad_to_align(),
        allocation_ndim_offset,
        metadata_offset,
    )
}

#[inline]
fn extend_metadata_layout<M>(metadata: Layout) -> (Layout, usize) {
    Layout::new::<M>()
        .extend(metadata)
        .map(|(layout, offset)| (layout.pad_to_align(), offset))
        .expect("DLPack storage allocation layout should fit in memory")
}

#[inline]
unsafe fn allocate<M>(layout: Layout) -> NonNull<M> {
    let ptr = unsafe { std::alloc::alloc(layout) }.cast::<M>();
    match NonNull::new(ptr) {
        Some(ptr) => ptr,
        None => std::alloc::handle_alloc_error(layout),
    }
}

impl<S, T, const N: usize> Metadata for CopiedArray<S, T, N>
where
    S: Borrow<[i64; N]>,
    T: Borrow<[i64; N]>,
{
    type Error = Infallible;

    #[inline]
    fn try_allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Self::Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        Ok(self.allocate(ctx))
    }

    #[inline]
    unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        self.allocate(ctx)
    }
}

impl<S, T, const N: usize> InfallibleMetadata for CopiedArray<S, T, N>
where
    S: Borrow<[i64; N]>,
    T: Borrow<[i64; N]>,
{
    #[inline]
    fn allocate<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        self.allocate(ctx)
    }
}

#[inline]
unsafe fn allocate_copied_array<M, const N: usize>() -> (NonNull<M>, *mut i64, *mut i64) {
    let (layout, metadata_offset) = copied_array_layout::<M, N>();
    let managed_tensor: NonNull<M> = unsafe { allocate(layout) };
    let shape = unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(metadata_offset)
            .cast::<i64>()
    };
    let strides = unsafe { shape.add(N) };
    (managed_tensor, shape, strides)
}

#[inline]
unsafe fn allocate_copied_slice<M>(ndim: usize) -> (NonNull<M>, *mut i64, *mut i64) {
    let (layout, allocation_ndim_offset, metadata_offset) = copied_slice_layout::<M>(ndim);
    let managed_tensor: NonNull<M> = unsafe { allocate(layout) };
    unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(allocation_ndim_offset)
            .cast::<usize>()
            .write(ndim);
    }
    let shape = unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(metadata_offset)
            .cast::<i64>()
    };
    let strides = unsafe { shape.add(ndim) };
    (managed_tensor, shape, strides)
}

#[inline]
unsafe fn allocate_borrowed<M>() -> NonNull<M> {
    unsafe { allocate(Layout::new::<M>()) }
}

/// Copies fixed-rank metadata into storage trailing the managed tensor.
///
/// The allocation is exactly `M + 2 * N * i64`; no rank is stored separately
/// because `N` is available to the monomorphized deleter.
#[derive(Debug, Clone, Copy)]
pub struct CopiedArray<S, T, const N: usize> {
    shape: S,
    strides: T,
}

impl<S, T, const N: usize> CopiedArray<S, T, N>
where
    S: Borrow<[i64; N]>,
    T: Borrow<[i64; N]>,
{
    #[inline]
    pub fn new(shape: S, strides: T) -> Self {
        Self { shape, strides }
    }

    #[inline]
    pub fn allocate<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        const { assert!(N <= i32::MAX as usize, "N must fit in i32") };

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_array::<M, N>();
            copy_i64_metadata(self.shape.borrow(), shape);
            copy_i64_metadata(self.strides.borrow(), strides);
            initialize(
                managed_tensor,
                shape,
                strides,
                N as i32,
                ctx,
                drop_copied_array::<C, M, N>,
            )
        }
    }
}

impl<S, T> Metadata for CopiedSlice<S, T>
where
    S: AsRef<[i64]>,
    T: AsRef<[i64]>,
{
    type Error = Error;

    #[inline]
    fn try_allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Self::Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        self.allocate(ctx)
    }

    #[inline]
    unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        let shape_src = self.shape.as_ref();
        let strides_src = self.strides.as_ref();
        let ndim = shape_src.len();
        debug_assert_eq!(shape_src.len(), strides_src.len());
        debug_assert!(ndim <= i32::MAX as usize);

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_slice::<M>(ndim);
            copy_i64_metadata(shape_src, shape);
            copy_i64_metadata_n(strides_src, strides, ndim);
            initialize(
                managed_tensor,
                shape,
                strides,
                ndim as i32,
                ctx,
                drop_copied_slice::<C, M>,
            )
        }
    }
}

/// Copies runtime-rank metadata into trailing storage.
///
/// Dynamic copied storage records its allocation rank outside the public
/// `DLTensor`, because consumers may mutate `DLTensor.ndim` before calling
/// the deleter.
#[derive(Debug, Clone, Copy)]
pub struct CopiedSlice<S, T> {
    shape: S,
    strides: T,
}

impl<S, T> CopiedSlice<S, T>
where
    S: AsRef<[i64]>,
    T: AsRef<[i64]>,
{
    #[inline]
    pub fn new(shape: S, strides: T) -> Self {
        Self { shape, strides }
    }

    #[inline]
    pub fn allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        let shape = self.shape.as_ref();
        let strides = self.strides.as_ref();
        let ndim = checked_ndim(shape.len(), strides.len())?;

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_slice::<M>(ndim as usize);
            copy_i64_metadata(self.shape.as_ref(), shape);
            copy_i64_metadata(self.strides.as_ref(), strides);

            Ok(initialize(
                managed_tensor,
                shape,
                strides,
                ndim,
                ctx,
                drop_copied_slice::<C, M>,
            ))
        }
    }
}

/// Borrows fixed-rank metadata and allocates only the managed tensor.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedArray<'a, const N: usize> {
    shape: &'a [i64; N],
    strides: &'a [i64; N],
}

impl<'a, const N: usize> BorrowedArray<'a, N> {
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

#[inline]
unsafe fn copy_i64_metadata(src: &[i64], dst: *mut i64) {
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

#[inline]
unsafe fn copy_i64_metadata_n(src: &[i64], dst: *mut i64, len: usize) {
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len) };
}

#[inline]
unsafe fn initialize<C, M>(
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
fn checked_ndim(shape_len: usize, strides_len: usize) -> Result<i32, Error> {
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

unsafe extern "C" fn drop_copied_array<C: OpaqueContext, M: ManagedTensorBase, const N: usize>(
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

unsafe extern "C" fn drop_copied_slice<C: OpaqueContext, M: ManagedTensorBase>(managed: *mut M) {
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

unsafe extern "C" fn drop_borrowed<C: OpaqueContext, M: ManagedTensorBase>(managed: *mut M) {
    if managed.is_null() {
        return;
    }
    unsafe {
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), Layout::new::<M>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::DLManagedTensor;

    #[test]
    fn copied_array_places_metadata_after_header() {
        let shape = [2i64, 3];
        let strides = [3i64, 1];
        let managed =
            CopiedArray::new(&shape, &strides).allocate::<_, DLManagedTensor>(Box::new(()));
        let tensor = unsafe { managed.as_ref().tensor() };

        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
        unsafe { DLManagedTensor::drop_raw(managed.as_ptr()) };
    }

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

    #[test]
    fn mismatched_dynamic_metadata_is_rejected_before_allocation() {
        let shape = [2i64, 3];
        let strides = [1i64];

        let result =
            CopiedSlice::new(&shape, &strides).allocate::<_, DLManagedTensor>(Box::new(()));

        assert!(matches!(
            result,
            Err(Error::MismatchedLength {
                shape_len: 2,
                strides_len: 1
            })
        ));
    }

    #[test]
    fn copied_slice_drop_uses_allocation_rank_not_tensor_ndim() {
        let shape = [2i64, 3];
        let strides = [3i64, 1];
        let mut managed = CopiedSlice::new(&shape, &strides)
            .allocate::<_, DLManagedTensor>(Box::new(()))
            .unwrap();

        unsafe {
            managed.as_mut().tensor_mut().ndim = 99;
            DLManagedTensor::drop_raw(managed.as_ptr());
        }
    }
}
