use super::{
    Error, InfallibleMetadata, Metadata,
    allocation::{allocate_copied_array, copied_array_layout},
    raw::{
        copy_generic_metadata_unchecked, copy_i64_metadata, drop_copied_array, initialize,
        try_copy_generic_metadata,
    },
};
use crate::{OpaqueContext, managed_tensor::ManagedTensorBase};
use std::{borrow::Borrow, convert::Infallible, marker::PhantomData, ptr::NonNull};

unsafe impl<S, T, const N: usize> Metadata for CopiedArray<S, T, N>
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

unsafe impl<S, T, const N: usize> InfallibleMetadata for CopiedArray<S, T, N>
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
    /// Creates fixed-rank metadata that will be copied into the managed tensor
    /// allocation.
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

unsafe impl<S, T, A, B, const N: usize> Metadata for GenericArray<S, T, A, B, N>
where
    S: Borrow<[A; N]>,
    T: Borrow<[B; N]>,
    A: Copy + TryInto<i64>,
    B: Copy + TryInto<i64>,
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
        unsafe { self.allocate_unchecked(ctx) }
    }
}

/// Converts fixed-rank metadata into `i64` storage trailing the managed tensor.
///
/// This uses the same single allocation as [`CopiedArray`], but accepts
/// non-`i64` shape and stride elements and converts them while writing.
/// Shape and stride elements may use different source types. Both must
/// implement `Copy + TryInto<i64>`.
///
/// The rank is known at compile time, but element conversion can fail, so use
/// [`crate::Builder::try_build`] or [`crate::Builder::try_build_raw`].
///
/// # Example
///
/// ```
/// use dlpark::{Builder, legacy, metadata::GenericArray};
///
/// let shape = [2u32, 3];
/// let strides = [3isize, 1];
/// let tensor: legacy::Dlpack =
///     Builder::new(Box::new(()), GenericArray::new(&shape, &strides))
///         .try_build()
///         .unwrap();
///
/// assert_eq!(tensor.shape().unwrap(), &[2, 3]);
/// assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GenericArray<S, T, A, B, const N: usize> {
    shape: S,
    strides: T,
    marker: PhantomData<fn() -> (A, B)>,
}

impl<S, T, A, B, const N: usize> GenericArray<S, T, A, B, N>
where
    S: Borrow<[A; N]>,
    T: Borrow<[B; N]>,
    A: Copy + TryInto<i64>,
    B: Copy + TryInto<i64>,
{
    /// Creates fixed-rank metadata whose values are converted to `i64` while
    /// allocating.
    #[inline]
    pub fn new(shape: S, strides: T) -> Self {
        Self {
            shape,
            strides,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        const { assert!(N <= i32::MAX as usize, "N must fit in i32") };

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_array::<M, N>();
            if let Err(axis) = try_copy_generic_metadata(self.shape.borrow(), shape) {
                std::alloc::dealloc(
                    managed_tensor.as_ptr().cast(),
                    copied_array_layout::<M, N>().0,
                );
                return Err(Error::ShapeValueOverflow { axis });
            }
            if let Err(axis) = try_copy_generic_metadata(self.strides.borrow(), strides) {
                std::alloc::dealloc(
                    managed_tensor.as_ptr().cast(),
                    copied_array_layout::<M, N>().0,
                );
                return Err(Error::StrideValueOverflow { axis });
            }
            Ok(initialize(
                managed_tensor,
                shape,
                strides,
                N as i32,
                ctx,
                drop_copied_array::<C, M, N>,
            ))
        }
    }

    /// Allocates without checking whether metadata values fit in `i64`.
    ///
    /// # Safety
    ///
    /// Every shape and stride value must convert successfully to `i64`.
    #[inline]
    pub unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase,
    {
        const { assert!(N <= i32::MAX as usize, "N must fit in i32") };

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_array::<M, N>();
            copy_generic_metadata_unchecked(self.shape.borrow(), shape);
            copy_generic_metadata_unchecked(self.strides.borrow(), strides);
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
        assert_eq!(unsafe { tensor.shape() }.unwrap(), &[2, 3]);
        assert_eq!(unsafe { tensor.strides() }.unwrap().unwrap(), &[3, 1]);
        unsafe { DLManagedTensor::drop_raw(managed.as_ptr()) };
    }
}
