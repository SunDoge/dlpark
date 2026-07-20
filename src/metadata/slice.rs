use super::{
    Error, Metadata,
    allocation::{allocate_copied_slice, copied_slice_layout},
    raw::{
        checked_ndim, copy_generic_metadata_unchecked, copy_i64_metadata, copy_i64_metadata_n,
        drop_copied_slice, initialize, try_copy_generic_metadata,
    },
};
use crate::{OpaqueContext, managed_tensor::ManagedTensorBase};
use std::{marker::PhantomData, ptr::NonNull};

unsafe impl<S, T> Metadata for CopiedSlice<S, T>
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
    /// Creates runtime-rank metadata that will be copied into the managed
    /// tensor allocation.
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

unsafe impl<S, T, A, B> Metadata for GenericSlice<S, T, A, B>
where
    S: AsRef<[A]>,
    T: AsRef<[B]>,
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
        let shape_src = self.shape.as_ref();
        let strides_src = self.strides.as_ref();
        let ndim = shape_src.len();
        debug_assert_eq!(shape_src.len(), strides_src.len());
        debug_assert!(ndim <= i32::MAX as usize);

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_slice::<M>(ndim);
            copy_generic_metadata_unchecked(shape_src, shape);
            copy_generic_metadata_unchecked(strides_src, strides);
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

/// Converts runtime-rank metadata into `i64` storage trailing the managed tensor.
///
/// This uses the same single allocation as [`CopiedSlice`], but accepts
/// non-`i64` shape and stride elements and converts them while writing.
/// Shape and stride elements may use different source types. Both must
/// implement `Copy + TryInto<i64>`.
///
/// Construction validates that shape and strides have equal lengths and that
/// the resulting rank fits in `i32`. Use [`crate::Builder::try_build`] or
/// [`crate::Builder::try_build_raw`] with this metadata type.
///
/// # Example
///
/// ```
/// use dlpark::{Builder, legacy, metadata::GenericSlice};
///
/// let shape = vec![2u32, 3];
/// let strides = vec![3i16, 1];
/// let tensor: legacy::Dlpack =
///     Builder::new(Box::new(()), GenericSlice::new(&shape, &strides))
///         .try_build()
///         .unwrap();
///
/// assert_eq!(tensor.shape().unwrap(), &[2, 3]);
/// assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GenericSlice<S, T, A, B> {
    shape: S,
    strides: T,
    marker: PhantomData<fn() -> (A, B)>,
}

impl<S, T, A, B> GenericSlice<S, T, A, B>
where
    S: AsRef<[A]>,
    T: AsRef<[B]>,
    A: Copy + TryInto<i64>,
    B: Copy + TryInto<i64>,
{
    /// Creates runtime-rank metadata whose values are converted to `i64`
    /// while allocating.
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
        let shape = self.shape.as_ref();
        let strides = self.strides.as_ref();
        let ndim = checked_ndim(shape.len(), strides.len())?;

        unsafe {
            let (managed_tensor, shape, strides) = allocate_copied_slice::<M>(ndim as usize);
            if let Err(axis) = try_copy_generic_metadata(self.shape.as_ref(), shape) {
                std::alloc::dealloc(
                    managed_tensor.as_ptr().cast(),
                    copied_slice_layout::<M>(ndim as usize).0,
                );
                return Err(Error::ShapeValueOverflow { axis });
            }
            if let Err(axis) = try_copy_generic_metadata(self.strides.as_ref(), strides) {
                std::alloc::dealloc(
                    managed_tensor.as_ptr().cast(),
                    copied_slice_layout::<M>(ndim as usize).0,
                );
                return Err(Error::StrideValueOverflow { axis });
            }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::DLManagedTensor;

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
