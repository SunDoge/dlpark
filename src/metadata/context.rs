use super::{
    Error,
    allocation::{allocate_copied_slice, copied_slice_layout},
    raw::{checked_ndim, drop_copied_slice, initialize, try_copy_generic_metadata},
};
use crate::{OpaqueContext, managed_tensor::ManagedTensorBase};
use std::{marker::PhantomData, ptr::NonNull};

/// Copies metadata derived from an owning context after that context has
/// reached this function's stack frame.
#[derive(Debug, Clone, Copy)]
pub struct FromContext<F, A, B> {
    derive: F,
    marker: PhantomData<fn() -> (A, B)>,
}

impl<F, A, B> FromContext<F, A, B> {
    #[inline]
    pub(crate) fn new(derive: F) -> Self {
        Self {
            derive,
            marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn into_inner(self) -> F {
        self.derive
    }
}

#[inline]
pub(crate) fn try_allocate_generic_from_context<C, M, A, B, F>(
    ctx: C,
    derive: F,
) -> Result<NonNull<M>, Error>
where
    C: OpaqueContext,
    M: ManagedTensorBase,
    A: Copy + TryInto<i64>,
    B: Copy + TryInto<i64>,
    F: FnOnce(&C) -> (&[A], &[B]),
{
    let (shape_src, strides_src) = derive(&ctx);
    let ndim = checked_ndim(shape_src.len(), strides_src.len())?;

    unsafe {
        let (managed_tensor, shape, strides) = allocate_copied_slice::<M>(ndim as usize);
        if let Err(axis) = try_copy_generic_metadata(shape_src, shape) {
            std::alloc::dealloc(
                managed_tensor.as_ptr().cast(),
                copied_slice_layout::<M>(ndim as usize).0,
            );
            return Err(Error::ShapeValueOverflow { axis });
        }
        if let Err(axis) = try_copy_generic_metadata(strides_src, strides) {
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
