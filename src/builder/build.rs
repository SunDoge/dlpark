use super::{Builder, Error, TensorFields};
use crate::{
    ManagedBox, ManagedTensorBase, OpaqueContext,
    metadata::{FromContext, InfallibleMetadata, Metadata},
};
use std::ptr::NonNull;

impl<C, L> Builder<C, L>
where
    C: OpaqueContext,
    L: Metadata,
{
    /// Tries to build the tensor and transfer ownership to a raw DLPack
    /// pointer.
    #[inline]
    pub fn try_build_raw<M>(self) -> Result<*mut M, L::Error>
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed = metadata.try_allocate::<C, M>(ctx)?;
        Ok(unsafe { finish(managed, fields) }.as_ptr())
    }

    /// Validates runtime metadata, allocates the managed tensor, and returns
    /// an owning handle.
    #[inline]
    pub fn try_build<M>(self) -> Result<ManagedBox<M>, L::Error>
    where
        M: ManagedTensorBase,
    {
        self.try_build_raw()
            .map(|raw| unsafe { ManagedBox::new_unchecked(raw) })
    }

    /// Builds the tensor without checking runtime metadata invariants.
    ///
    /// # Safety
    ///
    /// The metadata must satisfy the invariants required by its unchecked
    /// allocator. For dynamic metadata this includes matching shape/strides
    /// lengths and `ndim <= i32::MAX`; violating those requirements may cause
    /// out-of-bounds reads or an invalid `DLTensor`.
    #[inline]
    pub unsafe fn build_raw_unchecked<M>(self) -> *mut M
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed = unsafe { metadata.allocate_unchecked::<C, M>(ctx) };
        unsafe { finish(managed, fields) }.as_ptr()
    }

    /// Builds the tensor without checking runtime metadata invariants.
    ///
    /// # Safety
    ///
    /// The metadata must satisfy the invariants required by its unchecked
    /// allocator. For dynamic metadata this includes matching shape/strides
    /// lengths and `ndim <= i32::MAX`; violating those requirements may cause
    /// out-of-bounds reads or an invalid `DLTensor`.
    #[inline]
    pub unsafe fn build_unchecked<M>(self) -> ManagedBox<M>
    where
        M: ManagedTensorBase,
    {
        unsafe { ManagedBox::new_unchecked(self.build_raw_unchecked()) }
    }
}

impl<C, F, A, B> Builder<C, FromContext<F, A, B>>
where
    C: OpaqueContext,
    A: Copy + TryInto<i64>,
    B: Copy + TryInto<i64>,
    F: FnOnce(&C) -> (&[A], &[B]),
{
    /// Derives, validates, and copies metadata, then transfers ownership to a
    /// raw DLPack pointer.
    #[inline]
    pub fn try_build_raw<M>(self) -> Result<*mut M, Error>
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed =
            crate::metadata::try_allocate_generic_from_context(ctx, metadata.into_inner())?;
        Ok(unsafe { finish(managed, fields) }.as_ptr())
    }

    /// Derives metadata from the context and returns an owning managed tensor.
    #[inline]
    pub fn try_build<M>(self) -> Result<ManagedBox<M>, Error>
    where
        M: ManagedTensorBase,
    {
        self.try_build_raw()
            .map(|raw| unsafe { ManagedBox::new_unchecked(raw) })
    }
}

impl<C, L> Builder<C, L>
where
    C: OpaqueContext,
    L: InfallibleMetadata,
{
    /// Builds the tensor and transfers ownership to a raw DLPack pointer.
    #[inline]
    pub fn build_raw<M>(self) -> *mut M
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed = metadata.allocate::<C, M>(ctx);
        unsafe { finish(managed, fields) }.as_ptr()
    }

    /// Builds an owning managed tensor when the metadata layout is infallible.
    #[inline]
    pub fn build<M>(self) -> ManagedBox<M>
    where
        M: ManagedTensorBase,
    {
        unsafe { ManagedBox::new_unchecked(self.build_raw()) }
    }
}
#[inline]
pub(super) unsafe fn finish<M>(mut managed: NonNull<M>, fields: TensorFields) -> NonNull<M>
where
    M: ManagedTensorBase,
{
    unsafe {
        let managed_ref = managed.as_mut();
        {
            let tensor = managed_ref.tensor_mut();
            tensor.data = fields.data;
            tensor.device = fields.device;
            tensor.dtype = fields.dtype;
            tensor.byte_offset = fields.byte_offset;
        }
        managed_ref.set_flags_unchecked(fields.flags);
    }
    managed
}
