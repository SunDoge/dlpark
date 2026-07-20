use super::{Builder, Error, build::finish};
use crate::{
    ManagedBox, ManagedTensorBase, OpaqueContext,
    metadata::{BorrowedArray, BorrowedSlice},
};

impl<C, const N: usize> Builder<C, BorrowedArray<'_, N>>
where
    C: OpaqueContext,
{
    /// Builds the tensor and transfers ownership to a raw DLPack pointer.
    ///
    /// # Safety
    ///
    /// The borrowed arrays must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn build_raw<M>(self) -> *mut M
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed = unsafe { metadata.allocate::<C, M>(ctx) };
        unsafe { finish(managed, fields) }.as_ptr()
    }

    /// Builds a tensor that points to caller-owned shape and strides arrays.
    ///
    /// # Safety
    ///
    /// The borrowed arrays must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn build<M>(self) -> ManagedBox<M>
    where
        M: ManagedTensorBase,
    {
        unsafe { ManagedBox::new_unchecked(self.build_raw()) }
    }

    /// Tries to build the tensor and transfer ownership to a raw DLPack
    /// pointer.
    ///
    /// # Safety
    ///
    /// The borrowed arrays must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn try_build_raw<M>(self) -> Result<*mut M, std::convert::Infallible>
    where
        M: ManagedTensorBase,
    {
        Ok(unsafe { self.build_raw() })
    }

    /// Tries to build a tensor that points to caller-owned shape and strides
    /// arrays.
    ///
    /// # Safety
    ///
    /// The borrowed arrays must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn try_build<M>(self) -> Result<ManagedBox<M>, std::convert::Infallible>
    where
        M: ManagedTensorBase,
    {
        Ok(unsafe { self.build() })
    }
}

impl<C> Builder<C, BorrowedSlice<'_>>
where
    C: OpaqueContext,
{
    /// Tries to build the tensor and transfer ownership to a raw DLPack
    /// pointer.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn try_build_raw<M>(self) -> Result<*mut M, Error>
    where
        M: ManagedTensorBase,
    {
        let Self {
            ctx,
            metadata,
            fields,
        } = self;
        let managed = unsafe { metadata.allocate::<C, M>(ctx)? };
        Ok(unsafe { finish(managed, fields) }.as_ptr())
    }

    /// Builds a tensor that points to caller-owned shape and strides slices.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the returned managed tensor and must
    /// not be mutated through the DLPack `shape`/`strides` pointers while it
    /// is alive.
    #[inline]
    pub unsafe fn try_build<M>(self) -> Result<ManagedBox<M>, Error>
    where
        M: ManagedTensorBase,
    {
        unsafe { self.try_build_raw() }.map(|raw| unsafe { ManagedBox::new_unchecked(raw) })
    }

    /// Builds the tensor without checking runtime metadata invariants.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the returned managed tensor, and shape
    /// and strides must have the same length with `ndim` fitting in `i32`.
    /// They must not be mutated through the DLPack `shape`/`strides` pointers
    /// while the managed tensor is alive.
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

    /// Builds a tensor that points to caller-owned shape and strides slices
    /// without checking runtime metadata invariants.
    ///
    /// # Safety
    ///
    /// The borrowed slices must outlive the returned managed tensor, and shape
    /// and strides must have the same length with `ndim` fitting in `i32`.
    /// They must not be mutated through the DLPack `shape`/`strides` pointers
    /// while the managed tensor is alive.
    #[inline]
    pub unsafe fn build_unchecked<M>(self) -> ManagedBox<M>
    where
        M: ManagedTensorBase,
    {
        unsafe { ManagedBox::new_unchecked(self.build_raw_unchecked()) }
    }
}
