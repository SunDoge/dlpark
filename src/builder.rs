//! Deferred DLPack builder.
//!
//! The metadata type determines the allocation layout, whether metadata is
//! copied or borrowed, and whether building can fail. The builder itself stays
//! on the stack and only collects the scalar `DLTensor` fields.

use crate::{
    DlpackFlags, ManagedBox, ManagedTensorBase, OpaqueContext,
    ffi::{DLDataType, DLDevice},
    metadata::{BorrowedArray, BorrowedSlice, InfallibleMetadata, Metadata},
};
use std::{ffi::c_void, ptr::NonNull};

pub use crate::metadata::Error;

pub struct Builder<C, L> {
    ctx: C,
    metadata: L,
    fields: TensorFields,
}

#[derive(Clone, Copy)]
struct TensorFields {
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    byte_offset: u64,
    flags: DlpackFlags,
}

impl<C, L> Builder<C, L> {
    #[inline]
    pub fn new(ctx: C, metadata: L) -> Self {
        Self {
            ctx,
            metadata,
            fields: TensorFields {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                dtype: DLDataType::default(),
                byte_offset: 0,
                flags: DlpackFlags::empty(),
            },
        }
    }

    #[inline]
    pub fn data(mut self, data: *mut c_void) -> Self {
        self.fields.data = data;
        self
    }

    #[inline]
    pub fn metadata<L2>(self, metadata: L2) -> Builder<C, L2> {
        let Self { ctx, fields, .. } = self;
        Builder {
            ctx,
            metadata,
            fields,
        }
    }

    #[inline]
    pub fn device(mut self, device: DLDevice) -> Self {
        self.fields.device = device;
        self
    }

    #[inline]
    pub fn dtype(mut self, dtype: DLDataType) -> Self {
        self.fields.dtype = dtype;
        self
    }

    #[inline]
    pub fn byte_offset(mut self, byte_offset: u64) -> Self {
        self.fields.byte_offset = byte_offset;
        self
    }

    /// Sets DLPack flags, erroring if this would newly assert
    /// [`DlpackFlags::IS_COPIED`] (turn it on when it wasn't already set on
    /// this builder). See [`Self::flags_unchecked`].
    #[inline]
    pub fn flags(mut self, flags: DlpackFlags) -> Result<Self, crate::tensor::Error> {
        if flags.newly_asserts_is_copied(self.fields.flags) {
            return Err(crate::tensor::Error::CannotAssertIsCopied);
        }
        self.fields.flags = flags;
        Ok(self)
    }

    /// Adds DLPack flags without clearing flags already set on the builder.
    ///
    /// This errors if the operation would newly assert
    /// [`DlpackFlags::IS_COPIED`]. Use [`Self::insert_flags_unchecked`] when
    /// the caller can prove the required ownership guarantee.
    #[inline]
    pub fn insert_flags(mut self, flags: DlpackFlags) -> Result<Self, crate::tensor::Error> {
        if flags.newly_asserts_is_copied(self.fields.flags) {
            return Err(crate::tensor::Error::CannotAssertIsCopied);
        }
        self.fields.flags.insert(flags);
        Ok(self)
    }

    /// Sets DLPack flags verbatim, including [`DlpackFlags::IS_COPIED`].
    ///
    /// # Safety
    ///
    /// If `flags` includes `IS_COPIED`, the caller must ensure that no other
    /// reference to the tensor's data exists — see
    /// [`ManagedTensorBase::set_flags_unchecked`].
    #[inline]
    pub unsafe fn flags_unchecked(mut self, flags: DlpackFlags) -> Self {
        self.fields.flags = flags;
        self
    }

    /// Adds DLPack flags without clearing flags already set on the builder.
    ///
    /// # Safety
    ///
    /// If `flags` includes `IS_COPIED`, the caller must ensure that no other
    /// reference to the tensor's data exists.
    #[inline]
    pub unsafe fn insert_flags_unchecked(mut self, flags: DlpackFlags) -> Self {
        self.fields.flags.insert(flags);
        self
    }
}

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

    #[inline]
    pub fn build<M>(self) -> ManagedBox<M>
    where
        M: ManagedTensorBase,
    {
        unsafe { ManagedBox::new_unchecked(self.build_raw()) }
    }
}

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

#[inline]
unsafe fn finish<M>(mut managed: NonNull<M>, fields: TensorFields) -> NonNull<M>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ffi::{DLDeviceType, DLManagedTensor, DLManagedTensorVersioned},
        metadata,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct TestContext {
        drop_count: Arc<AtomicUsize>,
    }

    unsafe impl OpaqueContext for TestContext {
        fn into_raw(self) -> *mut c_void {
            Box::into_raw(Box::new(self)).cast()
        }

        unsafe fn drop_raw(raw: *mut c_void) {
            if !raw.is_null() {
                let boxed = unsafe { Box::from_raw(raw.cast::<TestContext>()) };
                boxed.drop_count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    fn context() -> (TestContext, Arc<AtomicUsize>) {
        let drop_count = Arc::new(AtomicUsize::new(0));
        (
            TestContext {
                drop_count: drop_count.clone(),
            },
            drop_count,
        )
    }

    #[test]
    fn copied_array_build_is_infallible() {
        let (ctx, drop_count) = context();
        let shape = [1, 2, 3];
        let strides = [6, 3, 1];

        let tensor: ManagedBox<DLManagedTensor> =
            Builder::new(ctx, metadata::CopiedArray::new(&shape, &strides)).build();

        assert_eq!(tensor.tensor().ndim, 3);
        assert_eq!(tensor.shape().unwrap(), shape);
        assert_eq!(tensor.strides().unwrap().unwrap(), strides);
        assert_eq!(tensor.tensor().device.device_type, DLDeviceType::CPU);
        drop(tensor);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn copied_array_build_raw_transfers_ownership() {
        let (ctx, drop_count) = context();
        let shape = [2, 3];
        let strides = [3, 1];

        let raw = Builder::new(ctx, metadata::CopiedArray::new(&shape, &strides))
            .build_raw::<DLManagedTensor>();

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        let tensor = unsafe { ManagedBox::new_unchecked(raw) };
        assert_eq!(tensor.shape().unwrap(), shape);
        drop(tensor);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn copied_slice_build_validates_lengths() {
        let (ctx, _) = context();
        let result: Result<ManagedBox<DLManagedTensor>, Error> =
            Builder::new(ctx, metadata::CopiedSlice::new(&[1, 2], &[2, 1, 1])).try_build();

        assert!(matches!(
            result,
            Err(Error::MismatchedLength {
                shape_len: 2,
                strides_len: 3
            })
        ));
    }

    #[test]
    fn copied_slice_try_build_raw_transfers_ownership() {
        let (ctx, drop_count) = context();
        let shape = [2, 3];
        let strides = [3, 1];

        let raw = Builder::new(ctx, metadata::CopiedSlice::new(&shape, &strides))
            .try_build_raw::<DLManagedTensor>()
            .unwrap();

        let tensor = unsafe { ManagedBox::new_unchecked(raw) };
        assert_eq!(tensor.strides().unwrap().unwrap(), strides);
        drop(tensor);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn generic_array_converts_metadata_without_temporary_i64_arrays() {
        let (ctx, _) = context();
        let shape = [2u32, 3];
        let strides = [3isize, 1];

        let tensor: ManagedBox<DLManagedTensor> =
            Builder::new(ctx, metadata::GenericArray::new(&shape, &strides))
                .try_build()
                .unwrap();

        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }

    #[test]
    fn generic_slice_converts_metadata_and_validates_lengths() {
        let (ctx, _) = context();
        let shape = vec![2u32, 3];
        let strides = vec![3isize, 1];

        let tensor: ManagedBox<DLManagedTensor> =
            Builder::new(ctx, metadata::GenericSlice::new(&shape, &strides))
                .try_build()
                .unwrap();
        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);

        let (ctx, _) = context();
        let result: Result<ManagedBox<DLManagedTensor>, Error> =
            Builder::new(ctx, metadata::GenericSlice::new(&[1usize, 2], &[1isize])).try_build();
        assert!(matches!(result, Err(Error::MismatchedLength { .. })));
    }

    #[test]
    fn generic_metadata_reports_conversion_axis() {
        let (ctx, drop_count) = context();
        let result: Result<ManagedBox<DLManagedTensor>, Error> = Builder::new(
            ctx,
            metadata::GenericArray::new(&[1u64, i64::MAX as u64 + 1], &[1u64, 1]),
        )
        .try_build();

        assert!(matches!(result, Err(Error::ShapeValueOverflow { axis: 1 })));
        assert_eq!(Arc::strong_count(&drop_count), 1);
    }

    #[test]
    fn borrowed_array_reuses_metadata() {
        let (ctx, _) = context();
        let shape = [2, 4];
        let strides = [4, 1];

        let tensor: ManagedBox<DLManagedTensor> =
            unsafe { Builder::new(ctx, metadata::BorrowedArray::new(&shape, &strides)).build() };

        assert_eq!(tensor.tensor().shape, shape.as_ptr().cast_mut());
        assert_eq!(tensor.tensor().strides, strides.as_ptr().cast_mut());
    }

    #[test]
    fn borrowed_slice_build_validates_lengths() {
        let (ctx, _) = context();
        let result: Result<ManagedBox<DLManagedTensor>, Error> = unsafe {
            Builder::new(ctx, metadata::BorrowedSlice::new(&[1, 2], &[2, 1, 1])).try_build()
        };

        assert!(matches!(
            result,
            Err(Error::MismatchedLength {
                shape_len: 2,
                strides_len: 3
            })
        ));
    }

    #[test]
    fn scalar_fields_and_versioned_flags_are_applied() {
        let (ctx, _) = context();
        let shape = [3];
        let strides = [1];
        let data = NonNull::<u8>::dangling().as_ptr().cast();

        let tensor: ManagedBox<DLManagedTensorVersioned> =
            Builder::new(ctx, metadata::CopiedArray::new(&shape, &strides))
                .data(data)
                .byte_offset(4)
                .flags(DlpackFlags::READ_ONLY)
                .unwrap()
                .build();

        assert_eq!(tensor.tensor().data, data);
        assert_eq!(tensor.tensor().byte_offset, 4);
        assert_eq!(tensor.flags(), DlpackFlags::READ_ONLY);
    }

    #[test]
    fn flags_rejects_newly_asserting_is_copied() {
        let (ctx, _) = context();

        let error = match Builder::new(ctx, metadata::CopiedArray::new(&[3], &[1]))
            .flags(DlpackFlags::READ_ONLY | DlpackFlags::IS_COPIED)
        {
            Ok(_) => panic!("newly asserting IS_COPIED through the safe setter should fail"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::tensor::Error::CannotAssertIsCopied));
    }

    #[test]
    fn insert_flags_rejects_newly_asserting_is_copied() {
        let (ctx, _) = context();

        let error = match Builder::new(ctx, metadata::CopiedArray::new(&[3], &[1]))
            .insert_flags(DlpackFlags::IS_COPIED)
        {
            Ok(_) => panic!("newly inserting IS_COPIED through the safe setter should fail"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::tensor::Error::CannotAssertIsCopied));
    }

    #[test]
    fn insert_flags_preserves_existing_flags() {
        let (ctx, _) = context();
        let builder = unsafe {
            Builder::new(ctx, metadata::CopiedArray::new(&[3], &[1]))
                .flags_unchecked(DlpackFlags::IS_COPIED)
        };
        let tensor: ManagedBox<DLManagedTensorVersioned> = builder
            .insert_flags(DlpackFlags::READ_ONLY)
            .unwrap()
            .build();

        assert_eq!(
            tensor.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn flags_unchecked_keeps_is_copied() {
        let (ctx, _) = context();

        let tensor: ManagedBox<DLManagedTensorVersioned> = unsafe {
            Builder::new(ctx, metadata::CopiedArray::new(&[3], &[1]))
                .flags_unchecked(DlpackFlags::READ_ONLY | DlpackFlags::IS_COPIED)
        }
        .build();

        assert_eq!(
            tensor.flags(),
            DlpackFlags::READ_ONLY | DlpackFlags::IS_COPIED
        );
    }

    #[test]
    fn metadata_replaces_layout_and_keeps_scalar_fields() {
        let (ctx, _) = context();
        let old_shape = [1];
        let old_strides = [1];
        let new_shape = [2, 3];
        let new_strides = [3, 1];
        let data = NonNull::<u8>::dangling().as_ptr().cast();

        let tensor: ManagedBox<DLManagedTensor> =
            Builder::new(ctx, metadata::CopiedArray::new(&old_shape, &old_strides))
                .data(data)
                .metadata(metadata::CopiedArray::new(&new_shape, &new_strides))
                .build();

        assert_eq!(tensor.tensor().data, data);
        assert_eq!(tensor.tensor().ndim, 2);
        assert_eq!(tensor.shape().unwrap(), new_shape);
        assert_eq!(tensor.strides().unwrap().unwrap(), new_strides);
    }
}
