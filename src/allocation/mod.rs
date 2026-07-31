//! Low-level allocation of managed tensors with writable metadata storage.
//!
//! A producer allocates a managed tensor together with its shape and strides
//! storage, installs an owning context and deleter, writes the scalar tensor
//! fields, then finishes. The flow is:
//!
//! 1. [`fixed::Allocation`] (compile-time rank) or [`dynamic::Allocation`]
//!    (runtime rank) allocates the managed tensor header plus metadata
//!    storage. [`fixed::Storage`] selects between inline `[i64; N]` storage
//!    ([`fixed::Copied`]) and no inline storage ([`fixed::Borrowed`]); the
//!    dynamic allocation stores shape and strides in a trailing `i64` buffer.
//! 2. `initialize(ctx)` installs the owning context and deleter and returns
//!    an [`Initialized`] handle.
//! 3. [`Initialized`] setters write the data pointer, device, dtype, byte
//!    offset, and flags; [`Initialized::finish`] returns the owning
//!    [`Managed<M>`](crate::Managed), consuming the allocation.
//!
//! The [`crate::metadata`] helpers drive steps 1–2 from shape and strides
//! values; interop producers drive them from a boxed container, which also
//! serves as the owning context.

use snafu::Snafu;
use std::{alloc::Layout, ptr::NonNull};

pub mod dynamic;
pub mod fixed;

/// Errors raised while allocating a managed tensor.
#[derive(Debug, Snafu)]
pub enum Error {
    /// The dimension count exceeds `i32::MAX`.
    #[snafu(display("dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        /// The offending dimension count.
        ndim: usize,
    },

    /// The managed tensor allocation layout overflows `usize`.
    #[snafu(display("managed tensor allocation layout overflows usize"))]
    LayoutOverflow,
}

/// An initialized managed tensor paired with allocation-specific metadata.
pub struct Initialized<M: crate::ManagedTensorBase, Storage> {
    pub(super) managed: crate::Managed<M>,
    pub(super) storage: Storage,
}

impl<M: crate::ManagedTensorBase, Storage> Initialized<M, Storage> {
    /// Returns the embedded descriptor for low-level initialization.
    pub fn tensor_mut(&mut self) -> &mut crate::ffi::DLTensor {
        unsafe { (&mut *self.managed.as_ptr()).tensor_mut() }
    }

    /// Sets the base data pointer stored in `DLTensor`.
    ///
    /// The pointer is not dereferenced during initialization. The caller must
    /// satisfy its validity and lifetime requirements before calling
    /// [`Self::finish`].
    pub fn set_data(&mut self, data: *mut std::ffi::c_void) -> &mut Self {
        self.tensor_mut().data = data;
        self
    }

    /// Sets the DLPack device descriptor.
    pub fn set_device(&mut self, device: crate::ffi::DLDevice) -> &mut Self {
        self.tensor_mut().device = device;
        self
    }

    /// Sets the DLPack element type descriptor.
    pub fn set_dtype(&mut self, dtype: crate::ffi::DLDataType) -> &mut Self {
        self.tensor_mut().dtype = dtype;
        self
    }

    /// Sets the byte offset from the base data pointer.
    pub fn set_byte_offset(&mut self, byte_offset: u64) -> &mut Self {
        self.tensor_mut().byte_offset = byte_offset;
        self
    }

    /// Sets flags unless doing so would newly assert `IS_COPIED`.
    pub fn set_flags(
        &mut self,
        flags: crate::DlpackFlags,
    ) -> Result<&mut Self, crate::tensor::Error> {
        if flags.newly_asserts_is_copied(self.managed.flags()) {
            return Err(crate::tensor::Error::CannotAssertIsCopied);
        }
        unsafe { (&mut *self.managed.as_ptr()).set_flags_unchecked(flags) };
        Ok(self)
    }

    /// Sets flags verbatim, including `IS_COPIED`.
    ///
    /// If `flags` includes `IS_COPIED`, the caller must ensure the producer
    /// actually created a copy for this export.
    pub fn set_flags_unchecked(&mut self, flags: crate::DlpackFlags) -> &mut Self {
        unsafe { (&mut *self.managed.as_ptr()).set_flags_unchecked(flags) };
        self
    }

    /// Finishes initialization and returns a locally produced tensor.
    ///
    /// # Safety
    ///
    /// The completed descriptor must satisfy the DLPack contract. Its data and
    /// metadata pointers must remain valid until the tensor is dropped, and
    /// its flags must accurately describe aliasing and mutability.
    pub unsafe fn finish(self) -> crate::Managed<M> {
        self.managed
    }
}

impl<Storage> Initialized<crate::ffi::DLManagedTensorVersioned, Storage> {
    /// Returns the version declared by the initialized managed tensor.
    pub fn version(&self) -> crate::ffi::DLPackVersion {
        unsafe { (*self.managed.as_ptr()).version }
    }

    /// Sets a compatible version declared by the initialized managed tensor.
    pub fn set_version(
        &mut self,
        version: crate::ffi::DLPackVersion,
    ) -> Result<&mut Self, crate::VersionError> {
        crate::version::validate_version(version)?;
        unsafe { (*self.managed.as_ptr()).version = version };
        Ok(self)
    }
}

fn allocate<M>(layout: Layout) -> NonNull<M> {
    let pointer = unsafe { std::alloc::alloc(layout) }.cast::<M>();
    NonNull::new(pointer).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

fn empty_tensor(ndim: i32) -> crate::ffi::DLTensor {
    crate::ffi::DLTensor::from_parts(std::ptr::null_mut(), std::ptr::null_mut(), ndim)
}
