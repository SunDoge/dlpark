use std::ptr::NonNull;

use crate::ffi::{self, Flags};
use crate::traits::{MemoryLayout, TensorLike, TensorView};

use super::ManagerContext;

/// A safe wrapper around a versioned DLPack tensor that manages its memory lifecycle.
///
/// This struct provides a safe interface to work with DLPack tensors while ensuring proper
/// memory management through RAII. It wraps a `ffi::DlpackVersioned` pointer and handles
/// cleanup when the tensor is dropped.
pub struct SafeManagedTensorVersioned(ffi::DlpackVersioned);

impl Drop for SafeManagedTensorVersioned {
    /// Implements the Drop trait to ensure proper cleanup of the managed tensor.
    /// When this wrapper is dropped, it calls the tensor's deleter function if one exists.
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl SafeManagedTensorVersioned {
    /// Creates a new `SafeManagedTensorVersioned` from a raw pointer.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid and points to a properly initialized `ManagedTensorVersioned`
    /// - The pointer is not null
    /// - The tensor's memory is managed by a valid deleter function
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensorVersioned) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    /// Creates a new `SafeManagedTensorVersioned` from a `NonNull` pointer.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid and points to a properly initialized `ManagedTensorVersioned`
    /// - The tensor's memory is managed by a valid deleter function
    pub unsafe fn from_non_null(ptr: ffi::DlpackVersioned) -> Self {
        Self(ptr)
    }

    /// Converts the safe wrapper into a raw pointer, transferring ownership.
    ///
    /// # Safety
    /// The caller takes responsibility for managing the tensor's memory after this call.
    /// The original wrapper is forgotten to prevent double-free.
    pub unsafe fn into_raw(self) -> *mut ffi::ManagedTensorVersioned {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Converts the safe wrapper into a `NonNull` pointer, transferring ownership.
    ///
    /// The original wrapper is forgotten to prevent double-free.
    pub fn into_non_null(self) -> ffi::DlpackVersioned {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    /// Creates a new `SafeManagedTensorVersioned` from a tensor-like type.
    ///
    /// # Arguments
    /// * `t` - A type that implements `TensorLike` and has a valid `MemoryLayout`
    ///
    /// # Returns
    /// A new `SafeManagedTensorVersioned` with default flags
    pub fn new<T, L>(t: T) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        Self::with_flags(t, Flags::default())
    }

    /// Creates a new `SafeManagedTensorVersioned` from a tensor-like type with specified flags.
    ///
    /// # Arguments
    /// * `t` - A type that implements `TensorLike` and has a valid `MemoryLayout`
    /// * `flags` - Flags to set on the managed tensor
    ///
    /// # Returns
    /// A new `SafeManagedTensorVersioned` with the specified flags
    pub fn with_flags<T, L>(t: T, flags: Flags) -> Self
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        let ctx = ManagerContext::new(t);
        Self(ctx.into_dlpack_versioned(flags))
    }

    /// Returns the raw bits of the tensor's flags.
    pub fn flags_bits(&self) -> u64 {
        unsafe { self.0.as_ref().flags }
    }

    /// Returns the tensor's flags as an `Option<Flags>`.
    /// Returns `None` if the flags contain invalid bits.
    pub fn flags(&self) -> Option<Flags> {
        Flags::from_bits(self.flags_bits())
    }

    /// Returns the tensor's flags, truncating any invalid bits.
    pub fn flags_truncate(&self) -> Flags {
        Flags::from_bits_truncate(self.flags_bits())
    }

    /// Returns whether the tensor is marked as read-only.
    pub fn read_only(&self) -> bool {
        self.flags_truncate().contains(Flags::READ_ONLY)
    }

    /// Returns whether the tensor is a copy of another tensor.
    pub fn is_copied(&self) -> bool {
        self.flags_truncate().contains(Flags::IS_COPIED)
    }

    /// Returns whether the tensor's sub-byte type is padded.
    pub fn is_subbtype_type_padded(&self) -> bool {
        self.flags_truncate()
            .contains(Flags::IS_SUBBYTE_TYPE_PADDED)
    }
}

/// Implements `TensorView` to provide access to the underlying DLPack tensor.
impl TensorView for SafeManagedTensorVersioned {
    /// Returns a reference to the underlying DLPack tensor.
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

/// Implements `Deref` to allow treating the tensor as a byte slice.
impl std::ops::Deref for SafeManagedTensorVersioned {
    type Target = [u8];

    /// Dereferences the tensor to a byte slice.
    ///
    /// This allows the tensor to be used in contexts that expect a byte slice,
    /// such as reading or writing the tensor's data.
    fn deref(&self) -> &Self::Target {
        self.as_slice_untyped()
    }
}

impl AsRef<SafeManagedTensorVersioned> for SafeManagedTensorVersioned {
    fn as_ref(&self) -> &Self {
        self
    }
}
