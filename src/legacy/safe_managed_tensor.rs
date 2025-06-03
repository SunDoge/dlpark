use super::ManagerContext;
use crate::ffi;
use crate::traits::{MemoryLayout, TensorLike, TensorView};
use std::ptr::NonNull;

/// A safe wrapper around DLPack tensor that manages its lifetime.
///
/// This struct provides safe memory management for DLPack tensors by ensuring
/// proper cleanup when the tensor is dropped. It wraps a raw DLPack tensor pointer
/// and calls the appropriate deleter function when the tensor is no longer needed.
pub struct SafeManagedTensor(ffi::Dlpack);

impl Drop for SafeManagedTensor {
    /// Implements the Drop trait to ensure proper cleanup of the tensor.
    /// When this struct is dropped, it calls the deleter function if one exists
    /// to free the underlying tensor memory.
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = self.0.as_ref().deleter {
                deleter(self.0.as_ptr());
            }
        }
    }
}

impl SafeManagedTensor {
    /// Creates a new SafeManagedTensor from a raw pointer to a ManagedTensor.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid and points to a properly initialized ManagedTensor
    /// - The tensor's memory is managed by a valid deleter function
    /// - The pointer is not used after being wrapped in SafeManagedTensor
    pub unsafe fn from_raw(ptr: *mut ffi::ManagedTensor) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    /// Creates a new SafeManagedTensor from a NonNull DLPack pointer.
    ///
    /// # Safety
    /// The caller must ensure that the NonNull pointer is valid and points to
    /// a properly initialized DLPack tensor.
    pub unsafe fn from_non_null(ptr: ffi::Dlpack) -> Self {
        Self(ptr)
    }

    /// Converts the SafeManagedTensor into a raw pointer.
    ///
    /// # Safety
    /// The caller takes ownership of the tensor and is responsible for
    /// calling the appropriate deleter function when done.
    pub unsafe fn into_raw(self) -> *mut ffi::ManagedTensor {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Converts the SafeManagedTensor into a NonNull DLPack pointer.
    ///
    /// The caller takes ownership of the tensor and is responsible for
    /// proper cleanup.
    pub fn into_non_null(self) -> ffi::Dlpack {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    /// Creates a new SafeManagedTensor from any type that implements TensorLike.
    ///
    /// This is the safe way to create a new tensor, as it handles all the
    /// memory management internally.
    ///
    /// # Type Parameters
    /// - T: The tensor type that implements TensorLike
    /// - L: The memory layout type that implements MemoryLayout
    pub fn new<T, L>(t: T) -> std::result::Result<Self, T::Error>
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        let ctx = ManagerContext::new(t);
        ctx.into_dlpack().map(Self)
    }
}

impl TensorView for SafeManagedTensor {
    /// Returns a reference to the underlying DLPack tensor.
    fn dl_tensor(&self) -> &ffi::Tensor {
        unsafe { &self.0.as_ref().dl_tensor }
    }
}

impl std::ops::Deref for SafeManagedTensor {
    type Target = [u8];

    /// Dereferences the tensor to a slice of bytes.
    /// This allows the tensor to be used as a byte slice.
    fn deref(&self) -> &Self::Target {
        self.as_slice_untyped()
    }
}

impl AsRef<SafeManagedTensor> for SafeManagedTensor {
    fn as_ref(&self) -> &Self {
        self
    }
}
