use std::ptr::NonNull;

use crate::ffi::{DataType, Device};

/// A trait for types that can serve directly as `manager_ctx` in a DLPack managed tensor.
///
/// Any type implementing this trait can be boxed and used as the context pointer in
/// `DLManagedTensorVersioned`. The deleter is automatically generated via `auto_deleter`.
///
/// # Safety
/// Implementors must ensure:
/// - `shape_ptr()` returns a valid non-null pointer to an array of `ndim()` i64 values
///   whose address remains stable for the lifetime of `self` on the heap.
/// - `strides_ptr()` if `Some`, points to an array of `ndim()` i64 values,
///   also stable on the heap.
/// - All returned pointers remain valid until `self` is dropped.
pub unsafe trait ManagerContext {
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn ndim(&self) -> i32;
    /// Pointer to the shape array. Must be non-null and stable on the heap.
    fn shape_ptr(&self) -> NonNull<i64>;
    /// Pointer to the strides array. `None` = row-major compact layout.
    fn strides_ptr(&self) -> Option<NonNull<i64>>;
    fn dtype(&self) -> DataType;
    fn device(&self) -> Device;
    fn byte_offset(&self) -> u64 {
        0
    }
}

/// Automatically generated C deleter for any `Box<C: ManagerContext>`.
/// Registered as the `deleter` field in `DLManagedTensorVersioned`.
pub(crate) unsafe extern "C" fn auto_deleter_versioned<C: ManagerContext>(
    ptr: *mut crate::ffi::ManagedTensorVersioned,
) {
    unsafe {
        let ctx = (*ptr).manager_ctx as *mut C;
        drop(Box::from_raw(ctx));
    }
}

pub(crate) unsafe extern "C" fn auto_deleter_legacy<C: ManagerContext>(
    ptr: *mut crate::ffi::ManagedTensor,
) {
    unsafe {
        let ctx = (*ptr).manager_ctx as *mut C;
        drop(Box::from_raw(ctx));
    }
}
