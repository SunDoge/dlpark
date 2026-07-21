use std::ffi::c_void;
use std::sync::Arc;

/// Owns or tracks the opaque context stored in a DLPack managed tensor.
///
/// DLPack consumers may invoke the managed tensor deleter on a different
/// thread from the one that created the context.
///
/// # Safety
///
/// Implementations must ensure that [`OpaqueContext::drop_raw`] may be called
/// on any thread and does not depend on thread-local state.
pub unsafe trait OpaqueContext {
    /// Transfers the context into the opaque pointer stored in `manager_ctx`.
    ///
    /// The pointer must be recoverable by [`OpaqueContext::drop_raw`].
    fn into_raw(self) -> *mut c_void;

    /// Drops the raw context pointer and deallocates the underlying resources.
    ///
    /// The context must carry any allocation metadata needed to destroy itself.
    /// DLPack tensor fields are public mutable ABI state and are intentionally
    /// not provided to this method.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` was obtained from `into_raw` and has
    /// not been dropped yet. Implementations must not unwind.
    unsafe fn drop_raw(raw: *mut c_void);
}

unsafe impl<T: Sized + Send> OpaqueContext for Box<T> {
    #[inline]
    fn into_raw(self) -> *mut c_void {
        Box::into_raw(self) as *mut _
    }

    #[inline]
    unsafe fn drop_raw(raw: *mut c_void) {
        if !raw.is_null() {
            unsafe {
                let _ = Box::from_raw(raw as *mut T);
            }
        }
    }
}

unsafe impl<T: Sized + Send + Sync> OpaqueContext for Arc<T> {
    #[inline]
    fn into_raw(self) -> *mut c_void {
        Arc::into_raw(self) as *mut c_void
    }
    #[inline]
    unsafe fn drop_raw(raw: *mut c_void) {
        if !raw.is_null() {
            unsafe {
                let _ = Arc::from_raw(raw as *const T);
            }
        }
    }
}
