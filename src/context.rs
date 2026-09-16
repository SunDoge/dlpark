use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// A type-erased, exactly-once allocation release callback.
///
/// This lets a container retain an imported allocation without retaining its
/// DLPack header representation. Dropping the deleter invokes its callback.
pub struct AllocationDeleter {
    callback: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl AllocationDeleter {
    /// Creates an allocation deleter from an arbitrary release callback.
    ///
    /// # Safety
    ///
    /// The callback must be safe to invoke exactly once from any thread and
    /// must not unwind. Any resources it references must remain valid until it
    /// runs.
    pub unsafe fn new(callback: impl FnOnce() + Send + 'static) -> Self {
        Self {
            callback: Mutex::new(Some(Box::new(callback))),
        }
    }
}

impl Drop for AllocationDeleter {
    fn drop(&mut self) {
        let callback = self
            .callback
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(callback) = callback {
            callback();
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn allocation_deleter_runs_once_on_drop() {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let deleter = unsafe {
            AllocationDeleter::new(move || {
                callback_calls.fetch_add(1, Ordering::Relaxed);
            })
        };

        drop(deleter);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
