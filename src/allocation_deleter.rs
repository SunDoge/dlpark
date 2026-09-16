use std::ffi::c_void;

/// A type-erased, exactly-once allocation release callback.
///
/// This lets a container retain an imported allocation without exposing the
/// concrete owner or DLPack header ABI in its own type. Dropping the deleter
/// invokes its callback.
pub struct AllocationDeleter {
    context: *mut c_void,
    callback: unsafe fn(*mut c_void),
}

// SAFETY: construction requires the callback and its context to be transferable
// to any thread. Shared references expose neither value, and the callback is
// invoked only by `Drop`, which has exclusive access to the deleter.
unsafe impl Send for AllocationDeleter {}
unsafe impl Sync for AllocationDeleter {}

impl AllocationDeleter {
    /// Creates an allocation deleter from an arbitrary release callback.
    ///
    pub fn new<F>(callback: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        unsafe fn invoke<F: FnOnce() + Send>(context: *mut c_void) {
            let callback = unsafe { Box::from_raw(context.cast::<F>()) };
            callback();
        }

        let context = Box::into_raw(Box::new(callback)).cast();
        // SAFETY: `context` owns the boxed callback and `invoke` reconstructs
        // that box exactly once when this deleter is dropped.
        unsafe { Self::from_raw_parts(context, invoke::<F>) }
    }

    /// Creates a deleter from a type-erased context and release function.
    ///
    /// This avoids allocating another closure when ownership is already
    /// represented by a raw handle, such as a DLPack managed tensor pointer.
    ///
    /// # Safety
    ///
    /// `context` must remain valid until `callback` consumes or releases it.
    /// The callback must be safe to invoke exactly once from any thread and
    /// must not unwind.
    pub unsafe fn from_raw_parts(context: *mut c_void, callback: unsafe fn(*mut c_void)) -> Self {
        Self { context, callback }
    }
}

impl Drop for AllocationDeleter {
    fn drop(&mut self) {
        unsafe { (self.callback)(self.context) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn runs_once_on_drop() {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let deleter = AllocationDeleter::new(move || {
            callback_calls.fetch_add(1, Ordering::Relaxed);
        });

        drop(deleter);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<AllocationDeleter>();
    }
}
