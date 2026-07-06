use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

pub trait OpaqueContext {
    fn into_raw(self) -> *mut c_void;

    /// Drops the raw context pointer and deallocates the underlying resources.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` was obtained from `into_raw` and has not been dropped yet.
    unsafe fn drop_raw(raw: *mut c_void);
}

impl<T: Sized> OpaqueContext for Box<T> {
    fn into_raw(self) -> *mut c_void {
        Box::into_raw(self) as *mut _
    }

    unsafe fn drop_raw(raw: *mut c_void) {
        if !raw.is_null() {
            unsafe {
                let _ = Box::from_raw(raw as *mut T);
            }
        }
    }
}

impl<T: Sized> OpaqueContext for Arc<T> {
    fn into_raw(self) -> *mut c_void {
        Arc::into_raw(self) as *mut c_void
    }
    unsafe fn drop_raw(raw: *mut c_void) {
        if !raw.is_null() {
            unsafe {
                let _ = Arc::from_raw(raw as *const T);
            }
        }
    }
}

impl<T: Sized> OpaqueContext for NonNull<T> {
    fn into_raw(self) -> *mut c_void {
        self.as_ptr() as *mut c_void
    }

    unsafe fn drop_raw(_raw: *mut c_void) {
        // 🔒 既然用户传的是裸指针，生命周期由用户自己用其他暗号控制，
        // 基础库在 deleter 触发时什么都不做（No-op），绝对不越权去 free 它的堆内存
    }
}
