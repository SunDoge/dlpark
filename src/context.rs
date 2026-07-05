use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

pub trait OpaqueContext {
    type Target;

    fn into_raw(self) -> *mut c_void;

    /// Drops the raw context pointer and deallocates the underlying resources.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` was obtained from `into_raw` and has not been dropped yet.
    unsafe fn drop_raw(raw: *mut c_void);

    /// Borrow the target from the raw context pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` is a valid pointer obtained from `into_raw` and points to a valid alive target.
    unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target;
}

impl<T> OpaqueContext for Box<T> {
    type Target = T;

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

    unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
        unsafe { &*(raw as *mut T) }
    }
}

impl<T> OpaqueContext for Arc<T> {
    type Target = T;

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
    unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
        // 完美的临时解引用视角
        unsafe { &*(raw as *const T) }
    }
}

// impl OpaqueContext for *mut c_void {
//     type Target = c_void;

//     fn into_raw(self) -> *mut c_void {
//         self // 原样返回
//     }
//     unsafe fn drop_raw(_raw: *mut c_void) {
//         // 用户自己控制生命周期，基础库完全不介入释放（No-op）
//     }
//     unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
//         unsafe { &*raw }
//     }
// }

impl<T> OpaqueContext for NonNull<T> {
    type Target = T;

    fn into_raw(self) -> *mut c_void {
        self.as_ptr() as *mut c_void
    }

    unsafe fn drop_raw(_raw: *mut c_void) {
        // 🔒 既然用户传的是裸指针，生命周期由用户自己用其他暗号控制，
        // 基础库在 deleter 触发时什么都不做（No-op），绝对不越权去 free 它的堆内存
    }

    unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
        // 完美的临时解引用
        unsafe { &*(raw as *mut T) }
    }
}
