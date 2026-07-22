use std::any::TypeId;

#[inline(always)]
unsafe fn copy_metadata<T>(src: *const T, dst: *mut i64, len: usize) -> bool
where
    T: Copy + TryInto<i64> + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<i64>() {
        unsafe { std::ptr::copy_nonoverlapping(src.cast::<i64>(), dst, len) };
        return true;
    }

    for index in 0..len {
        let value = unsafe { src.add(index).read() };
        let Ok(value) = value.try_into() else {
            return false;
        };
        unsafe { dst.add(index).write(value) };
    }
    true
}

#[unsafe(no_mangle)]
/// Copies `len` `i64` values for release-assembly inspection.
///
/// # Safety
///
/// `src` and `dst` must be valid for `len` elements and must not overlap.
pub unsafe extern "C" fn copy_i64_metadata(src: *const i64, dst: *mut i64, len: usize) -> bool {
    unsafe { copy_metadata(src, dst, len) }
}

#[unsafe(no_mangle)]
/// Converts `len` `u64` values for release-assembly inspection.
///
/// # Safety
///
/// `src` and `dst` must be valid for `len` elements and must not overlap.
pub unsafe extern "C" fn copy_u64_metadata(src: *const u64, dst: *mut i64, len: usize) -> bool {
    unsafe { copy_metadata(src, dst, len) }
}

fn main() {}
