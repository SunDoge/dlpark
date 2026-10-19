use std::any::TypeId;

#[inline]
pub(super) unsafe fn try_copy<T>(src: &[T], dst: *mut i64) -> Result<(), usize>
where
    T: Copy + TryInto<i64> + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<i64>() {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().cast::<i64>(), dst, src.len()) };
        return Ok(());
    }

    for (axis, &value) in src.iter().enumerate() {
        let value = value.try_into().map_err(|_| axis)?;
        unsafe { dst.add(axis).write(value) };
    }
    Ok(())
}

/// Metadata values copied into the managed tensor allocation.
#[derive(Debug, Clone, Copy)]
pub struct Copied<T>(pub T);

/// Metadata values borrowed from caller-owned `i64` storage.
#[derive(Debug, Clone, Copy)]
pub struct Borrowed<T>(pub T);

/// Writes one dynamic metadata part, or retains an exact-i64 source pointer.
///
/// # Safety
///
/// For copied input, `dst` must have writable space for `src.len()` i64 values
/// and must not overlap the source. For borrowed input, the caller must ensure
/// the source outlives the resulting managed tensor without mutation.
#[inline]
pub(super) unsafe fn write_dynamic<T>(
    src: &[T],
    dst: *mut i64,
    borrowed: bool,
) -> Result<*mut i64, usize>
where
    T: Copy + TryInto<i64> + 'static,
{
    if borrowed {
        // Public borrowed constructors accept only i64. Check here as well so
        // an input trait implementation cannot reinterpret converted values.
        assert!(
            TypeId::of::<T>() == TypeId::of::<i64>(),
            "borrowed DLPack metadata must contain i64 values"
        );
        return Ok(src.as_ptr().cast::<i64>().cast_mut());
    }
    unsafe { try_copy(src, dst)? };
    Ok(dst)
}
