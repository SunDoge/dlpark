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
