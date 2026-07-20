use std::{alloc::Layout, ptr::NonNull};

#[inline]
pub(super) fn copied_array_layout<M, const N: usize>() -> (Layout, usize) {
    let metadata = Layout::array::<i64>(
        N.checked_mul(2)
            .expect("DLPack metadata length should fit in usize"),
    )
    .expect("DLPack metadata allocation layout should fit in memory");

    extend_metadata_layout::<M>(metadata)
}

#[inline]
pub(super) fn copied_slice_layout<M>(ndim: usize) -> (Layout, usize, usize) {
    let allocation_ndim = Layout::new::<usize>();
    let metadata = Layout::array::<i64>(
        ndim.checked_mul(2)
            .expect("DLPack metadata length should fit in usize"),
    )
    .expect("DLPack metadata allocation layout should fit in memory");

    let (layout, allocation_ndim_offset) = Layout::new::<M>()
        .extend(allocation_ndim)
        .expect("DLPack storage allocation layout should fit in memory");
    let (layout, metadata_offset) = layout
        .extend(metadata)
        .expect("DLPack storage allocation layout should fit in memory");

    (
        layout.pad_to_align(),
        allocation_ndim_offset,
        metadata_offset,
    )
}

#[inline]
fn extend_metadata_layout<M>(metadata: Layout) -> (Layout, usize) {
    Layout::new::<M>()
        .extend(metadata)
        .map(|(layout, offset)| (layout.pad_to_align(), offset))
        .expect("DLPack storage allocation layout should fit in memory")
}

#[inline]
unsafe fn allocate<M>(layout: Layout) -> NonNull<M> {
    let ptr = unsafe { std::alloc::alloc(layout) }.cast::<M>();
    match NonNull::new(ptr) {
        Some(ptr) => ptr,
        None => std::alloc::handle_alloc_error(layout),
    }
}

#[inline]
pub(super) unsafe fn allocate_copied_array<M, const N: usize>() -> (NonNull<M>, *mut i64, *mut i64)
{
    let (layout, metadata_offset) = copied_array_layout::<M, N>();
    let managed_tensor: NonNull<M> = unsafe { allocate(layout) };
    let shape = unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(metadata_offset)
            .cast::<i64>()
    };
    let strides = unsafe { shape.add(N) };
    (managed_tensor, shape, strides)
}

#[inline]
pub(super) unsafe fn allocate_copied_slice<M>(ndim: usize) -> (NonNull<M>, *mut i64, *mut i64) {
    let (layout, allocation_ndim_offset, metadata_offset) = copied_slice_layout::<M>(ndim);
    let managed_tensor: NonNull<M> = unsafe { allocate(layout) };
    unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(allocation_ndim_offset)
            .cast::<usize>()
            .write(ndim);
    }
    let shape = unsafe {
        managed_tensor
            .as_ptr()
            .cast::<u8>()
            .add(metadata_offset)
            .cast::<i64>()
    };
    let strides = unsafe { shape.add(ndim) };
    (managed_tensor, shape, strides)
}

#[inline]
pub(super) unsafe fn allocate_borrowed<M>() -> NonNull<M> {
    unsafe { allocate(Layout::new::<M>()) }
}
