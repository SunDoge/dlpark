//! Runtime-sized extra metadata allocation.

use super::{Error, allocate, empty_tensor, initialized_accessors};
use crate::{ManagedBox, ManagedTensorBase, OpaqueContext};
use std::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

/// An uninitialized managed tensor allocation with dynamic extra capacity.
pub struct Allocation<M> {
    managed: NonNull<M>,
    extra: NonNull<i64>,
    extra_len: usize,
    layout: Layout,
}

impl<M: ManagedTensorBase> Allocation<M> {
    /// Allocates storage for `M` and `extra` additional `i64` values.
    pub fn allocate(extra: usize) -> Result<Self, Error> {
        let parts = allocation_parts::<M>(extra)?;
        let managed = allocate::<M>(parts.layout);

        unsafe {
            let base = managed.as_ptr().cast::<u8>();
            base.add(parts.header)
                .cast::<usize>()
                .write(parts.layout.size());
            let extra_ptr = base.add(parts.extra).cast::<i64>();
            extra_ptr.write_bytes(0, extra);
            Ok(Self {
                managed,
                extra: NonNull::new_unchecked(extra_ptr),
                extra_len: extra,
                layout: parts.layout,
            })
        }
    }

    /// Returns the zero-initialized extra metadata buffer.
    pub fn extra_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.extra.as_ptr(), self.extra_len) }
    }

    /// Initializes the managed tensor and installs its context and deleter.
    pub fn initialize<C: OpaqueContext>(
        self,
        ctx: C,
        ndim: usize,
    ) -> Result<Initialized<M>, Error> {
        let ndim = i32::try_from(ndim).map_err(|_| Error::NdimOverflow { ndim })?;
        let this = ManuallyDrop::new(self);
        unsafe {
            this.managed.as_ptr().write(M::from_parts(
                empty_tensor(ndim),
                ctx.into_raw(),
                Some(drop_allocation::<C, M>),
            ));
            Ok(Initialized {
                managed: ManagedBox::new_unchecked(this.managed.as_ptr()),
                extra: this.extra,
                extra_len: this.extra_len,
            })
        }
    }
}

impl<M> Drop for Allocation<M> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.managed.as_ptr().cast(), self.layout) };
    }
}

/// An initialized dynamically sized allocation.
pub struct Initialized<M: ManagedTensorBase> {
    pub(super) managed: ManagedBox<M>,
    extra: NonNull<i64>,
    extra_len: usize,
}

impl<M: ManagedTensorBase> Initialized<M> {
    /// Returns the extra metadata buffer.
    pub fn extra_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.extra.as_ptr(), self.extra_len) }
    }

    initialized_accessors!();
}

struct Parts {
    layout: Layout,
    header: usize,
    extra: usize,
}

fn allocation_parts<M>(extra: usize) -> Result<Parts, Error> {
    let extra_layout = Layout::array::<i64>(extra).map_err(|_| Error::LayoutOverflow)?;
    let (layout, header) = Layout::new::<M>()
        .extend(Layout::new::<usize>())
        .map_err(|_| Error::LayoutOverflow)?;
    let (layout, extra) = layout
        .extend(extra_layout)
        .map_err(|_| Error::LayoutOverflow)?;
    Ok(Parts {
        layout: layout.pad_to_align(),
        header,
        extra,
    })
}

unsafe extern "C" fn drop_allocation<C, M>(managed: *mut M)
where
    C: OpaqueContext,
    M: ManagedTensorBase,
{
    if managed.is_null() {
        return;
    }
    unsafe {
        let header = Layout::new::<M>().extend(Layout::new::<usize>()).unwrap().1;
        let size = managed.cast::<u8>().add(header).cast::<usize>().read();
        let align = align_of::<M>()
            .max(align_of::<usize>())
            .max(align_of::<i64>());
        let layout = Layout::from_size_align_unchecked(size, align);
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DlpackFlags, ffi::DLManagedTensorVersioned};
    use crate::{ManagedTensorBase, ffi::DLManagedTensor};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn extra_buffer_can_hold_shape_and_strides() {
        let mut allocation = Allocation::<DLManagedTensor>::allocate(4).unwrap();
        allocation.extra_mut().copy_from_slice(&[2, 3, 3, 1]);
        let mut initialized = allocation.initialize(Box::new(()), 2).unwrap();
        let extra = initialized.extra_mut().as_mut_ptr();
        initialized.tensor_mut().shape = extra;
        initialized.tensor_mut().strides = unsafe { extra.add(2) };
        let tensor = unsafe { initialized.finish() };
        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }

    #[test]
    fn borrowed_shape_and_copied_strides_are_independent() {
        let shape = [2_i64, 3];
        let mut allocation = Allocation::<DLManagedTensor>::allocate(2).unwrap();
        allocation.extra_mut().copy_from_slice(&[3, 1]);
        let mut initialized = allocation.initialize(Box::new(()), 2).unwrap();
        initialized.tensor_mut().shape = shape.as_ptr().cast_mut();
        initialized.tensor_mut().strides = initialized.extra_mut().as_mut_ptr();
        let tensor = unsafe { initialized.finish() };
        assert_eq!(tensor.shape().unwrap(), &shape);
    }

    #[test]
    fn initialization_paths_release_ownership() {
        drop(Allocation::<DLManagedTensor>::allocate(3).unwrap());

        let drops = Arc::new(AtomicUsize::new(0));
        let context = Box::new(DropCounter(Arc::clone(&drops)));
        let allocation = Allocation::<DLManagedTensor>::allocate(0).unwrap();
        drop(allocation.initialize(context, 0).unwrap());
        assert_eq!(drops.load(Ordering::Relaxed), 1);

        let context = Box::new(DropCounter(Arc::clone(&drops)));
        let allocation = Allocation::<DLManagedTensor>::allocate(0).unwrap();
        assert!(allocation.initialize(context, usize::MAX).is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn deleter_ignores_public_ndim() {
        let allocation = Allocation::<DLManagedTensor>::allocate(2).unwrap();
        let initialized = allocation.initialize(Box::new(()), 1).unwrap();
        let tensor = unsafe { initialized.finish() };
        unsafe { (*tensor.as_ptr()).tensor_mut().ndim = i32::MAX };
        drop(tensor);
    }

    #[test]
    fn versioned_flags_are_configurable() {
        let allocation = Allocation::<DLManagedTensorVersioned>::allocate(0).unwrap();
        let mut initialized = allocation.initialize(Box::new(()), 0).unwrap();
        initialized.set_flags(DlpackFlags::READ_ONLY).unwrap();
        assert_eq!(
            unsafe { initialized.finish() }.flags(),
            DlpackFlags::READ_ONLY
        );
    }
}
