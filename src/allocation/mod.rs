//! Low-level allocation of managed tensors with writable metadata storage.

use snafu::Snafu;
use std::{alloc::Layout, ptr::NonNull};

pub mod dynamic;
pub mod fixed;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow { ndim: usize },

    #[snafu(display("managed tensor allocation layout overflows usize"))]
    LayoutOverflow,
}

macro_rules! initialized_accessors {
    () => {
        /// Returns the embedded descriptor for low-level initialization.
        pub fn tensor_mut(&mut self) -> &mut crate::ffi::DLTensor {
            unsafe { (&mut *self.managed.as_ptr()).tensor_mut() }
        }

        /// Sets the base data pointer stored in `DLTensor`.
        ///
        /// # Safety
        ///
        /// The context must keep the allocation behind `data` alive until the
        /// managed tensor deleter runs. Together with the other tensor fields,
        /// the pointer must describe initialized data valid under DLPack.
        pub unsafe fn set_data(&mut self, data: *mut std::ffi::c_void) {
            self.tensor_mut().data = data;
        }

        /// Sets the DLPack device descriptor.
        pub fn set_device(&mut self, device: crate::ffi::DLDevice) {
            self.tensor_mut().device = device;
        }

        /// Sets the DLPack element type descriptor.
        pub fn set_dtype(&mut self, dtype: crate::ffi::DLDataType) {
            self.tensor_mut().dtype = dtype;
        }

        /// Sets the byte offset from the base data pointer.
        pub fn set_byte_offset(&mut self, byte_offset: u64) {
            self.tensor_mut().byte_offset = byte_offset;
        }

        /// Sets flags unless doing so would newly assert `IS_COPIED`.
        pub fn set_flags(&mut self, flags: crate::DlpackFlags) -> Result<(), crate::tensor::Error> {
            if flags.newly_asserts_is_copied(self.managed.flags()) {
                return Err(crate::tensor::Error::CannotAssertIsCopied);
            }
            unsafe { (&mut *self.managed.as_ptr()).set_flags_unchecked(flags) };
            Ok(())
        }

        /// Sets flags verbatim, including `IS_COPIED`.
        ///
        /// # Safety
        ///
        /// If `flags` includes `IS_COPIED`, no other reference to the tensor
        /// data may exist.
        pub unsafe fn set_flags_unchecked(&mut self, flags: crate::DlpackFlags) {
            unsafe { (&mut *self.managed.as_ptr()).set_flags_unchecked(flags) };
        }

        /// Finishes initialization and returns a locally produced tensor.
        ///
        /// # Safety
        ///
        /// The completed descriptor must satisfy the DLPack contract. Its data
        /// and metadata pointers must remain valid until the tensor is dropped.
        pub unsafe fn finish(self) -> crate::Local<M> {
            unsafe { crate::Local::from_managed(self.managed) }
        }
    };
}

pub(super) use initialized_accessors;

fn allocate<M>(layout: Layout) -> NonNull<M> {
    let pointer = unsafe { std::alloc::alloc(layout) }.cast::<M>();
    NonNull::new(pointer).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

fn empty_tensor(ndim: i32) -> crate::ffi::DLTensor {
    crate::ffi::DLTensor::from_parts(std::ptr::null_mut(), std::ptr::null_mut(), ndim)
}
