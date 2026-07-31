//! Zero-copy HWC image interop for owned and borrowed `image` buffers.
//!
//! Boxed owned image buffers convert into [`crate::allocation::fixed::Initialized`] values with
//! `IS_COPIED` unset. Reverse conversions validate an HWC
//! compact layout before exposing the DLPack data as image storage.
//!
//! ```
//! use dlpark::{Managed, TryFromDlpack, allocation::fixed, ffi::DLManagedTensorVersioned};
//! use image::{ImageBuffer, Rgb};
//!
//! let image = ImageBuffer::<Rgb<u8>, _>::from_raw(1, 1, vec![10, 20, 30]).unwrap();
//! let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> = Box::new(image).try_into()?;
//! let dlpack: Managed<DLManagedTensorVersioned> =
//!     unsafe { initialized.finish() };
//! let image = unsafe { ImageBuffer::<Rgb<u8>, &[u8]>::try_from_dlpack(&dlpack, ())? };
//! assert_eq!(image.get_pixel(0, 0).0, [10, 20, 30]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use snafu::Snafu;

mod consumer;
mod producer;

pub use consumer::DlpackContainer;

#[derive(Debug, Snafu)]
/// Errors produced while validating a DLPack tensor as an image buffer.
pub enum Error {
    /// The tensor is not a three-dimensional HWC layout.
    #[snafu(display("tensor must have exactly 3 dimensions (H, W, C), got {ndim}"))]
    InvalidNdim {
        /// The reported dimension count.
        ndim: i32,
    },

    /// The final dimension differs from the pixel type's channel count.
    #[snafu(display("channel count mismatch: expected {expected}, got {actual}"))]
    ChannelMismatch {
        /// The expected channel count.
        expected: u8,
        /// The actual channel count.
        actual: i64,
    },

    /// At least one HWC dimension is zero or negative.
    #[snafu(display("all dimensions must be positive"))]
    NonPositiveDimension,

    /// An image dimension cannot be represented by the `image` crate.
    #[snafu(display("dimension {dimension} with value {value} does not fit in u32"))]
    DimensionOverflow {
        /// The name of the overflowing dimension.
        dimension: &'static str,
        /// The offending dimension value.
        value: i64,
    },

    /// The number of image elements overflowed `usize`.
    #[snafu(display("element count overflows usize"))]
    ElementCountOverflow,

    /// The tensor is not compact in HWC row-major order.
    #[snafu(display(
        "unsupported strides: expected [{expected_0}, {expected_1}, {expected_2}], \
         got [{actual_0}, {actual_1}, {actual_2}]"
    ))]
    UnsupportedStrides {
        /// Expected compact stride along dimension 0.
        expected_0: i64,
        /// Expected compact stride along dimension 1.
        expected_1: i64,
        /// Expected compact stride along dimension 2.
        expected_2: i64,
        /// Actual stride along dimension 0.
        actual_0: i64,
        /// Actual stride along dimension 1.
        actual_1: i64,
        /// Actual stride along dimension 2.
        actual_2: i64,
    },

    /// The validated storage was still too short for the requested image.
    #[snafu(display("failed to construct ImageBuffer: buffer size does not match dimensions"))]
    BufferTooSmall,

    /// The underlying DLPack tensor failed validation.
    #[snafu(transparent)]
    Tensor {
        /// The underlying tensor error.
        source: crate::tensor::Error,
    },
}

#[cfg(test)]
use crate::{
    DlpackElement, DlpackFlags, ManagedTensorBase, TryFromDlpack,
    allocation::fixed,
    ffi::DLDevice,
    metadata::{Copied, Fixed},
};
#[cfg(test)]
use image::ImageBuffer;
#[cfg(test)]
use std::ffi::c_void;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Managed,
        allocation::fixed::make_test_tensor,
        ffi::{DLManagedTensor, DLManagedTensorVersioned},
    };
    use image::Rgb;

    type LegacyDlpack = Managed<DLManagedTensor>;
    type VersionedDlpack = Managed<DLManagedTensorVersioned>;

    fn image_tensor<M: ManagedTensorBase>(
        img: ImageBuffer<Rgb<u8>, Vec<u8>>,
        flags: DlpackFlags,
    ) -> Managed<M> {
        let mut initialized: fixed::Initialized<M, 3> = Box::new(img).try_into().unwrap();
        initialized.set_flags_unchecked(flags);
        unsafe { initialized.finish() }
    }

    #[test]
    fn test_image_to_dlpack() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: LegacyDlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::empty());

        assert_eq!(dlpack.validate().unwrap().shape(), &[4, 4, 3]);
    }

    #[test]
    fn versioned_image_to_dlpack_is_zero_copy() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: VersionedDlpack =
            image_tensor::<DLManagedTensorVersioned>(img, DlpackFlags::empty());

        assert_eq!(dlpack.flags(), DlpackFlags::empty());
    }

    #[test]
    fn image_builder_allows_setting_read_only_safely() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: VersionedDlpack =
            image_tensor::<DLManagedTensorVersioned>(img, DlpackFlags::READ_ONLY);

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
    }

    #[test]
    fn versioned_image_to_dlpack_allows_unsafe_mutation() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let mut dlpack: VersionedDlpack =
            image_tensor::<DLManagedTensorVersioned>(img, DlpackFlags::empty());

        unsafe {
            dlpack
                .validate_mut()
                .unwrap()
                .cpu_slice_mut::<u8>()
                .unwrap()[0] = 42;
        }

        assert_eq!(unsafe { dlpack.tensor().cpu_slice::<u8>() }.unwrap()[0], 42);
    }

    #[test]
    fn test_borrowed_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![42u8; 48]).unwrap();
        let dlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::empty());

        let img2 = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack, ()) }.unwrap();
        assert_eq!(img2.width(), 4);
        assert_eq!(img2.height(), 4);
        assert_eq!(img2.as_raw()[0], 42);
    }

    #[test]
    fn test_owned_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![99u8; 48]).unwrap();
        let dlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::empty());

        let img2 =
            unsafe { ImageBuffer::<Rgb<u8>, DlpackContainer<_, u8>>::try_from_dlpack(dlpack, ()) }
                .unwrap();
        assert_eq!(img2.width(), 4);
        assert_eq!(img2.height(), 4);
        assert_eq!(img2.as_raw()[0], 99);
    }

    #[test]
    fn test_reverse_conversion_applies_byte_offset() {
        let data = Box::new(vec![0u8, 10, 20, 30]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let shape = [1, 1, 3];
        let strides = [3, 3, 1];
        let prepared = Fixed::new(Copied(shape), Copied(strides))
            .prepare::<DLManagedTensor>()
            .unwrap();
        let mut initialized = prepared.initialize(data);
        initialized.set_data(data_ptr);
        initialized
            .set_dtype(u8::DTYPE)
            .set_device(DLDevice::CPU)
            .set_byte_offset(1);
        let dlpack = unsafe { initialized.finish() };

        let img = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack, ()) }.unwrap();
        assert_eq!(img.as_raw(), &[10, 20, 30]);
    }

    #[test]
    fn test_reverse_conversion_rejects_null_data() {
        let data = Box::new(vec![0u8; 3]);
        let shape = [1, 1, 3];
        let strides = [3, 3, 1];
        let dlpack = make_test_tensor::<_, DLManagedTensor, 3>(
            data,
            std::ptr::null_mut(),
            u8::DTYPE,
            DLDevice::CPU,
            shape,
            strides,
            DlpackFlags::empty(),
        );

        let err = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack, ()) }.unwrap_err();
        assert!(matches!(
            err,
            Error::Tensor {
                source: crate::tensor::Error::NullData
            }
        ));
    }

    #[test]
    fn test_reverse_conversion_rejects_non_compact_strides() {
        let data = Box::new(vec![1u8, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        let shape = [1, 1, 3];
        let strides = [6, 3, 1];
        let dlpack = make_test_tensor::<_, DLManagedTensor, 3>(
            data,
            data_ptr,
            u8::DTYPE,
            DLDevice::CPU,
            shape,
            strides,
            DlpackFlags::empty(),
        );

        let err = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack, ()) }.unwrap_err();
        assert!(matches!(err, Error::UnsupportedStrides { .. }));
    }
}
