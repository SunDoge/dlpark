//! Zero-copy HWC image interop for owned and borrowed `image` buffers.
//!
//! Boxed owned image buffers convert into [`crate::allocation::fixed::Initialized`] values with
//! [`crate::DlpackFlags::IS_COPIED`] set. Reverse conversions validate an HWC
//! compact layout before exposing the DLPack data as image storage.
//!
//! ```
//! use dlpark::{Foreign, TryFromDlpack, allocation::fixed, ffi::DLManagedTensorVersioned};
//! use image::{ImageBuffer, Rgb};
//!
//! let image = ImageBuffer::<Rgb<u8>, _>::from_raw(1, 1, vec![10, 20, 30]).unwrap();
//! let initialized: fixed::Initialized<DLManagedTensorVersioned, 3> = Box::new(image).try_into()?;
//! let dlpack: Foreign<DLManagedTensorVersioned> =
//!     unsafe { initialized.finish() }.into_foreign();
//! let image = unsafe { ImageBuffer::<Rgb<u8>, &[u8]>::try_from_dlpack(&dlpack)? };
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
    InvalidNdim { ndim: i32 },

    /// The final dimension differs from the pixel type's channel count.
    #[snafu(display("channel count mismatch: expected {expected}, got {actual}"))]
    ChannelMismatch { expected: u8, actual: i64 },

    /// At least one HWC dimension is zero or negative.
    #[snafu(display("all dimensions must be positive"))]
    NonPositiveDimension,

    /// An image dimension cannot be represented by the `image` crate.
    #[snafu(display("dimension {dimension} with value {value} does not fit in u32"))]
    DimensionOverflow { dimension: &'static str, value: i64 },

    /// The number of image elements overflowed `usize`.
    #[snafu(display("element count overflows usize"))]
    ElementCountOverflow,

    /// The tensor is not compact in HWC row-major order.
    #[snafu(display(
        "unsupported strides: expected [{expected_0}, {expected_1}, {expected_2}], \
         got [{actual_0}, {actual_1}, {actual_2}]"
    ))]
    UnsupportedStrides {
        expected_0: i64,
        expected_1: i64,
        expected_2: i64,
        actual_0: i64,
        actual_1: i64,
        actual_2: i64,
    },

    /// The validated storage was still too short for the requested image.
    #[snafu(display("failed to construct ImageBuffer: buffer size does not match dimensions"))]
    BufferTooSmall,

    /// The underlying DLPack tensor failed validation.
    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },
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
        Local,
        allocation::fixed::make_test_tensor,
        ffi::{DLManagedTensor, DLManagedTensorVersioned},
    };
    use image::Rgb;

    type LegacyDlpack = Local<DLManagedTensor>;
    type VersionedDlpack = Local<DLManagedTensorVersioned>;

    fn image_tensor<M: ManagedTensorBase>(
        img: ImageBuffer<Rgb<u8>, Vec<u8>>,
        flags: DlpackFlags,
    ) -> Local<M> {
        let mut initialized: fixed::Initialized<M, 3> = Box::new(img).try_into().unwrap();
        initialized.set_flags_unchecked(flags);
        unsafe { initialized.finish() }
    }

    #[test]
    fn test_image_to_dlpack() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: LegacyDlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::IS_COPIED);

        assert_eq!(dlpack.shape().unwrap(), &[4, 4, 3]);
    }

    #[test]
    fn versioned_image_to_dlpack_sets_is_copied() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: VersionedDlpack =
            image_tensor::<DLManagedTensorVersioned>(img, DlpackFlags::IS_COPIED);

        assert_eq!(dlpack.flags(), DlpackFlags::IS_COPIED);
    }

    #[test]
    fn image_builder_allows_setting_read_only_safely() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: VersionedDlpack = image_tensor::<DLManagedTensorVersioned>(
            img,
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY,
        );

        assert_eq!(
            dlpack.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn versioned_image_to_dlpack_allows_unsafe_mutation() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let mut dlpack: VersionedDlpack =
            image_tensor::<DLManagedTensorVersioned>(img, DlpackFlags::IS_COPIED);

        unsafe {
            dlpack.cpu_slice_mut_unchecked::<u8>().unwrap()[0] = 42;
        }

        assert_eq!(unsafe { dlpack.tensor().cpu_slice::<u8>() }.unwrap()[0], 42);
    }

    #[test]
    fn test_borrowed_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![42u8; 48]).unwrap();
        let dlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::IS_COPIED).into_foreign();

        let img2 = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack) }.unwrap();
        assert_eq!(img2.width(), 4);
        assert_eq!(img2.height(), 4);
        assert_eq!(img2.as_raw()[0], 42);
    }

    #[test]
    fn test_owned_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![99u8; 48]).unwrap();
        let dlpack = image_tensor::<DLManagedTensor>(img, DlpackFlags::IS_COPIED).into_foreign();

        let img2 =
            unsafe { ImageBuffer::<Rgb<u8>, DlpackContainer<_, u8>>::try_from_dlpack(dlpack) }
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
        let dlpack = unsafe { initialized.finish() }.into_foreign();

        let img = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack) }.unwrap();
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
        )
        .into_foreign();

        let err = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack) }.unwrap_err();
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
        )
        .into_foreign();

        let err = unsafe { ImageBuffer::<Rgb<u8>, _>::try_from_dlpack(&dlpack) }.unwrap_err();
        assert!(matches!(err, Error::UnsupportedStrides { .. }));
    }
}
