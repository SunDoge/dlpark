//! Zero-copy HWC image interop for owned and borrowed `image` buffers.
//!
//! Boxed owned image buffers convert into [`crate::Builder`] values with
//! [`crate::DlpackFlags::IS_COPIED`] set. Reverse conversions validate an HWC
//! compact layout before exposing the DLPack data as image storage.
//!
//! ```
//! use dlpark::{Builder, versioned};
//! use image::{ImageBuffer, Rgb};
//!
//! let image = ImageBuffer::<Rgb<u8>, _>::from_raw(1, 1, vec![10, 20, 30]).unwrap();
//! let dlpack: versioned::Dlpack = Builder::from(Box::new(image)).build();
//! let image = ImageBuffer::<Rgb<u8>, &[u8]>::try_from(&dlpack)?;
//! assert_eq!(image.get_pixel(0, 0).0, [10, 20, 30]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    Builder, DlpackElement, DlpackFlags, ManagedBox, ManagedTensorBase,
    ffi::DLDevice,
    metadata,
    tensor::{compact_strides_array, is_compact_strides},
};
use image::{ImageBuffer, Pixel};
use snafu::{Snafu, ensure};
use std::{marker::PhantomData, ops::Deref, os::raw::c_void};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Forward: ImageBuffer → Builder  (infallible, so From not TryFrom)
//
// Box<ImageBuffer> is used directly as the OpaqueContext, which avoids
// extracting the inner Vec and double-boxing it.
// ---------------------------------------------------------------------------

/// Converts a boxed, vector-backed image into a configurable DLPack builder.
///
/// The pixel allocation is reused without copying and exported as compact HWC
/// data. The builder starts with [`DlpackFlags::IS_COPIED`] because ownership
/// of the image has been transferred without retaining aliases.
impl<P> From<Box<ImageBuffer<P, Vec<P::Subpixel>>>>
    for Builder<Box<ImageBuffer<P, Vec<P::Subpixel>>>, metadata::CopiedArray<[i64; 3], [i64; 3], 3>>
where
    P: Pixel + Send,
    P::Subpixel: DlpackElement + Send,
{
    fn from(img: Box<ImageBuffer<P, Vec<P::Subpixel>>>) -> Self {
        let width = img.width();
        let height = img.height();
        let channels = P::CHANNEL_COUNT;
        let data_ptr = img.as_raw().as_ptr() as *mut c_void;
        let shape = [height as i64, width as i64, channels as i64];
        let strides = compact_strides_array(shape).expect("image shape must fit compact strides");

        let builder = Builder::new(img, metadata::CopiedArray::new(shape, strides));
        // SAFETY: the boxed image owns the initialized pixel allocation
        // addressed by data_ptr for the tensor's lifetime.
        let builder = unsafe { builder.data(data_ptr) }
            .dtype(P::Subpixel::DTYPE)
            .device(DLDevice::CPU);
        // SAFETY: `img` was moved in above and its `Vec`-backed pixel buffer
        // has no other live references.
        unsafe { builder.flags_unchecked(DlpackFlags::IS_COPIED) }
    }
}

// ---------------------------------------------------------------------------
// Reverse borrowed: &ManagedBox → ImageBuffer<P, &[T]>
//
// Zero-copy borrowed view. The ImageBuffer borrows from the ManagedBox
// and cannot outlive it.
// ---------------------------------------------------------------------------

impl<'a, P, M> TryFrom<&'a ManagedBox<M>> for ImageBuffer<P, &'a [P::Subpixel]>
where
    P: Pixel,
    P::Subpixel: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a ManagedBox<M>) -> Result<Self, Self::Error> {
        let tensor = dlpack.tensor();
        let layout = validated_hwc::<P>(tensor)?;

        let data_slice = unsafe {
            std::slice::from_raw_parts(layout.data_ptr as *const P::Subpixel, layout.num_elements)
        };

        ImageBuffer::from_raw(layout.width, layout.height, data_slice).ok_or(Error::BufferTooSmall)
    }
}

// ---------------------------------------------------------------------------
// Reverse owned: ManagedBox → ImageBuffer<P, DlpackContainer<M, T>>
//
// Zero-copy owned conversion. DlpackContainer holds the ManagedBox by value and
// exposes its pixel data as a slice through Deref. No data is copied.
// ---------------------------------------------------------------------------

/// An owned container that wraps a [`ManagedBox`] and exposes its raw data as a
/// `&[T]` slice, suitable for use as the backing store of an [`ImageBuffer`].
pub struct DlpackContainer<M: ManagedTensorBase, T> {
    dlpack: ManagedBox<M>,
    data_ptr: *const T,
    num_elements: usize,
    _marker: PhantomData<T>,
}

impl<M: ManagedTensorBase, T> Deref for DlpackContainer<M, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let _keep_alive = &self.dlpack;
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.num_elements) }
    }
}

impl<P, M> TryFrom<ManagedBox<M>> for ImageBuffer<P, DlpackContainer<M, P::Subpixel>>
where
    P: Pixel,
    P::Subpixel: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: ManagedBox<M>) -> Result<Self, Self::Error> {
        let layout = {
            let tensor = dlpack.tensor();
            validated_hwc::<P>(tensor)?
        };

        let container = DlpackContainer {
            dlpack,
            data_ptr: layout.data_ptr as *const P::Subpixel,
            num_elements: layout.num_elements,
            _marker: PhantomData,
        };
        ImageBuffer::from_raw(layout.width, layout.height, container).ok_or(Error::BufferTooSmall)
    }
}

// ---------------------------------------------------------------------------
// Shared validation helper
// ---------------------------------------------------------------------------

struct HwcLayout {
    height: u32,
    width: u32,
    data_ptr: *const (),
    num_elements: usize,
}

fn validated_hwc<P>(tensor: &crate::ffi::DLTensor) -> Result<HwcLayout, Error>
where
    P: Pixel,
    P::Subpixel: DlpackElement,
{
    ensure!(tensor.ndim == 3, InvalidNdimSnafu { ndim: tensor.ndim });

    let shape = unsafe { tensor.shape()? };
    let [height, width, channels] = [shape[0], shape[1], shape[2]];

    ensure!(
        height > 0 && width > 0 && channels > 0,
        NonPositiveDimensionSnafu
    );

    let height = u32::try_from(height).map_err(|_| Error::DimensionOverflow {
        dimension: "height",
        value: height,
    })?;
    let width = u32::try_from(width).map_err(|_| Error::DimensionOverflow {
        dimension: "width",
        value: width,
    })?;
    let _channels_u32 = u32::try_from(channels).map_err(|_| Error::DimensionOverflow {
        dimension: "channels",
        value: channels,
    })?;

    ensure!(
        channels == P::CHANNEL_COUNT as i64,
        ChannelMismatchSnafu {
            expected: P::CHANNEL_COUNT,
            actual: channels
        }
    );

    let expected_strides = [
        i64::from(width)
            .checked_mul(P::CHANNEL_COUNT as i64)
            .ok_or(Error::ElementCountOverflow)?,
        P::CHANNEL_COUNT as i64,
        1,
    ];
    if let Some(strides) = unsafe { tensor.strides()? } {
        let actual = [strides[0], strides[1], strides[2]];
        ensure!(
            is_compact_strides(shape, Some(strides))?,
            UnsupportedStridesSnafu {
                expected_0: expected_strides[0],
                expected_1: expected_strides[1],
                expected_2: expected_strides[2],
                actual_0: actual[0],
                actual_1: actual[1],
                actual_2: actual[2],
            }
        );
    }

    let data = unsafe { tensor.cpu_slice::<P::Subpixel>()? };

    Ok(HwcLayout {
        height,
        width,
        data_ptr: data.as_ptr().cast(),
        num_elements: data.len(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ffi::DLManagedTensor, legacy, versioned};
    use image::Rgb;

    #[test]
    fn test_image_to_dlpack() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: legacy::Dlpack = Builder::from(Box::new(img)).build();

        assert_eq!(dlpack.shape().unwrap(), &[4, 4, 3]);
    }

    #[test]
    fn versioned_image_to_dlpack_sets_is_copied() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: versioned::Dlpack = Builder::from(Box::new(img)).build();

        assert_eq!(dlpack.flags(), DlpackFlags::IS_COPIED);
    }

    #[test]
    fn image_builder_allows_setting_read_only_safely() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let dlpack: versioned::Dlpack = Builder::from(Box::new(img))
            .insert_flags(DlpackFlags::READ_ONLY)
            .unwrap()
            .build();

        assert_eq!(
            dlpack.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn versioned_image_to_dlpack_allows_unsafe_mutation() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![0u8; 48]).unwrap();
        let mut dlpack: versioned::Dlpack = Builder::from(Box::new(img)).build();

        unsafe {
            dlpack.cpu_slice_mut_unchecked::<u8>().unwrap()[0] = 42;
        }

        assert_eq!(unsafe { dlpack.tensor().cpu_slice::<u8>() }.unwrap()[0], 42);
    }

    #[test]
    fn test_borrowed_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![42u8; 48]).unwrap();
        let dlpack: legacy::Dlpack = Builder::from(Box::new(img)).build();

        let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&dlpack).unwrap();
        assert_eq!(img2.width(), 4);
        assert_eq!(img2.height(), 4);
        assert_eq!(img2.as_raw()[0], 42);
    }

    #[test]
    fn test_owned_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(4, 4, vec![99u8; 48]).unwrap();
        let dlpack: legacy::Dlpack = Builder::from(Box::new(img)).build();

        let img2 = ImageBuffer::<Rgb<u8>, DlpackContainer<_, u8>>::try_from(dlpack).unwrap();
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
        let dlpack = unsafe {
            Builder::new(data, metadata::CopiedArray::new(&shape, &strides)).data(data_ptr)
        }
        .byte_offset(1)
        .dtype(u8::DTYPE)
        .build::<DLManagedTensor>();

        let img = ImageBuffer::<Rgb<u8>, _>::try_from(&dlpack).unwrap();
        assert_eq!(img.as_raw(), &[10, 20, 30]);
    }

    #[test]
    fn test_reverse_conversion_rejects_null_data() {
        let data = Box::new(vec![0u8; 3]);
        let shape = [1, 1, 3];
        let strides = [3, 3, 1];
        let dlpack = Builder::new(data, metadata::CopiedArray::new(&shape, &strides))
            .dtype(u8::DTYPE)
            .build::<DLManagedTensor>();

        let err = ImageBuffer::<Rgb<u8>, _>::try_from(&dlpack).unwrap_err();
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
        let dlpack = unsafe {
            Builder::new(data, metadata::CopiedArray::new(&shape, &strides)).data(data_ptr)
        }
        .dtype(u8::DTYPE)
        .build::<DLManagedTensor>();

        let err = ImageBuffer::<Rgb<u8>, _>::try_from(&dlpack).unwrap_err();
        assert!(matches!(err, Error::UnsupportedStrides { .. }));
    }
}
