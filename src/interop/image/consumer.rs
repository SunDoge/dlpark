use super::*;
use crate::{DlpackElement, Foreign, ManagedTensorBase, TryFromDlpack, tensor::is_compact_strides};
use image::{ImageBuffer, Pixel};
use snafu::ensure;
use std::{marker::PhantomData, ops::Deref};

impl<'a, P, M> TryFromDlpack<&'a Foreign<M>> for ImageBuffer<P, &'a [P::Subpixel]>
where
    P: Pixel,
    P::Subpixel: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: &'a Foreign<M>) -> Result<Self, Self::Error> {
        let tensor = unsafe { dlpack.tensor() };
        let layout = validated_hwc::<P>(tensor)?;

        let data_slice = unsafe {
            std::slice::from_raw_parts(layout.data_ptr as *const P::Subpixel, layout.num_elements)
        };

        ImageBuffer::from_raw(layout.width, layout.height, data_slice).ok_or(Error::BufferTooSmall)
    }
}

// ---------------------------------------------------------------------------
// Reverse owned: Local → ImageBuffer<P, DlpackContainer<M, T>>
//
// Zero-copy owned conversion. DlpackContainer holds the Local by value and
// exposes its pixel data as a slice through Deref. No data is copied.
// ---------------------------------------------------------------------------

/// An owned container that wraps a [`Foreign`] and exposes its raw data as a
/// `&[T]` slice, suitable for use as the backing store of an [`ImageBuffer`].
pub struct DlpackContainer<M: ManagedTensorBase, T> {
    dlpack: Foreign<M>,
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

impl<P, M> TryFromDlpack<Foreign<M>> for ImageBuffer<P, DlpackContainer<M, P::Subpixel>>
where
    P: Pixel,
    P::Subpixel: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: Foreign<M>) -> Result<Self, Self::Error> {
        let layout = {
            let tensor = unsafe { dlpack.tensor() };
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
