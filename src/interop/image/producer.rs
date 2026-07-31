use crate::{
    DlpackElement, ManagedTensorBase,
    allocation::fixed,
    ffi::DLDevice,
    metadata::{Copied, Fixed},
    tensor::compact_strides_array,
};
use image::{ImageBuffer, Pixel};
use std::os::raw::c_void;

/// Converts a boxed, vector-backed image into a configurable DLPack builder.
///
/// The pixel allocation is reused without copying and exported as compact HWC
/// data. Moving the image transfers ownership without copying its pixel data.
impl<P, M> TryFrom<Box<ImageBuffer<P, Vec<P::Subpixel>>>> for fixed::Initialized<M, 3>
where
    P: Pixel + Send,
    P::Subpixel: DlpackElement + Send,
    M: ManagedTensorBase,
{
    type Error = crate::metadata::Error;

    fn try_from(img: Box<ImageBuffer<P, Vec<P::Subpixel>>>) -> Result<Self, Self::Error> {
        let width = img.width();
        let height = img.height();
        let channels = P::CHANNEL_COUNT;
        let data_ptr = img.as_raw().as_ptr() as *mut c_void;
        let shape = [height as i64, width as i64, channels as i64];
        let strides = compact_strides_array(shape).expect("image shape must fit compact strides");

        let prepared = Fixed::new(Copied(shape), Copied(strides)).prepare::<M>()?;
        let mut initialized = prepared.initialize(img);
        initialized
            .set_data(data_ptr)
            .set_dtype(P::Subpixel::DTYPE)
            .set_device(DLDevice::CPU);
        Ok(initialized)
    }
}

// ---------------------------------------------------------------------------
// Reverse borrowed: &Managed → ImageBuffer<P, &[T]>
//
// Zero-copy borrowed view. The ImageBuffer borrows from the Managed
// and cannot outlive it.
// ---------------------------------------------------------------------------
