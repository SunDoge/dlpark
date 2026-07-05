use crate::{
    builder::DlpackBuilder,
    dlpack::Dlpack,
    ffi::{DLDataType, DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned},
    managed_tensor::ManagedTensor,
};
use image::{ImageBuffer, Pixel};
use std::os::raw::c_void;

pub trait ImageSubpixel {
    fn data_type() -> DLDataType;
}

impl ImageSubpixel for u8 {
    fn data_type() -> DLDataType {
        DLDataType {
            code: crate::ffi::DLDataTypeCode::UINT,
            bits: 8,
            lanes: 1,
        }
    }
}

impl ImageSubpixel for u16 {
    fn data_type() -> DLDataType {
        DLDataType {
            code: crate::ffi::DLDataTypeCode::UINT,
            bits: 16,
            lanes: 1,
        }
    }
}

impl ImageSubpixel for f32 {
    fn data_type() -> DLDataType {
        DLDataType {
            code: crate::ffi::DLDataTypeCode::FLOAT,
            bits: 32,
            lanes: 1,
        }
    }
}

pub struct ImageContext<T>(Vec<T>);

impl<T> crate::context::OpaqueContext for ImageContext<T> {
    type Target = Vec<T>;

    fn into_raw(self) -> *mut c_void {
        let boxed = Box::new(self);
        Box::into_raw(boxed) as *mut c_void
    }

    unsafe fn drop_raw(raw: *mut c_void) {
        if !raw.is_null() {
            let _ = unsafe { Box::from_raw(raw as *mut ImageContext<T>) };
        }
    }

    unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
        let ctx = unsafe { &*(raw as *mut ImageContext<T>) };
        &ctx.0
    }
}

impl<P> TryFrom<ImageBuffer<P, Vec<P::Subpixel>>> for Dlpack<DLManagedTensor>
where
    P: Pixel,
    P::Subpixel: ImageSubpixel,
{
    type Error = &'static str;

    fn try_from(img: ImageBuffer<P, Vec<P::Subpixel>>) -> Result<Self, Self::Error> {
        let width = img.width();
        let height = img.height();
        let channels = P::CHANNEL_COUNT;
        let data = img.into_raw();
        let data_ptr = data.as_ptr() as *mut c_void;

        let ctx = ImageContext(data);

        let shape = [height as i64, width as i64, channels as i64];
        let strides = [
            (width * channels as u32) as i64,
            channels as i64,
            1,
        ];

        let builder = DlpackBuilder::<DLManagedTensor, 3>::with_slice_layout(ctx, &shape, &strides);
        let builder = builder
            .data(data_ptr)
            .dtype(P::Subpixel::data_type())
            .device(DLDevice {
                device_type: DLDeviceType::CPU,
                device_id: 0,
            });

        Ok(builder.build())
    }
}

impl<P> TryFrom<ImageBuffer<P, Vec<P::Subpixel>>> for Dlpack<DLManagedTensorVersioned>
where
    P: Pixel,
    P::Subpixel: ImageSubpixel,
{
    type Error = &'static str;

    fn try_from(img: ImageBuffer<P, Vec<P::Subpixel>>) -> Result<Self, Self::Error> {
        let width = img.width();
        let height = img.height();
        let channels = P::CHANNEL_COUNT;
        let data = img.into_raw();
        let data_ptr = data.as_ptr() as *mut c_void;

        let ctx = ImageContext(data);

        let shape = [height as i64, width as i64, channels as i64];
        let strides = [
            (width * channels as u32) as i64,
            channels as i64,
            1,
        ];

        let builder = DlpackBuilder::<DLManagedTensorVersioned, 3>::with_slice_layout(ctx, &shape, &strides);
        let builder = builder
            .data(data_ptr)
            .dtype(P::Subpixel::data_type())
            .device(DLDevice {
                device_type: DLDeviceType::CPU,
                device_id: 0,
            });

        Ok(builder.build())
    }
}

impl<'a, P, M> TryFrom<&'a Dlpack<M>> for ImageBuffer<P, &'a [P::Subpixel]>
where
    P: Pixel,
    P::Subpixel: ImageSubpixel,
    M: ManagedTensor,
{
    type Error = &'static str;

    fn try_from(dlpack: &'a Dlpack<M>) -> Result<Self, Self::Error> {
        let tensor = dlpack.dl_tensor();
        if tensor.ndim != 3 {
            return Err("Tensor must have exactly 3 dimensions (height, width, channels)");
        }

        unsafe {
            let height = *tensor.shape.add(0);
            let width = *tensor.shape.add(1);
            let channels = *tensor.shape.add(2);

            if height <= 0 || width <= 0 || channels <= 0 {
                return Err("Tensor dimensions must be positive");
            }

            let height = height as u32;
            let width = width as u32;
            let channels = channels as u32;

            if channels != P::CHANNEL_COUNT as u32 {
                return Err("Channel count mismatch");
            }

            if tensor.device.device_type != DLDeviceType::CPU {
                return Err("Tensor must be on CPU to convert to an ImageBuffer");
            }

            let expected_dtype = P::Subpixel::data_type();
            if tensor.dtype.code != expected_dtype.code
                || tensor.dtype.bits != expected_dtype.bits
                || tensor.dtype.lanes != expected_dtype.lanes
            {
                return Err("Data type mismatch");
            }

            let len = (height * width * channels) as usize;
            let data_slice = std::slice::from_raw_parts(tensor.data as *const P::Subpixel, len);

            ImageBuffer::from_raw(width, height, data_slice)
                .ok_or("Failed to construct ImageBuffer from raw slice")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_image_roundtrip() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(10, 10, vec![0; 300]).unwrap();
        let dlpack = Dlpack::<DLManagedTensor>::try_from(img).unwrap();

        assert_eq!(dlpack.dl_tensor().ndim, 3);
        unsafe {
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 10);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 10);
            assert_eq!(*dlpack.dl_tensor().shape.add(2), 3);
        }

        let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&dlpack).unwrap();
        assert_eq!(img2.width(), 10);
        assert_eq!(img2.height(), 10);
    }
}
