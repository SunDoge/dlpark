use crate::Result;
use crate::error::InvalidChannelsSnafu;
use crate::error::InvalidDimensionsSnafu;
use crate::error::UnsupportedMemoryOrderSnafu;
use crate::ffi;
use crate::traits::{InferDataType, RowMajorCompactLayout, TensorLike, TensorView};
use crate::utils::MemoryOrder;
use crate::{SafeManagedTensor, SafeManagedTensorVersioned};
use image::{ImageBuffer, Pixel};
use snafu::ensure;

impl<P> TensorLike<RowMajorCompactLayout> for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel,
    <P as Pixel>::Subpixel: InferDataType,
{
    type Error = crate::Error;
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut P::Subpixel as *mut _
    }

    fn device(&self) -> Result<ffi::Device> {
        Ok(ffi::Device::CPU)
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![
            self.height() as i64,
            self.width() as i64,
            P::CHANNEL_COUNT as i64,
        ])
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn data_type(&self) -> Result<ffi::DataType> {
        Ok(P::Subpixel::data_type())
    }
}

impl<'a, P> TryFrom<&'a SafeManagedTensorVersioned> for ImageBuffer<P, &'a [P::Subpixel]>
where
    P: Pixel,
{
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensorVersioned) -> Result<Self> {
        ensure!(
            value.memory_order() == MemoryOrder::RowMajorContiguous,
            UnsupportedMemoryOrderSnafu {
                order: value.memory_order(),
                expected: MemoryOrder::RowMajorContiguous
            }
        );
        let shape = value.shape();
        let s = unsafe { value.as_slice::<P::Subpixel>()? };
        let img = ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, s)
            .expect("container is not big enough");
        Ok(img)
    }
}

impl<P> TryFrom<SafeManagedTensorVersioned> for ImageBuffer<P, SafeManagedTensorVersioned>
where
    P: Pixel<Subpixel = u8>,
{
    type Error = crate::Error;

    fn try_from(value: SafeManagedTensorVersioned) -> Result<Self> {
        ensure!(
            value.num_dimensions() == 3,
            InvalidDimensionsSnafu {
                expected: 3usize,
                actual: value.num_dimensions()
            }
        );
        let shape = value.shape();
        let width = shape[1] as u32;
        let height = shape[0] as u32;
        let channel = shape[2] as u8;
        ensure!(
            channel == P::CHANNEL_COUNT,
            InvalidChannelsSnafu {
                expected: P::CHANNEL_COUNT as i64,
                actual: channel as i64
            }
        );
        let img = ImageBuffer::from_raw(width, height, value).expect("container is not big enough");
        Ok(img)
    }
}

impl<'a, P> TryFrom<&'a SafeManagedTensor> for ImageBuffer<P, &'a [P::Subpixel]>
where
    P: Pixel,
{
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensor) -> Result<Self> {
        ensure!(
            value.memory_order() == MemoryOrder::RowMajorContiguous,
            UnsupportedMemoryOrderSnafu {
                order: value.memory_order(),
                expected: MemoryOrder::RowMajorContiguous
            }
        );
        let shape = value.shape();
        let s = unsafe { value.as_slice::<P::Subpixel>()? };
        let img = ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, s)
            .expect("container is not big enough");
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn test_dlpack() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])
            .expect("container is not big enough");
        let mt = SafeManagedTensor::new(img).unwrap();
        let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&mt).unwrap();
        assert_eq!(img2.width(), 100);
        assert_eq!(img2.height(), 100);
    }

    #[test]
    fn test_dlpack_versioned() {
        let img = ImageBuffer::<Rgb<u8>, _>::from_vec(100, 100, vec![0; 100 * 100 * 3])
            .expect("container is not big enough");
        let mt = SafeManagedTensorVersioned::new(img).unwrap();
        let img2 = ImageBuffer::<Rgb<u8>, _>::try_from(&mt).unwrap();
        assert_eq!(img2.width(), 100);
        assert_eq!(img2.height(), 100);
    }
}
