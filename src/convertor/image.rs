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
        Ok(ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, s).expect("fuck"))
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
        Ok(ImageBuffer::from_raw(width, height, value).expect("fuck"))
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
        Ok(ImageBuffer::from_raw(shape[1] as u32, shape[0] as u32, s).expect("fuck"))
    }
}

impl<P> TryFrom<SafeManagedTensor> for ImageBuffer<P, SafeManagedTensor>
where
    P: Pixel<Subpixel = u8>,
{
    type Error = crate::Error;

    fn try_from(value: SafeManagedTensor) -> Result<Self> {
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
        Ok(ImageBuffer::from_raw(width, height, value).expect("fuck"))
    }
}
