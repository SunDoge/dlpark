use cudarc::driver::{CudaContext, CudaSlice, CudaView, DevicePtr, DeviceSlice};
use snafu::ensure;

use crate::error::UnsupportedDeviceSnafu;
use crate::ffi::DeviceType;
use crate::prelude::*;
use crate::traits::{InferDataType, RowMajorCompactLayout, TensorLike};

impl<T> TensorLike<RowMajorCompactLayout> for CudaSlice<T>
where
    T: InferDataType,
{
    type Error = crate::Error;

    fn data_ptr(&self) -> *mut std::ffi::c_void {
        let stream = self.stream();
        let (ptr, _) = self.device_ptr(stream);
        ptr as *mut T as *mut _
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> crate::Result<crate::ffi::Device> {
        Ok(crate::ffi::Device::cuda(self.ordinal()))
    }

    fn data_type(&self) -> crate::Result<crate::ffi::DataType> {
        Ok(T::data_type())
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<T> TensorLike<RowMajorCompactLayout> for CudaView<'_, T>
where
    T: InferDataType,
{
    type Error = crate::Error;

    fn data_ptr(&self) -> *mut std::ffi::c_void {
        let stream = self.stream();
        let (ptr, _) = self.device_ptr(stream);
        ptr as *mut T as *mut _
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![self.len() as i64])
    }

    fn device(&self) -> crate::Result<crate::ffi::Device> {
        // Ok(crate::ffi::Device::cuda(self.ordinal()))
        Ok(crate::ffi::Device::cuda(0))
    }

    fn data_type(&self) -> crate::Result<crate::ffi::DataType> {
        Ok(T::data_type())
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

impl<'a, T> TryFrom<&'a SafeManagedTensor> for CudaView<'a, T> {
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensor) -> Result<Self, Self::Error> {
        let device = value.device();
        ensure!(
            device.device_type == DeviceType::Cuda,
            UnsupportedDeviceSnafu {
                device: device.device_type
            }
        );

        let ctx = CudaContext::new(device.device_id as usize).unwrap();
        let stream = ctx.default_stream();
        unsafe {
            let s = stream.upgrade_device_ptr(value.data_ptr() as u64, value.num_elements());
            let temp_view: CudaView<'_, T> = s.slice(..);
            let view = std::mem::transmute::<CudaView<'_, T>, CudaView<'_, T>>(temp_view);
            s.leak();
            Ok(view)
        }
    }
}

impl<'a, T> TryFrom<&'a SafeManagedTensorVersioned> for CudaView<'a, T> {
    type Error = crate::Error;

    fn try_from(value: &'a SafeManagedTensorVersioned) -> Result<Self, Self::Error> {
        let device = value.device();
        ensure!(
            device.device_type == DeviceType::Cuda,
            UnsupportedDeviceSnafu {
                device: device.device_type
            }
        );

        let ctx = CudaContext::new(device.device_id as usize).unwrap();
        let stream = ctx.default_stream();
        unsafe {
            let s = stream.upgrade_device_ptr(value.data_ptr() as u64, value.num_elements());
            let temp_view: CudaView<'_, T> = s.slice(..);
            let view = std::mem::transmute::<CudaView<'_, T>, CudaView<'_, T>>(temp_view);
            s.leak();
            Ok(view)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let slice = stream.alloc_zeros::<f32>(10).unwrap();
        let mt = SafeManagedTensor::new(slice).unwrap();
        assert_eq!(mt.num_elements(), 10);
        let view = CudaView::<f32>::try_from(&mt).unwrap();
        assert_eq!(view.len(), 10);
    }

    #[test]
    fn test_cuda_view() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let slice = stream.alloc_zeros::<f32>(10).unwrap();
        let view = slice.slice(..);

        let mt = SafeManagedTensorVersioned::new(view).unwrap();
        assert_eq!(mt.num_elements(), 10);
        let view = CudaView::<f32>::try_from(&mt).unwrap();
        assert_eq!(view.len(), 10);
    }
}
