use std::sync::Arc;

use crate::{
    ffi::{ManagedTensor, ManagedTensorVersioned, Tensor},
    traits::IntoDlpack,
};

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        // SAFETY: The pointer is valid and the memory is managed by the DLPack library.
        if let Some(deleter) = self.deleter {
            unsafe {
                deleter(self);
            }
        }
    }
}

impl Drop for ManagedTensorVersioned {
    fn drop(&mut self) {
        // SAFETY: The pointer is valid and the memory is managed by the DLPack library.
        if let Some(deleter) = self.deleter {
            unsafe {
                deleter(self);
            }
        }
    }
}

unsafe extern "C" fn box_deleter<T>(managed_tensor: *mut ManagedTensor) {
    let ctx = unsafe { (*managed_tensor).manager_ctx } as *mut T;
    let _ = unsafe { Box::from_raw(ctx) };
}

unsafe extern "C" fn arc_deleter<T>(managed_tensor: *mut ManagedTensor) {
    let ctx = unsafe { (*managed_tensor).manager_ctx } as *mut T;
    let _ = unsafe { Arc::from_raw(ctx) };
}

// impl<T> From<&T> for Tensor
// where
//     T: TensorLike,
// {
//     fn from(t: &T) -> Self {
//         let ndim = t.shape().len() as i32;
//         Tensor {
//             data: t.data_ptr(),
//             device: t.device(),
//             dtype: t.dtype(),
//             shape: t.shape().as_ptr() as *mut _,
//             strides: t
//                 .strides()
//                 .map(|x| x.as_ptr())
//                 .unwrap_or_else(|| std::ptr::null()) as *mut _,
//             byte_offset: t.byte_offset(),
//             ndim,
//         }
//     }
// }

// impl TensorLike for (Vec<f32>, Vec<i64>) {
//     fn data_ptr(&self) -> *mut std::ffi::c_void {
//         self.0.as_ptr() as *mut std::ffi::c_void
//     }

//     fn shape(&self) -> &[i64] {
//         &self.1
//     }

//     fn strides(&self) -> Option<&[i64]> {
//         None
//     }

//     fn device(&self) -> crate::ffi::Device {
//         crate::ffi::Device {
//             device_type: crate::ffi::DeviceType::Cpu,
//             device_id: 0,
//         }
//     }

//     fn dtype(&self) -> crate::ffi::DataType {
//         crate::ffi::DataType {
//             code: crate::ffi::DataTypeCode::Float,
//             bits: 32,
//             lanes: 1,
//         }
//     }

//     fn byte_offset(&self) -> u64 {
//         0
//     }
// }

// impl<T> IntoDlpack for Box<T>
// where
//     T: TensorLike,
// {
//     fn into_dlpack(self) -> Box<ManagedTensor> {
//         let tensor = Tensor::from(self.as_ref());
//         Box::new(ManagedTensor {
//             dl_tensor: tensor,
//             manager_ctx: Box::into_raw(Box::new(self)) as *mut
// std::ffi::c_void,             deleter: Some(box_deleter::<T>),
//         })
//     }
// }

// impl<T> IntoDlpack for Arc<T>  where T: TensorLike{
//     fn into_dlpack(self) -> Box<ManagedTensor> {
//         let tensor = Tensor::from(self.as_ref());
//         Box::new(ManagedTensor {
//             dl_tensor: tensor,
//             manager_ctx: Arc::into_raw(self) as *mut std::ffi::c_void,
//             deleter: Some(arc_deleter::<T>),
//         })
//     }
// }
