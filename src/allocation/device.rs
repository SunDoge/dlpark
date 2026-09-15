//! Backend-neutral export of owned device allocations.
//!
//! This module deliberately describes only allocation ownership, the device
//! data handle, and the DLPack device. Stream synchronization belongs to the
//! backend: a producer must make its writes visible to the stream requested by
//! the eventual consumer before handing out the finished tensor.

use crate::{
    DlpackElement, ManagedTensorBase,
    allocation::dynamic,
    ffi::DLDevice,
    metadata::{Copied, Dynamic},
};
use std::ffi::c_void;

/// An owned allocation whose storage is accessible on a DLPack device.
///
/// Pass the allocation boxed to [`from_device_allocation`]. It is retained as
/// the DLPack `manager_ctx`, so its address and allocation lifetime extend
/// until the managed-tensor deleter runs.
///
/// # Safety
///
/// `dlpack_data` must remain valid for the allocation's lifetime and identify
/// storage on `device`. Its meaning follows the DLPack backend convention: it
/// is a CUDA device pointer for CUDA, but an opaque `id<MTLBuffer>` handle for
/// Metal rather than that buffer's host-visible `contents` pointer. Dropping
/// the implementor must release the allocation correctly and may happen on
/// any thread.
pub unsafe trait DeviceAllocation: Send + 'static {
    /// Returns the value to store in `DLTensor.data`.
    fn dlpack_data(&self) -> *mut c_void;

    /// Returns the DLPack device on which the allocation resides.
    fn device(&self) -> DLDevice;
}

/// Wraps an owned device allocation in a runtime-rank DLPack allocation.
///
/// Shape and strides are copied into the managed allocation. This function
/// does not synchronize a device stream or validate that the allocation is
/// large enough for the declared layout; those remain backend invariants.
pub fn from_device_allocation<T, M, B>(
    allocation: Box<B>,
    shape: &[i64],
    strides: &[i64],
) -> Result<dynamic::Initialized<M>, crate::metadata::Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
    B: DeviceAllocation,
{
    let data = allocation.dlpack_data();
    let device = allocation.device();
    let prepared = Dynamic::new(Copied(shape), Copied(strides)).prepare::<M>()?;
    let mut initialized = prepared.initialize(allocation);
    initialized
        .set_data(data)
        .set_device(device)
        .set_dtype(T::DTYPE);
    Ok(initialized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{DLDeviceType, DLManagedTensorVersioned};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct TestAllocation {
        data: Box<[f32]>,
        dropped: Arc<AtomicBool>,
    }

    unsafe impl DeviceAllocation for TestAllocation {
        fn dlpack_data(&self) -> *mut c_void {
            self.data.as_ptr().cast_mut().cast()
        }

        fn device(&self) -> DLDevice {
            DLDevice::cuda(3)
        }
    }

    impl Drop for TestAllocation {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn retains_allocation_and_copies_metadata() {
        let dropped = Arc::new(AtomicBool::new(false));
        let allocation = Box::new(TestAllocation {
            data: vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice(),
            dropped: Arc::clone(&dropped),
        });
        let pointer = allocation.dlpack_data();

        let initialized = from_device_allocation::<f32, DLManagedTensorVersioned, _>(
            allocation,
            &[2, 2],
            &[2, 1],
        )
        .unwrap();
        let tensor = unsafe { initialized.finish() };
        let descriptor = tensor.validate().unwrap();
        assert_eq!(descriptor.data_ptr(), pointer);
        assert_eq!(descriptor.device().device_type, DLDeviceType::CUDA);
        assert_eq!(descriptor.device().device_id, 3);
        assert_eq!(descriptor.shape(), &[2, 2]);
        assert_eq!(descriptor.strides().unwrap(), &[2, 1]);
        assert!(!dropped.load(Ordering::Relaxed));

        drop(tensor);
        assert!(dropped.load(Ordering::Relaxed));
    }
}
