//! CUDA interop via [`cudarc`].
//!
//! Provides zero-copy conversion between `CudaSlice<T>` and DLPack managed
//! tensors in both directions.
//!
//! # `CudaSlice<T>` to initialized allocation
//!
//! `TryFrom<Box<CudaSlice<T>>>` treats the flat device buffer as a contiguous
//! 1-D tensor with shape `[slice.len()]` and strides `[1]`. Use
//! [`crate::interop::cudarc::from_cuda_slice`] for a higher-rank layout. The slice is stored as the
//! `manager_ctx`; the underlying CUDA allocation is freed when the DLPack
//! deleter fires. Because ownership is transferred without any remaining Rust
//! views, the initialized allocation starts with [`crate::DlpackFlags::IS_COPIED`].
//!
//! # `to_cuda_slice` direction (`Foreign` → [`crate::interop::cudarc::BorrowedCudaSlice`])
//!
//! `upgrade_device_ptr` wraps the DLPack tensor's raw device pointer into a
//! proper `CudaSlice<T>`. Because the DLPack tensor owns that allocation, we
//! must NOT call `cudaFree` when our `CudaSlice` is done.
//! [`crate::interop::cudarc::BorrowedCudaSlice`]
//! owns both the managed tensor and the slice view. Its view destructor calls
//! `CudaSlice::leak`, preventing the double-free, before the managed tensor
//! is dropped.
//!
//! Unlike the forward direction, this conversion takes a single owned
//! `Foreign<M>` and can fail, so it is exposed as
//! [`crate::TryFromDlpack`] for `BorrowedCudaSlice<M, T>`.
//!
//! ## Why not return `CudaView<T>`?
//!
//! A `CudaView<'a, T>` holds `&'a` references to the parent slice's
//! `read`/`write`/`stream` fields. If those fields are freed (e.g. via
//! `CudaSlice::leak`, which calls `drop_in_place` on them), those references
//! become dangling. Calling `view.device_ptr()` would then be UB.
//! `BorrowedCudaSlice` avoids this by keeping the slice alive.
//!
//! In particular, the workaround sometimes shown for older cudarc releases
//! cannot be used with the current API:
//!
//! 1. create a temporary `CudaSlice` with `CudaDevice::upgrade_device_ptr`;
//! 2. create a `CudaView` from that slice;
//! 3. extend the view lifetime with `transmute`;
//! 4. call `CudaSlice::leak` to avoid freeing the DLPack allocation.
//!
//! Step 4 drops the exact event and stream fields borrowed by the view, so the
//! returned view contains dangling references. dlpark instead retains the
//! temporary `CudaSlice` for the whole lifetime of `BorrowedCudaSlice`, calls
//! `CudaSlice::leak` only when that wrapper is dropped, and then releases the
//! owning DLPack tensor. This adapter can be simplified if cudarc gains a
//! public non-owning raw-device-buffer type that does not require this
//! ownership workaround.
//!
//! # Stream synchronization
//!
//! Converting a `CudaSlice` into an initialized allocation records a read fence on the
//! slice's stream via
//! `DevicePtr::device_ptr` before capturing the pointer. The consumer must
//! wait on that stream before reading the data.
//!
//! `to_cuda_slice` creates a fresh `CudaContext`/stream for the given device
//! ordinal. If the producer and consumer are on different streams, the caller
//! is responsible for explicit synchronization.

use crate::ffi::DLDeviceType;
use snafu::Snafu;

mod consumer;
mod producer;

pub use consumer::BorrowedCudaSlice;
pub use producer::from_cuda_slice;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Metadata { source: crate::metadata::Error },

    #[snafu(display("tensor is not on a CUDA device, got {:?}", device_type))]
    NotCuda { device_type: DLDeviceType },

    #[snafu(display("tensor data pointer is null"))]
    NullData,

    #[snafu(display("CUDA slice length {len} does not fit in i64"))]
    LengthOverflow {
        len: usize,
        source: std::num::TryFromIntError,
    },

    #[snafu(display("CUDA device ordinal {ordinal} does not fit in i32"))]
    DeviceIdOverflow {
        ordinal: usize,
        source: std::num::TryFromIntError,
    },

    #[snafu(display("CUDA device ID must be non-negative, got {device_id}"))]
    InvalidDeviceId { device_id: i32 },

    #[snafu(display("dtype mismatch: expected {expected:?}, got {actual:?}"))]
    DtypeMismatch {
        expected: crate::ffi::DLDataType,
        actual: crate::ffi::DLDataType,
    },

    #[snafu(display("cudarc driver error: {source}"))]
    Driver { source: cudarc::driver::DriverError },

    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },
}

#[cfg(test)]
use crate::ffi::DLDevice;
#[cfg(test)]
use consumer::validated_cuda_parts;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{DLDataType, DLTensor};

    #[test]
    fn validated_cuda_parts_applies_byte_offset() {
        let data = [0i32; 3];
        let shape = [2i64];
        let strides = [1i64];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::cuda(0),
            ndim: 1,
            dtype: DLDataType::of::<i32>(),
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            byte_offset: std::mem::size_of::<i32>() as u64,
        };

        let (ptr, len, device_id) = validated_cuda_parts::<i32>(&tensor).unwrap();
        assert_eq!(ptr, unsafe { data.as_ptr().add(1) } as usize as u64);
        assert_eq!(len, 2);
        assert_eq!(device_id, 0);
    }

    #[test]
    fn validated_cuda_parts_rejects_non_compact_strides() {
        let data = [0i32; 5];
        let shape = [2i64, 2];
        let strides = [3i64, 1];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::cuda(0),
            ndim: 2,
            dtype: DLDataType::of::<i32>(),
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            byte_offset: 0,
        };

        assert!(matches!(
            validated_cuda_parts::<i32>(&tensor),
            Err(Error::Tensor {
                source: crate::tensor::Error::NonCompactStrides
            })
        ));
    }

    #[test]
    fn validated_cuda_parts_rejects_negative_device_id() {
        let data = [0i32; 1];
        let shape = [1i64];
        let strides = [1i64];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::cuda(-1),
            ndim: 1,
            dtype: DLDataType::of::<i32>(),
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            byte_offset: 0,
        };

        assert!(matches!(
            validated_cuda_parts::<i32>(&tensor),
            Err(Error::InvalidDeviceId { device_id: -1 })
        ));
    }
}
