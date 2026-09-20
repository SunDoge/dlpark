//! CUDA interop via [`cudarc`].
//!
//! Provides zero-copy conversion between `CudaSlice<T>` and DLPack managed
//! tensors in both directions.
//!
//! # `CudaSlice<T>` to initialized allocation
//!
//! `TryFrom<Box<CudaSlice<T>>>` produces a
//! [`CudaDlpackExport`](crate::interop::cudarc::CudaDlpackExport) containing a
//! contiguous 1-D tensor with shape `[slice.len()]`, strides `[1]`, and its
//! current work stream. Use
//! [`crate::interop::cudarc::from_cuda_slice`] for a higher-rank layout. The slice is stored as the
//! `manager_ctx`; the underlying CUDA allocation is freed when the DLPack
//! deleter fires. Moving the slice transfers ownership without copying the CUDA
//! allocation, so `IS_COPIED` remains unset.
//!
//! # Import direction (`Managed` → [`crate::interop::cudarc::ManagedCudaSlice`])
//!
//! `upgrade_device_ptr` wraps the DLPack tensor's raw device pointer into a
//! proper `CudaSlice<T>`. Because the DLPack tensor owns that allocation, we
//! must NOT call `cudaFree` when our `CudaSlice` is done.
//! [`crate::interop::cudarc::ManagedCudaSlice`]
//! owns both the managed tensor and the slice view. Its view destructor calls
//! `CudaSlice::leak`, preventing the double-free, before the managed tensor
//! is dropped.
//!
//! Unlike the forward direction, this conversion takes a single owned
//! `Managed<M>` and can fail, so it is exposed as
//! [`crate::TryFromDlpack`] for `ManagedCudaSlice<M, T>`.
//!
//! ## Why not return `CudaView<T>`?
//!
//! A `CudaView<'a, T>` holds `&'a` references to the parent slice's
//! `read`/`write`/`stream` fields. If those fields are freed (e.g. via
//! `CudaSlice::leak`, which calls `drop_in_place` on them), those references
//! become dangling. Calling `view.device_ptr()` would then be UB.
//! `ManagedCudaSlice` avoids this by keeping the slice alive.
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
//! temporary `CudaSlice` for the whole lifetime of `ManagedCudaSlice`, calls
//! `CudaSlice::leak` only when that wrapper is dropped, and then releases the
//! owning DLPack tensor. This adapter can be simplified if cudarc gains a
//! public non-owning raw-device-buffer type that does not require this
//! ownership workaround.
//!
//! # Stream context
//!
//! Converting a `CudaSlice` into an initialized allocation records a read fence
//! on the slice's stream via
//! `DevicePtr::device_ptr` before capturing the pointer. The consumer must
//! either use that stream or establish that the data is ready on another one.
//! Import takes an explicit [`CudaImport`](crate::interop::cudarc::CudaImport)
//! and adopts its ready stream without creating another CUDA context or
//! inserting hidden synchronization.

use crate::ffi::DLDeviceType;
use snafu::Snafu;

mod consumer;
mod producer;

pub use consumer::{CudaImport, ManagedCudaSlice};
pub use producer::{CudaDlpackExport, from_cuda_slice};

/// Errors raised during cudarc interop.
#[derive(Debug, Snafu)]
pub enum Error {
    /// Metadata preparation failed.
    #[snafu(transparent)]
    Metadata {
        /// The underlying metadata error.
        source: crate::metadata::Error,
    },

    /// The tensor is not on a CUDA device.
    #[snafu(display("tensor is not on a CUDA device, got {:?}", device_type))]
    NotCuda {
        /// The device type reported by the tensor.
        device_type: DLDeviceType,
    },

    /// The tensor data pointer is null.
    #[snafu(display("tensor data pointer is null"))]
    NullData,

    /// The CUDA slice length does not fit in `i64`.
    #[snafu(display("CUDA slice length {len} does not fit in i64"))]
    LengthOverflow {
        /// The offending length.
        len: usize,
        /// The underlying conversion error.
        source: std::num::TryFromIntError,
    },

    /// The CUDA device ordinal does not fit in `i32`.
    #[snafu(display("CUDA device ordinal {ordinal} does not fit in i32"))]
    DeviceIdOverflow {
        /// The offending device ordinal.
        ordinal: usize,
        /// The underlying conversion error.
        source: std::num::TryFromIntError,
    },

    /// The CUDA device ID is negative.
    #[snafu(display("CUDA device ID must be non-negative, got {device_id}"))]
    InvalidDeviceId {
        /// The offending device ID.
        device_id: i32,
    },

    /// The tensor's dtype does not match the requested Rust element type.
    #[snafu(display("dtype mismatch: expected {expected:?}, got {actual:?}"))]
    DtypeMismatch {
        /// The dtype expected by the caller.
        expected: crate::ffi::DLDataType,
        /// The dtype carried by the tensor.
        actual: crate::ffi::DLDataType,
    },

    /// The stream supplied for import belongs to another CUDA device.
    #[snafu(display(
        "CUDA import stream is on device {stream_device_id}, but the tensor is on device {tensor_device_id}"
    ))]
    StreamDeviceMismatch {
        /// The tensor's CUDA device ordinal.
        tensor_device_id: usize,
        /// The stream's CUDA device ordinal.
        stream_device_id: usize,
    },

    /// The underlying DLPack tensor failed validation.
    #[snafu(transparent)]
    Tensor {
        /// The underlying tensor error.
        source: crate::tensor::Error,
    },
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
        let tensor = unsafe { crate::tensor::TensorRef::from_raw(&tensor) }.unwrap();

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
        let tensor = unsafe { crate::tensor::TensorRef::from_raw(&tensor) }.unwrap();

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
        let tensor = unsafe { crate::tensor::TensorRef::from_raw(&tensor) }.unwrap();

        assert!(matches!(
            validated_cuda_parts::<i32>(&tensor),
            Err(Error::InvalidDeviceId { device_id: -1 })
        ));
    }

    /// End-to-end `CudaSlice` → DLPack → `ManagedCudaSlice` round-trip that
    /// continues directly on the export's current work stream.
    #[test]
    #[ignore = "requires a CUDA device; run with --ignored"]
    fn cuda_slice_roundtrips_on_current_work_stream() {
        use crate::{Managed, TryFromDlpack, ffi::DLManagedTensorVersioned};
        use cudarc::driver::{CudaContext, CudaSlice};
        use std::sync::Arc;

        let ctx = CudaContext::new(0).expect("CUDA context");
        let producer_stream = ctx.new_stream().expect("producer stream");
        let data = vec![1i32, 2, 3, 4];
        let slice: CudaSlice<i32> = producer_stream.clone_htod(&data).expect("htod copy");

        let export: CudaDlpackExport<
            crate::allocation::fixed::Initialized<DLManagedTensorVersioned, 1>,
        > = Box::new(slice).try_into().expect("producer");
        assert!(
            Arc::ptr_eq(export.current_stream(), &producer_stream),
            "producer returns the slice's stream"
        );

        let (initialized, current_stream) = export.into_parts();
        let managed: Managed<DLManagedTensorVersioned> = unsafe { initialized.finish() };

        let borrowed: ManagedCudaSlice<DLManagedTensorVersioned, i32> = unsafe {
            TryFromDlpack::try_from_dlpack(managed, CudaImport::ready_on(current_stream))
        }
        .expect("consumer import");

        assert!(Arc::ptr_eq(borrowed.stream(), &producer_stream));
        let host: Vec<i32> = borrowed.stream().clone_dtoh(&*borrowed).expect("dtoh copy");
        assert_eq!(host, data);
    }

    /// Same round-trip after the protocol layer bridges producer work to a
    /// different consumer stream.
    #[test]
    #[ignore = "requires a CUDA device; run with --ignored"]
    fn cuda_slice_roundtrips_on_preordered_consumer_stream() {
        use crate::{Managed, TryFromDlpack, ffi::DLManagedTensorVersioned};
        use cudarc::driver::{CudaContext, CudaSlice};

        let ctx = CudaContext::new(0).expect("CUDA context");
        let producer_stream = ctx.new_stream().expect("producer stream");
        let data = vec![5i32, 6, 7];
        let slice: CudaSlice<i32> = producer_stream.clone_htod(&data).expect("htod copy");

        let export = from_cuda_slice::<i32, DLManagedTensorVersioned>(Box::new(slice), &[3], &[1])
            .expect("producer");
        let consumer_stream = export
            .current_stream()
            .fork()
            .expect("consumer stream ordered after producer");
        let (initialized, _producer_stream) = export.into_parts();
        let managed: Managed<DLManagedTensorVersioned> = unsafe { initialized.finish() };

        let borrowed: ManagedCudaSlice<DLManagedTensorVersioned, i32> = unsafe {
            TryFromDlpack::try_from_dlpack(managed, CudaImport::ready_on(consumer_stream.clone()))
        }
        .expect("consumer import");

        assert!(std::sync::Arc::ptr_eq(borrowed.stream(), &consumer_stream));
        let host: Vec<i32> = borrowed.stream().clone_dtoh(&*borrowed).expect("dtoh copy");
        assert_eq!(host, data);
    }
}
