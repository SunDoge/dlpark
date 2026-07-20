//! CUDA interop via [`cudarc`].
//!
//! Provides zero-copy conversion between [`CudaSlice<T>`] and DLPack managed
//! tensors in both directions.
//!
//! # `CudaSlice<T>` → [`Builder`]
//!
//! [`Builder::try_from`] treats the flat device buffer as a contiguous 1-D
//! tensor with shape `[slice.len()]` and strides `[1]`. Call
//! [`Builder::metadata`] to replace that default before building a tensor with
//! a higher-rank layout. The slice is heap-boxed and stored as the
//! `manager_ctx`; the underlying CUDA allocation is freed when the DLPack
//! deleter fires. Because ownership is transferred without any remaining Rust
//! views, the builder starts with [`DlpackFlags::IS_COPIED`].
//!
//! # `to_cuda_slice` direction (`ManagedBox` → [`BorrowedCudaSlice`])
//!
//! `upgrade_device_ptr` wraps the DLPack tensor's raw device pointer into a
//! proper `CudaSlice<T>`. Because the DLPack tensor owns that allocation, we
//! must NOT call `cudaFree` when our `CudaSlice` is done. [`BorrowedCudaSlice`]
//! owns both the managed tensor and the slice view. Its view destructor calls
//! [`CudaSlice::leak`], preventing the double-free, before the managed tensor
//! is dropped.
//!
//! Unlike the forward direction, this conversion takes a single owned
//! `ManagedBox<M>` and can fail, so it is exposed as
//! `TryFrom<ManagedBox<M>> for BorrowedCudaSlice<M, T>`.
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
//! Converting a [`CudaSlice`] into a builder records a read fence on the
//! slice's stream via
//! [`DevicePtr::device_ptr`] before capturing the pointer. The consumer must
//! wait on that stream before reading the data.
//!
//! `to_cuda_slice` creates a fresh `CudaContext`/stream for the given device
//! ordinal. If the producer and consumer are on different streams, the caller
//! is responsible for explicit synchronization.

use crate::{
    Borrowed, DlpackElement, DlpackFlags,
    builder::Builder,
    dlpack::ManagedBox,
    ffi::{DLDevice, DLDeviceType},
    managed_tensor::ManagedTensorBase,
    metadata,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use snafu::{Snafu, ensure};
use std::{mem::ManuallyDrop, ops::Deref, os::raw::c_void, sync::Arc};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Snafu)]
pub enum Error {
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

// ---------------------------------------------------------------------------
// Forward: CudaSlice<T> → ManagedBox
// ---------------------------------------------------------------------------

impl<T: DlpackElement> TryFrom<CudaSlice<T>>
    for Builder<Box<CudaSlice<T>>, metadata::CopiedArray<[i64; 1], [i64; 1], 1>>
{
    type Error = Error;

    fn try_from(slice: CudaSlice<T>) -> Result<Self, Self::Error> {
        let len = i64::try_from(slice.len()).map_err(|source| Error::LengthOverflow {
            len: slice.len(),
            source,
        })?;
        let device_id =
            i32::try_from(slice.ordinal()).map_err(|source| Error::DeviceIdOverflow {
                ordinal: slice.ordinal(),
                source,
            })?;
        let data_ptr = device_ptr_of(&slice);

        let builder = Builder::new(Box::new(slice), metadata::CopiedArray::new([len], [1]))
            .device(DLDevice::cuda(device_id))
            .data(data_ptr)
            .dtype(T::DTYPE);

        // SAFETY: the owned CudaSlice was moved into the builder and Rust
        // permits no outstanding views while that move occurs.
        Ok(unsafe { builder.flags_unchecked(DlpackFlags::IS_COPIED) })
    }
}

/// Wraps a [`CudaSlice<T>`] in a [`Builder`] with explicit shape and strides.
///
/// - `slice` — owned GPU buffer; ownership is transferred into the DLPack
///   tensor's `manager_ctx` via `Box<CudaSlice<T>>`
/// - `shape`   — dimension sizes in elements (any rank)
/// - `strides` — element strides, must have the same length as `shape`
///
/// # Errors
///
/// - [`metadata::Error::MismatchedLength`] if `shape.len() != strides.len()`
/// - [`metadata::Error::NdimOverflow`] if `shape.len()` overflows `i32`
pub fn from_cuda_slice<'a, T: DlpackElement>(
    slice: CudaSlice<T>,
    shape: &'a [i64],
    strides: &'a [i64],
) -> Result<Builder<Box<CudaSlice<T>>, metadata::CopiedSlice<&'a [i64], &'a [i64]>>, Error> {
    Ok(Builder::try_from(slice)?.metadata(metadata::CopiedSlice::new(shape, strides)))
}

// ---------------------------------------------------------------------------
// Reverse: ManagedBox<M> → BorrowedCudaSlice<M, T>
// ---------------------------------------------------------------------------

/// A `CudaSlice<T>` view that also owns the backing [`ManagedBox`] tensor.
///
/// Implements [`Deref<Target = CudaSlice<T>>`] so it can be passed directly
/// to any cudarc API that takes `&CudaSlice<T>`.
///
/// # Memory safety
///
/// This type owns the [`ManagedBox`] rather than borrowing it. On drop, the
/// inner `CudaSlice<T>` view calls [`CudaSlice::leak`] instead of running its
/// normal destructor, and is then dropped before the `ManagedBox`. This avoids
/// calling `cudaFree` directly while allowing the DLPack deleter to release the
/// allocation through its original owner.
pub struct BorrowedCudaSlice<M: ManagedTensorBase, T> {
    inner: Borrowed<ManagedBox<M>, CudaSliceView<T>>,
}

struct CudaSliceView<T>(ManuallyDrop<CudaSlice<T>>);

impl<T> Drop for CudaSliceView<T> {
    fn drop(&mut self) {
        let slice = unsafe { ManuallyDrop::take(&mut self.0) };
        slice.leak();
    }
}

impl<T> Deref for CudaSliceView<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.0
    }
}

impl<M: ManagedTensorBase, T> BorrowedCudaSlice<M, T> {
    /// Returns the managed tensor that owns the CUDA allocation.
    pub fn dlpack(&self) -> &ManagedBox<M> {
        self.inner.owner()
    }
}

impl<M: ManagedTensorBase, T> Deref for BorrowedCudaSlice<M, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.inner
    }
}

/// Converts a [`ManagedBox`] tensor into an owning CUDA slice view.
///
/// Creates a new [`CudaContext`] and default stream for the tensor's device,
/// then uses `CudaDevice::upgrade_device_ptr` to construct a `CudaSlice<T>` over the
/// tensor's raw device pointer without taking ownership of the allocation.
///
/// The returned [`BorrowedCudaSlice`] implements `Deref<Target = CudaSlice<T>>`,
/// so it can be passed to any cudarc API. It retains ownership of the input
/// [`ManagedBox`] and releases it only after disabling the `CudaSlice`
/// destructor with [`CudaSlice::leak`].
///
/// # Errors
///
/// - [`Error::NotCuda`] if the tensor is not on a CUDA device
/// - [`Error::DtypeMismatch`] if the element type does not match `T`
/// - [`Error::NullData`] if the data pointer is null
/// - [`Error::InvalidDeviceId`] if the CUDA device ID is negative
/// - [`Error::Driver`] if `CudaContext::new` fails
/// - [`Error::Tensor`] for non-compact layouts or shape, element-count,
///   byte-offset, and alignment errors
///
/// # Stream synchronization
///
/// A fresh `CudaContext`/stream is created for the device. If the DLPack
/// producer used a different stream, the caller must synchronize explicitly
/// (e.g. via `cudaDeviceSynchronize`) before submitting GPU work.
impl<T, M> TryFrom<ManagedBox<M>> for BorrowedCudaSlice<M, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: ManagedBox<M>) -> Result<Self, Self::Error> {
        let tensor = dlpack.tensor();
        let (cu_device_ptr, len, device_id) = validated_cuda_parts::<T>(tensor)?;

        let ctx = CudaContext::new(device_id).map_err(|source| Error::Driver { source })?;
        let stream: Arc<CudaStream> = ctx.default_stream();

        // SAFETY:
        // - cu_device_ptr is the checked, byte-offset-adjusted DLPack data pointer,
        //   which must reference a valid compact CUDA allocation for at least
        //   `len * size_of::<T>()` bytes.
        // - CudaSliceView::drop calls leak() instead of allowing cudaFree to run.
        let slice = unsafe { stream.upgrade_device_ptr::<T>(cu_device_ptr, len) };

        let view = CudaSliceView(ManuallyDrop::new(slice));
        let inner = unsafe { Borrowed::new_unchecked(dlpack, view) };

        Ok(BorrowedCudaSlice { inner })
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

fn validated_cuda_parts<T: DlpackElement>(
    tensor: &crate::ffi::DLTensor,
) -> Result<(u64, usize, usize), Error> {
    ensure!(
        tensor.device.device_type == DLDeviceType::CUDA,
        NotCudaSnafu {
            device_type: tensor.device.device_type
        }
    );
    ensure!(
        tensor.dtype.is::<T>(),
        DtypeMismatchSnafu {
            expected: T::DTYPE,
            actual: tensor.dtype,
        }
    );
    ensure!(!tensor.data.is_null(), NullDataSnafu);
    let device_id =
        usize::try_from(tensor.device.device_id).map_err(|_| Error::InvalidDeviceId {
            device_id: tensor.device.device_id,
        })?;

    let len = tensor.num_elements()?;
    if !tensor.is_compact()? {
        return Err(Error::Tensor {
            source: crate::tensor::Error::NonCompactStrides,
        });
    }
    let cu_device_ptr = tensor.offset_data_ptr::<T>()? as usize as u64;
    Ok((cu_device_ptr, len, device_id))
}

/// Returns the CUDA device pointer of `slice` as a `*mut c_void` and records
/// a read fence on the slice's stream.
///
/// The borrow of `slice` is fully released before this function returns, so
/// the caller may move `slice` afterward.
fn device_ptr_of<T>(slice: &CudaSlice<T>) -> *mut c_void {
    // Clone the Arc so that `stream` does not borrow `slice`.
    let stream = slice.stream().clone();

    let (cu_ptr, sync) = slice.device_ptr(&stream);
    drop(sync); // commits the read-fence event; releases the borrow of slice

    cu_ptr as usize as *mut c_void
}

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
