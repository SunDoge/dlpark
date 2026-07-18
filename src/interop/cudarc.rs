//! CUDA interop via [`cudarc`].
//!
//! Provides zero-copy conversion between [`CudaSlice<T>`] and DLPack managed
//! tensors in both directions.
//!
//! # `from_cuda_slice` direction (`CudaSlice<T>` → `ManagedBox`)
//!
//! The slice is heap-boxed and stored as the `manager_ctx`; the underlying
//! CUDA allocation is freed when the DLPack deleter fires.
//!
//! This stays a plain function rather than `TryFrom` because the DLPack shape
//! and strides are not derivable from a bare `CudaSlice<T>` (it is just a flat
//! device buffer) — the conversion genuinely needs more than one argument, so
//! forcing it through `TryFrom`'s single-argument contract would not simplify
//! anything.
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
//! # Stream synchronization
//!
//! `from_cuda_slice` records a read fence on the slice's stream via
//! [`DevicePtr::device_ptr`] before capturing the pointer. The consumer must
//! wait on that stream before reading the data.
//!
//! `to_cuda_slice` creates a fresh `CudaContext`/stream for the given device
//! ordinal. If the producer and consumer are on different streams, the caller
//! is responsible for explicit synchronization.

use crate::{
    Borrowed, DlpackElement,
    builder::Builder,
    dlpack::ManagedBox,
    ffi::{DLDevice, DLDeviceType, DLManagedTensor, DLManagedTensorVersioned},
    legacy,
    managed_tensor::ManagedTensorBase,
    metadata, versioned,
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

    #[snafu(display("dtype mismatch: expected {expected:?}, got {actual:?}"))]
    DtypeMismatch {
        expected: crate::ffi::DLDataType,
        actual: crate::ffi::DLDataType,
    },

    #[snafu(display("cudarc driver error: {source}"))]
    Driver { source: cudarc::driver::DriverError },

    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },

    #[snafu(transparent)]
    Builder { source: crate::builder::Error },
}

// ---------------------------------------------------------------------------
// Forward: CudaSlice<T> → ManagedBox
// ---------------------------------------------------------------------------

/// Wrap a [`CudaSlice<T>`] as a [`Dlpack`].
///
/// - `slice` — owned GPU buffer; ownership is transferred into the DLPack
///   tensor's `manager_ctx` via `Box<CudaSlice<T>>`
/// - `shape`   — dimension sizes in elements (any rank)
/// - `strides` — element strides, must have the same length as `shape`
///
/// # Errors
///
/// - [`Error::MismatchedLength`] if `shape.len() != strides.len()`
/// - [`Error::NdimOverflow`] if `shape.len()` overflows `i32`
pub fn from_cuda_slice<T: DlpackElement>(
    slice: CudaSlice<T>,
    shape: &[i64],
    strides: &[i64],
) -> Result<legacy::Dlpack, Error> {
    let device_id = slice.ordinal() as i32;
    let data_ptr = device_ptr_of(&slice);

    Ok(
        Builder::new(Box::new(slice), metadata::CopiedSlice::new(shape, strides))
            .device(DLDevice::cuda(device_id))
            .data(data_ptr)
            .dtype(T::DTYPE)
            .try_build::<DLManagedTensor>()?,
    )
}

/// Same as [`from_cuda_slice`] but produces a [`DLManagedTensorVersioned`] (DLPack ≥ 1.0).
pub fn from_cuda_slice_versioned<T: DlpackElement>(
    slice: CudaSlice<T>,
    shape: &[i64],
    strides: &[i64],
) -> Result<versioned::Dlpack, Error> {
    let device_id = slice.ordinal() as i32;
    let data_ptr = device_ptr_of(&slice);

    Ok(
        Builder::new(Box::new(slice), metadata::CopiedSlice::new(shape, strides))
            .device(DLDevice::cuda(device_id))
            .data(data_ptr)
            .dtype(T::DTYPE)
            .try_build::<DLManagedTensorVersioned>()?,
    )
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
/// then uses [`upgrade_device_ptr`] to construct a `CudaSlice<T>` over the
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
/// - [`Error::Driver`] if `CudaContext::new` fails
/// - [`Error::Tensor`] for shape/element-count errors
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

        let len = tensor.num_elements()?;
        let cu_device_ptr = tensor.data as usize as u64;

        let ctx = CudaContext::new(tensor.device.device_id as usize)
            .map_err(|source| Error::Driver { source })?;
        let stream: Arc<CudaStream> = ctx.default_stream();

        // SAFETY:
        // - cu_device_ptr comes from the DLPack tensor's data field, which must be
        //   a valid CUDA allocation for at least `len * size_of::<T>()` bytes.
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
