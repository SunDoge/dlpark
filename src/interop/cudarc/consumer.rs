use super::{DtypeMismatchSnafu, Error, NotCudaSnafu, NullDataSnafu};
use crate::{
    Borrowed, DlpackElement, Foreign, ManagedTensorBase, TryFromDlpack, ffi::DLDeviceType,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use snafu::ensure;
use std::{mem::ManuallyDrop, ops::Deref, sync::Arc};

/// A `CudaSlice<T>` view that also owns the backing [`Foreign`] tensor.
///
/// Implements [`Deref<Target = CudaSlice<T>>`] so it can be passed directly
/// to any cudarc API that takes `&CudaSlice<T>`.
///
/// # Memory safety
///
/// This type owns the [`Foreign`] rather than borrowing it. On drop, the
/// inner `CudaSlice<T>` view calls [`CudaSlice::leak`] instead of running its
/// normal destructor, and is then dropped before the `Foreign`. This avoids
/// calling `cudaFree` directly while allowing the DLPack deleter to release the
/// allocation through its original owner.
pub struct BorrowedCudaSlice<M: ManagedTensorBase, T> {
    inner: Borrowed<Foreign<M>, CudaSliceView<T>>,
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
    pub fn dlpack(&self) -> &Foreign<M> {
        self.inner.owner()
    }
}

impl<M: ManagedTensorBase, T> Deref for BorrowedCudaSlice<M, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.inner
    }
}

/// Converts a [`Foreign`] tensor into an owning CUDA slice view.
///
/// Creates a new [`CudaContext`] and default stream for the tensor's device,
/// then uses `CudaDevice::upgrade_device_ptr` to construct a `CudaSlice<T>` over the
/// tensor's raw device pointer without taking ownership of the allocation.
///
/// The returned [`BorrowedCudaSlice`] implements `Deref<Target = CudaSlice<T>>`,
/// so it can be passed to any cudarc API. It retains ownership of the input
/// [`Foreign`] and releases it only after disabling the `CudaSlice`
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
impl<T, M> TryFromDlpack<Foreign<M>> for BorrowedCudaSlice<M, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: Foreign<M>) -> Result<Self, Self::Error> {
        let tensor = unsafe { dlpack.tensor() };
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

pub(super) fn validated_cuda_parts<T: DlpackElement>(
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

    let len = unsafe { tensor.num_elements()? };
    if !unsafe { tensor.is_compact()? } {
        return Err(Error::Tensor {
            source: crate::tensor::Error::NonCompactStrides,
        });
    }
    let cu_device_ptr = unsafe { tensor.offset_data_ptr::<T>()? } as usize as u64;
    Ok((cu_device_ptr, len, device_id))
}
