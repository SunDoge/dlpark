use super::{DtypeMismatchSnafu, Error, NotCudaSnafu, NullDataSnafu};
use crate::{
    Borrowed, DlpackElement, Managed, ManagedTensorBase, TryFromDlpack, ffi::DLDeviceType,
};
use cudarc::driver::{CudaSlice, CudaStream};
use snafu::ensure;
use std::{mem::ManuallyDrop, ops::Deref, sync::Arc};

/// Runtime context for importing a CUDA DLPack tensor into cudarc.
///
/// `ready_on` is the stream on which the caller has established that the
/// tensor data is ready. For C Exchange this can be the producer's current
/// work stream. For Python `__dlpack__`, it is the consumer stream previously
/// passed to the producer.
#[derive(Clone)]
pub struct CudaImport {
    ready_on: Arc<CudaStream>,
}

impl CudaImport {
    /// Uses `stream` as the execution context on which the tensor is ready.
    pub fn ready_on(stream: Arc<CudaStream>) -> Self {
        Self { ready_on: stream }
    }

    /// Returns the stream on which the imported tensor is ready.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.ready_on
    }
}

/// A `CudaSlice<T>` view that also owns the backing [`Managed`] tensor.
///
/// Implements [`Deref<Target = CudaSlice<T>>`] so it can be passed directly
/// to any cudarc API that takes `&CudaSlice<T>`.
///
/// # Memory safety
///
/// This type owns the [`Managed`] rather than borrowing it. On drop, the
/// inner `CudaSlice<T>` view calls [`CudaSlice::leak`] instead of running its
/// normal destructor, and is then dropped before the `Managed`. This avoids
/// calling `cudaFree` directly while allowing the DLPack deleter to release the
/// allocation through its original owner.
pub struct ManagedCudaSlice<M: ManagedTensorBase, T> {
    inner: Borrowed<Managed<M>, CudaSliceView<T>>,
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

impl<M: ManagedTensorBase, T> ManagedCudaSlice<M, T> {
    /// Returns the managed tensor that owns the CUDA allocation.
    pub fn dlpack(&self) -> &Managed<M> {
        self.inner.owner()
    }

    /// Drops the CUDA slice view and returns the DLPack tensor that owns its allocation.
    pub fn into_dlpack(self) -> Managed<M> {
        self.inner.into_owner()
    }
}

impl<M: ManagedTensorBase, T> Deref for ManagedCudaSlice<M, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        &self.inner
    }
}

/// Converts a [`Managed`] tensor into an owning CUDA slice view.
///
/// Uses the stream supplied by [`CudaImport`] to construct a `CudaSlice<T>`
/// over the tensor's raw device pointer without taking ownership of the
/// allocation.
///
/// The returned [`ManagedCudaSlice`] implements `Deref<Target = CudaSlice<T>>`,
/// so it can be passed to any cudarc API. It retains ownership of the input
/// [`Managed`] and releases it only after disabling the `CudaSlice`
/// destructor with [`CudaSlice::leak`].
///
/// # Errors
///
/// - [`Error::NotCuda`] if the tensor is not on a CUDA device
/// - [`Error::DtypeMismatch`] if the element type does not match `T`
/// - [`Error::NullData`] if the data pointer is null
/// - [`Error::InvalidDeviceId`] if the CUDA device ID is negative
/// - [`Error::StreamDeviceMismatch`] if the ready stream is for another device
/// - [`Error::Tensor`] for non-compact layouts or shape, element-count,
///   byte-offset, and alignment errors
///
/// # Execution context
///
/// This conversion performs no synchronization. The caller must establish
/// that the data is ready on [`CudaImport::stream`] before importing it.
impl<T, M> TryFromDlpack<Managed<M>, CudaImport> for ManagedCudaSlice<M, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(
        dlpack: Managed<M>,
        context: CudaImport,
    ) -> Result<Self, Self::Error> {
        build(dlpack, context)
    }
}

fn build<T, M>(dlpack: Managed<M>, context: CudaImport) -> Result<ManagedCudaSlice<M, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    let tensor = dlpack.validate()?;
    let (cu_device_ptr, len, device_id) = validated_cuda_parts::<T>(&tensor)?;
    let stream = context.ready_on;
    let stream_device_id = stream.context().ordinal();
    ensure!(
        stream_device_id == device_id,
        super::StreamDeviceMismatchSnafu {
            tensor_device_id: device_id,
            stream_device_id,
        }
    );

    // SAFETY:
    // - cu_device_ptr is the checked, byte-offset-adjusted DLPack data pointer,
    //   which must reference a valid compact CUDA allocation for at least
    //   `len * size_of::<T>()` bytes.
    // - CudaSliceView::drop calls leak() instead of allowing cudaFree to run.
    let slice = unsafe { stream.upgrade_device_ptr::<T>(cu_device_ptr, len) };

    let view = CudaSliceView(ManuallyDrop::new(slice));
    let inner = unsafe { Borrowed::new_unchecked(dlpack, view) };

    Ok(ManagedCudaSlice { inner })
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

pub(super) fn validated_cuda_parts<T: DlpackElement>(
    tensor: &crate::tensor::TensorRef<'_>,
) -> Result<(u64, usize, usize), Error> {
    ensure!(
        tensor.device().device_type == DLDeviceType::CUDA,
        NotCudaSnafu {
            device_type: tensor.device().device_type
        }
    );
    ensure!(
        tensor.dtype().is::<T>(),
        DtypeMismatchSnafu {
            expected: T::DTYPE,
            actual: tensor.dtype(),
        }
    );
    ensure!(!tensor.data_ptr().is_null(), NullDataSnafu);
    let device_id =
        usize::try_from(tensor.device().device_id).map_err(|_| Error::InvalidDeviceId {
            device_id: tensor.device().device_id,
        })?;

    let len = tensor.num_elements();
    if !tensor.is_compact()? {
        return Err(Error::Tensor {
            source: crate::tensor::Error::NonCompactStrides,
        });
    }
    let cu_device_ptr = unsafe { tensor.offset_data_ptr::<T>()? } as usize as u64;
    Ok((cu_device_ptr, len, device_id))
}
