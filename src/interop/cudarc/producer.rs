use super::Error;
use crate::{
    DlpackElement, ManagedTensorBase,
    allocation::{dynamic, fixed},
    ffi::DLDevice,
    metadata::{Dynamic, Fixed},
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::{os::raw::c_void, sync::Arc};

/// A no-sync CUDA DLPack export and the stream carrying its current work.
///
/// The initialized tensor does not encode stream state. Consumers must either
/// continue work on [`Self::current_stream`] or establish an ordering edge to
/// another stream before importing it there.
pub struct CudaDlpackExport<I> {
    initialized: I,
    current_stream: Arc<CudaStream>,
}

impl<I> CudaDlpackExport<I> {
    /// Borrows the initialized DLPack allocation.
    pub fn initialized(&self) -> &I {
        &self.initialized
    }

    /// Mutably borrows the initialized allocation for configuring fields such
    /// as versioned flags before finishing the managed tensor.
    pub fn initialized_mut(&mut self) -> &mut I {
        &mut self.initialized
    }

    /// Returns the stream on which the exported allocation's data becomes ready.
    pub fn current_stream(&self) -> &Arc<CudaStream> {
        &self.current_stream
    }

    /// Splits the initialized tensor from its current work stream.
    pub fn into_parts(self) -> (I, Arc<CudaStream>) {
        (self.initialized, self.current_stream)
    }
}

impl<T: DlpackElement, M: ManagedTensorBase> TryFrom<Box<CudaSlice<T>>>
    for CudaDlpackExport<fixed::Initialized<M, 1>>
{
    type Error = Error;

    fn try_from(slice: Box<CudaSlice<T>>) -> Result<Self, Self::Error> {
        let len = i64::try_from(slice.len()).map_err(|source| Error::LengthOverflow {
            len: slice.len(),
            source,
        })?;
        let device_id =
            i32::try_from(slice.ordinal()).map_err(|source| Error::DeviceIdOverflow {
                ordinal: slice.ordinal(),
                source,
            })?;
        let current_stream = slice.stream().clone();
        let data_ptr = device_ptr_of(&slice);

        let prepared = Fixed::new([len], [1]).prepare::<M>()?;
        let mut initialized = prepared.initialize(slice);
        initialized.set_device(DLDevice::cuda(device_id));
        initialized.set_data(data_ptr);
        initialized.set_dtype(T::DTYPE);
        Ok(CudaDlpackExport {
            initialized,
            current_stream,
        })
    }
}

/// Wraps a [`CudaSlice<T>`] in an initialized allocation with explicit metadata.
///
/// - `slice` — owned GPU buffer; ownership is transferred into the DLPack
///   tensor's `manager_ctx` via `Box<CudaSlice<T>>`
/// - `shape`   — dimension sizes in elements (any rank)
/// - `strides` — element strides, must have the same length as `shape`
///
/// Returns a no-sync export carrying both the initialized allocation and the
/// stream on which its data becomes ready.
///
/// # Errors
///
/// - [`crate::metadata::Error::MismatchedLength`] if `shape.len() != strides.len()`
/// - [`crate::metadata::Error::NdimOverflow`] if `shape.len()` overflows `i32`
pub fn from_cuda_slice<T: DlpackElement, M: ManagedTensorBase>(
    slice: Box<CudaSlice<T>>,
    shape: &[i64],
    strides: &[i64],
) -> Result<CudaDlpackExport<dynamic::Initialized<M>>, Error> {
    let device_id = i32::try_from(slice.ordinal()).map_err(|source| Error::DeviceIdOverflow {
        ordinal: slice.ordinal(),
        source,
    })?;
    let stream = slice.stream().clone();
    let data_ptr = device_ptr_of(&slice);
    let prepared = Dynamic::new(shape, strides).prepare::<M>()?;
    let mut initialized = prepared.initialize(slice);
    initialized.set_device(DLDevice::cuda(device_id));
    initialized.set_dtype(T::DTYPE);
    initialized.set_data(data_ptr);
    Ok(CudaDlpackExport {
        initialized,
        current_stream: stream,
    })
}

// ---------------------------------------------------------------------------
// Reverse: Managed<M> → ManagedCudaSlice<M, T>
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
