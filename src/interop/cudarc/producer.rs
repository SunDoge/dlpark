use super::Error;
use crate::{
    DlpackElement, DlpackFlags, ManagedTensorBase,
    allocation::{dynamic, fixed},
    ffi::DLDevice,
    metadata::{Copied, Dynamic, Fixed},
};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::os::raw::c_void;

impl<T: DlpackElement, M: ManagedTensorBase> TryFrom<Box<CudaSlice<T>>>
    for fixed::Initialized<M, 1>
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
        let data_ptr = device_ptr_of(&slice);

        let prepared = Fixed::new(Copied([len]), Copied([1])).prepare::<M>()?;
        let mut initialized = prepared.initialize(slice);
        initialized.set_device(DLDevice::cuda(device_id));
        initialized.set_data(data_ptr);
        initialized.set_dtype(T::DTYPE);

        // SAFETY: the owned CudaSlice was moved into the context and Rust
        // permits no outstanding views while that move occurs.
        initialized.set_flags_unchecked(DlpackFlags::IS_COPIED);
        Ok(initialized)
    }
}

/// Wraps a [`CudaSlice<T>`] in an initialized allocation with explicit metadata.
///
/// - `slice` — owned GPU buffer; ownership is transferred into the DLPack
///   tensor's `manager_ctx` via `Box<CudaSlice<T>>`
/// - `shape`   — dimension sizes in elements (any rank)
/// - `strides` — element strides, must have the same length as `shape`
///
/// # Errors
///
/// - [`crate::metadata::Error::MismatchedLength`] if `shape.len() != strides.len()`
/// - [`crate::metadata::Error::NdimOverflow`] if `shape.len()` overflows `i32`
#[allow(clippy::type_complexity)]
pub fn from_cuda_slice<T: DlpackElement, M: ManagedTensorBase>(
    slice: Box<CudaSlice<T>>,
    shape: &[i64],
    strides: &[i64],
) -> Result<dynamic::Initialized<M>, Error> {
    let device_id = i32::try_from(slice.ordinal()).map_err(|source| Error::DeviceIdOverflow {
        ordinal: slice.ordinal(),
        source,
    })?;
    let data_ptr = device_ptr_of(&slice);
    let prepared = Dynamic::new(Copied(shape), Copied(strides)).prepare::<M>()?;
    let mut initialized = prepared
        .initialize(slice)
        .map_err(crate::metadata::Error::from)?;
    initialized.set_device(DLDevice::cuda(device_id));
    initialized.set_dtype(T::DTYPE);
    initialized.set_data(data_ptr);
    initialized.set_flags_unchecked(DlpackFlags::IS_COPIED);
    Ok(initialized)
}

// ---------------------------------------------------------------------------
// Reverse: Foreign<M> → BorrowedCudaSlice<M, T>
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
