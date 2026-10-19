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

        let prepared = Fixed::new([len], [1]).prepare_as::<M>()?;
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
/// - `shape`   — dimension sizes in elements (any rank, all non-negative)
/// - `strides` — element strides, must have the same length as `shape`
///
/// The full shape/stride index grid — including negative strides — is
/// validated to lie within `slice.len()` elements, mirroring the span checks
/// of the import direction.
///
/// Returns a no-sync export carrying both the initialized allocation and the
/// stream on which its data becomes ready.
///
/// # Errors
///
/// - [`Error::NegativeDimension`] if a shape dimension is negative
/// - [`Error::StridedSpanOverflow`] if the grid reaches outside the slice
/// - [`crate::metadata::Error::MismatchedLength`] if `shape.len() != strides.len()`
/// - [`crate::metadata::Error::NdimOverflow`] if `shape.len()` overflows `i32`
pub fn from_cuda_slice<T: DlpackElement, M: ManagedTensorBase>(
    slice: Box<CudaSlice<T>>,
    shape: &[i64],
    strides: &[i64],
) -> Result<CudaDlpackExport<dynamic::Initialized<M>>, Error> {
    validate_strided_span(shape, strides, slice.len())?;
    let device_id = i32::try_from(slice.ordinal()).map_err(|source| Error::DeviceIdOverflow {
        ordinal: slice.ordinal(),
        source,
    })?;
    let stream = slice.stream().clone();
    let data_ptr = device_ptr_of(&slice);
    let prepared = Dynamic::new(shape, strides).prepare_as::<M>()?;
    let mut initialized = prepared.initialize(slice);
    initialized.set_device(DLDevice::cuda(device_id));
    initialized.set_dtype(T::DTYPE);
    initialized.set_data(data_ptr);
    Ok(CudaDlpackExport {
        initialized,
        current_stream: stream,
    })
}

/// Proves that the shape/stride index grid lies within `slice_len` elements.
///
/// `shape` must be non-negative; strides may be negative (DLPack permits
/// both). An empty grid (any dimension zero) never touches data, so any
/// strides are accepted — mirroring how `is_compact_strides` treats empty
/// tensors. Otherwise, per-axis min/max element offsets are accumulated in
/// `i128` so negative strides are handled exactly, and the resulting element
/// span must fit inside `[0, slice_len)`.
fn validate_strided_span(
    shape: &[i64],
    strides: &[i64],
    slice_len: usize,
) -> Result<(), Error> {
    let mut total: i128 = 1;
    let mut min_elem: i128 = 0;
    let mut max_elem: i128 = 0;
    for (axis, (&d, &s)) in shape.iter().zip(strides).enumerate() {
        if d < 0 {
            return Err(Error::NegativeDimension { axis, value: d });
        }
        total *= d as i128;
        if d == 0 {
            continue;
        }
        let last = (d - 1) as i128;
        let s = s as i128;
        let (lo, hi) = if s >= 0 { (0, last * s) } else { (last * s, 0) };
        min_elem += lo;
        max_elem += hi;
    }
    if total == 0 {
        return Ok(());
    }
    if min_elem < 0 || max_elem + 1 > slice_len as i128 {
        return Err(Error::StridedSpanOverflow { len: slice_len });
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_validation_accepts_in_bounds_grids() {
        // Compact and strided-within-bounds layouts pass.
        validate_strided_span(&[4], &[1], 4).unwrap();
        validate_strided_span(&[2, 3], &[3, 1], 6).unwrap();
        // A length-1 dimension contributes nothing, so even a negative or
        // huge stride on it stays in bounds.
        validate_strided_span(&[3, 1], &[1, -5], 3).unwrap();
        // Zero-sized dimension touches nothing.
        validate_strided_span(&[0, 5], &[999, 999], 1).unwrap();
        // Zero strides never leave the first element.
        validate_strided_span(&[3, 3], &[0, 0], 1).unwrap();
    }

    #[test]
    fn span_validation_rejects_out_of_bounds_grids() {
        assert!(matches!(
            validate_strided_span(&[1_000_000], &[1], 4),
            Err(Error::StridedSpanOverflow { len: 4 })
        ));
        // Max index (d-1)*s = 2*3 = 6 is one past a 6-element slice.
        assert!(matches!(
            validate_strided_span(&[3, 3], &[3, 1], 6),
            Err(Error::StridedSpanOverflow { len: 6 })
        ));
        // Negative stride reaching before the base pointer.
        assert!(matches!(
            validate_strided_span(&[3], &[-1], 3),
            Err(Error::StridedSpanOverflow { len: 3 })
        ));
    }

    #[test]
    fn span_validation_rejects_negative_dimensions() {
        assert!(matches!(
            validate_strided_span(&[-1], &[1], 4),
            Err(Error::NegativeDimension { axis: 0, value: -1 })
        ));
    }
}
