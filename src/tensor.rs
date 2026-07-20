//! Validation, layout inspection, and CPU data access for raw DLPack tensors.

use crate::{
    DlpackElement,
    ffi::{DLDataType, DLDevice, DLDeviceType, DLTensor},
};
use snafu::{Snafu, ensure};
use std::{borrow::Cow, mem, os::raw::c_void};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("shape pointer is null but ndim is {ndim}"))]
    NullShapePtr { ndim: i32 },

    #[snafu(display("ndim is negative: {ndim}"))]
    NegativeNdim { ndim: i32 },

    #[snafu(display("shape dimension {axis} is negative: {value}"))]
    NegativeDimension { axis: usize, value: i64 },

    #[snafu(display("number of elements overflows usize"))]
    NumElementsOverflow,

    #[snafu(display("number of bytes overflows usize"))]
    NumBytesOverflow,

    #[snafu(display("shape length ({shape_len}) does not match strides length ({strides_len})"))]
    MismatchedStrides {
        shape_len: usize,
        strides_len: usize,
    },

    #[snafu(display("a contiguous Rust slice requires compact row-major strides"))]
    NonCompactStrides,

    #[snafu(display("tensor must be on CPU to expose a Rust slice, got {device_type:?}"))]
    NotCpu { device_type: DLDeviceType },

    #[snafu(display("dtype mismatch: expected {expected:?}, got {actual:?}"))]
    DtypeMismatch {
        expected: DLDataType,
        actual: DLDataType,
    },

    #[snafu(display("tensor is read-only"))]
    ReadOnly,

    #[snafu(display("tensor is not marked IS_COPIED, so exclusive ownership cannot be proven"))]
    NotCopied,

    #[snafu(display(
        "cannot safely assert IS_COPIED; use the unchecked flag setter if exclusivity is verified"
    ))]
    CannotAssertIsCopied,

    #[snafu(display("tensor data pointer is null for a non-empty tensor"))]
    NullData,

    #[snafu(display("byte_offset {byte_offset} does not fit in usize"))]
    ByteOffsetOverflow { byte_offset: u64 },

    #[snafu(display("data pointer plus byte_offset overflows address space"))]
    DataPointerOverflow,

    #[snafu(display("data pointer {ptr:#x} is not aligned to {align} bytes"))]
    MisalignedData { ptr: usize, align: usize },
}

/// Computes compact row-major strides for a shape.
///
/// Returns an empty vector for scalar tensors.
pub fn compact_strides(shape: &[i64]) -> Result<Vec<i64>, Error> {
    validate_shape_dimensions(shape)?;

    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or(Error::NumElementsOverflow)?;
    }
    Ok(strides)
}

/// Computes compact row-major strides for a fixed-rank shape.
pub fn compact_strides_array<T, const N: usize>(shape: [T; N]) -> Result<[i64; N], Error>
where
    T: Into<i64> + Copy,
{
    let shape = shape.map(Into::into);
    validate_shape_dimensions(&shape)?;

    let mut strides = [0i64; N];
    let mut stride = 1i64;
    for axis in (0..N).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or(Error::NumElementsOverflow)?;
    }
    Ok(strides)
}

/// Returns whether strides describe compact row-major layout.
///
/// A null DLPack strides pointer is represented as `None` and accepted as
/// compact for compatibility with pre-1.2 tensors. New tensors should store
/// explicit strides for non-scalar layouts.
pub fn is_compact_strides(shape: &[i64], strides: Option<&[i64]>) -> Result<bool, Error> {
    validate_shape_dimensions(shape)?;
    let Some(strides) = strides else {
        return Ok(true);
    };
    ensure!(
        shape.len() == strides.len(),
        MismatchedStridesSnafu {
            shape_len: shape.len(),
            strides_len: strides.len()
        }
    );
    if shape.contains(&0) {
        return Ok(true);
    }
    Ok(strides == compact_strides(shape)?.as_slice())
}

fn validate_shape_dimensions(shape: &[i64]) -> Result<(), Error> {
    for (axis, &value) in shape.iter().enumerate() {
        ensure!(value >= 0, NegativeDimensionSnafu { axis, value });
    }
    Ok(())
}

impl Default for DLTensor {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            device: DLDevice::default(),
            ndim: 0,
            dtype: DLDataType::default(),
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        }
    }
}

impl DLTensor {
    /// Returns the shape of the tensor as a slice.
    ///
    /// Returns an empty slice for 0-dimensional tensors (`ndim == 0`).
    ///
    /// # Errors
    ///
    /// - [`Error::NegativeNdim`] if `ndim < 0`.
    /// - [`Error::NullShapePtr`] if `ndim > 0` but `shape` is null.
    ///
    /// # Safety
    ///
    /// For positive `ndim`, `shape` must point to `ndim` initialized `i64`
    /// values that remain readable for the returned slice's lifetime.
    pub unsafe fn shape(&self) -> Result<&[i64], Error> {
        ensure!(self.ndim >= 0, NegativeNdimSnafu { ndim: self.ndim });
        if self.ndim == 0 {
            return Ok(&[]);
        }
        ensure!(!self.shape.is_null(), NullShapePtrSnafu { ndim: self.ndim });
        Ok(unsafe { std::slice::from_raw_parts(self.shape, self.ndim as usize) })
    }

    /// Returns the strides of the tensor as a slice, or `None` for compact row-major layout.
    ///
    /// Per the DLPack spec, a null `strides` pointer indicates a compact row-major (C-contiguous)
    /// layout where strides are implicitly derived from the shape.
    ///
    /// # Errors
    ///
    /// - [`Error::NegativeNdim`] if `ndim < 0`.
    ///
    /// # Safety
    ///
    /// For positive `ndim`, a non-null `strides` pointer must point to `ndim`
    /// initialized `i64` values that remain readable for the returned slice's
    /// lifetime.
    pub unsafe fn strides(&self) -> Result<Option<&[i64]>, Error> {
        ensure!(self.ndim >= 0, NegativeNdimSnafu { ndim: self.ndim });
        if self.strides.is_null() || self.ndim == 0 {
            return Ok(None);
        }
        Ok(Some(unsafe {
            std::slice::from_raw_parts(self.strides, self.ndim as usize)
        }))
    }

    /// Returns explicit strides, or computed compact row-major strides when
    /// DLPack stores them implicitly as a null pointer.
    ///
    /// Explicit strides are borrowed from the tensor. Implicit compact strides
    /// are returned as an owned allocation.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::shape`] and [`compact_strides`].
    ///
    /// # Safety
    ///
    /// The shape and optional strides pointers must satisfy the requirements
    /// of [`Self::shape`] and [`Self::strides`].
    pub unsafe fn strides_or_compact(&self) -> Result<Cow<'_, [i64]>, Error> {
        match unsafe { self.strides()? } {
            Some(strides) => Ok(Cow::Borrowed(strides)),
            None => {
                let shape = unsafe { self.shape()? };
                if shape.is_empty() {
                    Ok(Cow::Borrowed(&[]))
                } else {
                    Ok(Cow::Owned(compact_strides(shape)?))
                }
            }
        }
    }

    /// Returns the total number of elements in the tensor (product of all shape dimensions).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::shape`].
    ///
    /// # Safety
    ///
    /// The shape pointer must satisfy [`Self::shape`]'s requirements.
    pub unsafe fn num_elements(&self) -> Result<usize, Error> {
        let shape = unsafe { self.shape()? };
        validate_shape_dimensions(shape)?;
        shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim as usize)
                .ok_or(Error::NumElementsOverflow)
        })
    }

    /// Returns the total size of the tensor data in bytes.
    ///
    /// Computed as `ceil(num_elements × bits × lanes / 8)` — the ceiling is
    /// taken once over the whole tensor, not per element. For byte-aligned
    /// dtypes (`bits` a multiple of 8, true of every [`DlpackElement`]
    /// currently defined in this crate) this is identical to
    /// `num_elements × dtype.element_size()`. For genuinely sub-byte packed
    /// dtypes (`bits < 8`, e.g. DLPack's 4-/6-bit float codes) it is not:
    /// rounding per element like [`DLDataType::element_size`] would
    /// overcount, since several packed elements share a byte.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::num_elements`].
    ///
    /// # Safety
    ///
    /// The shape pointer must satisfy [`Self::shape`]'s requirements.
    pub unsafe fn num_bytes(&self) -> Result<usize, Error> {
        let bits_per_element = (self.dtype.bits as usize)
            .checked_mul(self.dtype.lanes as usize)
            .ok_or(Error::NumBytesOverflow)?;
        let total_bits = unsafe { self.num_elements()? }
            .checked_mul(bits_per_element)
            .ok_or(Error::NumBytesOverflow)?;
        Ok(total_bits.div_ceil(8))
    }

    /// Returns whether this tensor has compact row-major strides.
    ///
    /// # Safety
    ///
    /// The shape and optional strides pointers must satisfy the requirements
    /// of [`Self::shape`] and [`Self::strides`].
    pub unsafe fn is_compact(&self) -> Result<bool, Error> {
        is_compact_strides(unsafe { self.shape()? }, unsafe { self.strides()? })
    }

    /// Returns the tensor data as a typed Rust slice.
    ///
    /// This is only valid for CPU tensors. Device tensors cannot be safely
    /// represented as host slices and are rejected.
    ///
    /// # Errors
    ///
    /// - [`Error::NotCpu`] if the tensor is not on CPU.
    /// - [`Error::DtypeMismatch`] if `T` does not match `self.dtype`.
    /// - [`Error::NonCompactStrides`] if the tensor is not compact row-major.
    /// - Shape, pointer, offset, and alignment errors if the DLPack metadata
    ///   cannot satisfy Rust slice requirements.
    /// # Safety
    ///
    /// In addition to valid shape and strides metadata, the byte-offset-adjusted
    /// data pointer must reference `num_elements` initialized values of `T`
    /// that remain readable for the returned slice's lifetime.
    pub unsafe fn cpu_data_slice<T: DlpackElement>(&self) -> Result<&[T], Error> {
        ensure!(
            self.device.device_type == DLDeviceType::CPU,
            NotCpuSnafu {
                device_type: self.device.device_type
            }
        );
        ensure!(
            self.dtype.is::<T>(),
            DtypeMismatchSnafu {
                expected: T::DTYPE,
                actual: self.dtype
            }
        );
        ensure!(unsafe { self.is_compact()? }, NonCompactStridesSnafu);

        let num_elements = unsafe { self.num_elements()? };
        if num_elements == 0 {
            return Ok(&[]);
        }

        let data_ptr = unsafe { self.offset_data_ptr::<T>()? };
        Ok(unsafe { std::slice::from_raw_parts(data_ptr, num_elements) })
    }

    /// Returns the byte-offset-adjusted CPU data pointer for typed consumers.
    ///
    /// This validates device, dtype, nullness for non-empty tensors, offset, and
    /// alignment without assuming that the tensor is compact in memory.
    /// # Safety
    ///
    /// The shape metadata must be readable, and for a non-empty tensor the
    /// byte-offset-adjusted data pointer must point to an initialized `T`
    /// element within the backing allocation.
    pub unsafe fn cpu_data_ptr<T: DlpackElement>(&self) -> Result<*const T, Error> {
        ensure!(
            self.device.device_type == DLDeviceType::CPU,
            NotCpuSnafu {
                device_type: self.device.device_type
            }
        );
        ensure!(
            self.dtype.is::<T>(),
            DtypeMismatchSnafu {
                expected: T::DTYPE,
                actual: self.dtype
            }
        );

        if unsafe { self.num_elements()? } == 0 {
            return Ok(std::ptr::NonNull::<T>::dangling().as_ptr());
        }

        unsafe { self.offset_data_ptr::<T>() }
    }

    /// Returns the byte-offset-adjusted CPU data pointer without requiring a
    /// concrete Rust element type.
    ///
    /// Unlike [`Self::cpu_data_ptr`], this does not check `dtype` against any
    /// particular type (there is none to check) and never fails on
    /// alignment — a `u8` pointer is trivially aligned. Useful for callers
    /// that dispatch on `self.dtype` themselves (e.g. against another
    /// library's own dtype enum) instead of a compile-time `T`.
    ///
    /// # Errors
    ///
    /// - [`Error::NotCpu`] if the tensor is not on CPU.
    /// - [`Error::NullData`] if the data pointer is null for a non-empty tensor.
    /// - Errors while applying the tensor's byte offset to its data pointer.
    /// # Safety
    ///
    /// The shape metadata must be readable, and for a non-empty tensor the
    /// byte-offset-adjusted data pointer must lie within the backing
    /// allocation.
    pub unsafe fn cpu_data_ptr_bytes(&self) -> Result<*const u8, Error> {
        ensure!(
            self.device.device_type == DLDeviceType::CPU,
            NotCpuSnafu {
                device_type: self.device.device_type
            }
        );

        if unsafe { self.num_bytes()? } == 0 {
            return Ok(std::ptr::NonNull::<u8>::dangling().as_ptr());
        }

        unsafe { self.offset_data_ptr::<u8>() }
    }

    pub(crate) unsafe fn offset_data_ptr<T>(&self) -> Result<*const T, Error> {
        ensure!(!self.data.is_null(), NullDataSnafu);

        let byte_offset =
            usize::try_from(self.byte_offset).map_err(|_| Error::ByteOffsetOverflow {
                byte_offset: self.byte_offset,
            })?;
        let data = self.data.cast::<u8>();
        let data_addr = data
            .addr()
            .checked_add(byte_offset)
            .ok_or(Error::DataPointerOverflow)?;
        let align = mem::align_of::<T>();
        ensure!(
            data_addr.is_multiple_of(align),
            MisalignedDataSnafu {
                ptr: data_addr,
                align,
            }
        );

        Ok(data.with_addr(data_addr).cast::<T>())
    }

    pub fn data_ptr(&self) -> *const c_void {
        self.data as *const c_void
    }

    pub(crate) fn from_parts(shape: *mut i64, strides: *mut i64, ndim: i32) -> Self {
        Self {
            ndim,
            shape,
            strides,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_or_compact_borrows_explicit_strides() {
        let shape = [2i64, 3];
        let strides = [10i64, 2];
        let tensor = DLTensor {
            ndim: 2,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ..DLTensor::default()
        };

        let actual = unsafe { tensor.strides_or_compact() }.unwrap();

        assert!(matches!(actual, Cow::Borrowed(_)));
        assert_eq!(&*actual, &[10, 2]);
    }

    #[test]
    fn strides_or_compact_computes_implicit_compact_strides() {
        let shape = [2i64, 3, 4];
        let tensor = DLTensor {
            ndim: 3,
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            ..DLTensor::default()
        };

        let actual = unsafe { tensor.strides_or_compact() }.unwrap();

        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(&*actual, &[12, 4, 1]);
    }

    #[test]
    fn strides_or_compact_keeps_scalar_strides_empty() {
        let tensor = DLTensor::default();

        let actual = unsafe { tensor.strides_or_compact() }.unwrap();

        assert!(matches!(actual, Cow::Borrowed(_)));
        assert!(actual.is_empty());
    }

    #[test]
    fn cpu_data_slice_rejects_non_compact_strides_but_pointer_is_available() {
        let data = [1i32, 2, 3, 4];
        let shape = [2i64, 2];
        let strides = [1i64, 2];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::CPU,
            ndim: 2,
            dtype: i32::DTYPE,
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            ..DLTensor::default()
        };

        assert!(matches!(
            unsafe { tensor.cpu_data_slice::<i32>() },
            Err(Error::NonCompactStrides)
        ));
        assert_eq!(
            unsafe { tensor.cpu_data_ptr::<i32>() }.unwrap(),
            data.as_ptr()
        );
    }

    #[test]
    fn empty_tensor_is_compact_regardless_of_strides() {
        let shape = [2i64, 0];

        for strides in [[0i64, 1], [1, 1], [i64::MAX, -1]] {
            assert!(is_compact_strides(&shape, Some(&strides)).unwrap());
        }
    }

    #[test]
    fn empty_tensor_still_validates_metadata() {
        assert!(matches!(
            is_compact_strides(&[2, 0], Some(&[1])),
            Err(Error::MismatchedStrides { .. })
        ));
        assert!(matches!(
            is_compact_strides(&[-1, 0], Some(&[1, 1])),
            Err(Error::NegativeDimension { .. })
        ));
    }
}
