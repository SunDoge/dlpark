use super::*;
use crate::ffi::DLDevice;
use snafu::ensure;
use std::borrow::Cow;

/// Computes compact row-major strides for a shape.
///
/// Returns an empty vector for scalar tensors.
#[inline]
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
#[inline]
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
    /// dtypes (`bits` a multiple of 8, true of every [`crate::DlpackElement`]
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
    #[inline]
    pub unsafe fn is_compact(&self) -> Result<bool, Error> {
        is_compact_strides(unsafe { self.shape()? }, unsafe { self.strides()? })
    }
}
