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

    #[snafu(display("tensor must be on CPU to expose a Rust slice, got {device_type:?}"))]
    NotCpu { device_type: DLDeviceType },

    #[snafu(display("dtype mismatch: expected {expected:?}, got {actual:?}"))]
    DtypeMismatch {
        expected: DLDataType,
        actual: DLDataType,
    },

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
    pub fn shape(&self) -> Result<&[i64], Error> {
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
    pub fn strides(&self) -> Result<Option<&[i64]>, Error> {
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
    pub fn strides_or_compact(&self) -> Result<Cow<'_, [i64]>, Error> {
        match self.strides()? {
            Some(strides) => Ok(Cow::Borrowed(strides)),
            None => {
                let shape = self.shape()?;
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
    pub fn num_elements(&self) -> Result<usize, Error> {
        let shape = self.shape()?;
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
    pub fn num_bytes(&self) -> Result<usize, Error> {
        let bits_per_element = (self.dtype.bits as usize)
            .checked_mul(self.dtype.lanes as usize)
            .ok_or(Error::NumBytesOverflow)?;
        let total_bits = self
            .num_elements()?
            .checked_mul(bits_per_element)
            .ok_or(Error::NumBytesOverflow)?;
        Ok(total_bits.div_ceil(8))
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
    /// - Shape, pointer, offset, and alignment errors if the DLPack metadata
    ///   cannot satisfy Rust slice requirements.
    pub fn cpu_data_slice<T: DlpackElement>(&self) -> Result<&[T], Error> {
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

        let num_elements = self.num_elements()?;
        if num_elements == 0 {
            return Ok(&[]);
        }

        let data_ptr = self.offset_data_ptr::<T>()?;
        Ok(unsafe { std::slice::from_raw_parts(data_ptr, num_elements) })
    }

    /// Returns the byte-offset-adjusted CPU data pointer for typed consumers.
    ///
    /// This validates device, dtype, nullness for non-empty tensors, offset, and
    /// alignment without assuming that the tensor is compact in memory.
    pub fn cpu_data_ptr<T: DlpackElement>(&self) -> Result<*const T, Error> {
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

        if self.num_elements()? == 0 {
            return Ok(std::ptr::NonNull::<T>::dangling().as_ptr());
        }

        self.offset_data_ptr::<T>()
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
    /// - Offset arithmetic errors from [`Self::offset_data_ptr`].
    pub fn cpu_data_ptr_bytes(&self) -> Result<*const u8, Error> {
        ensure!(
            self.device.device_type == DLDeviceType::CPU,
            NotCpuSnafu {
                device_type: self.device.device_type
            }
        );

        if self.num_bytes()? == 0 {
            return Ok(std::ptr::NonNull::<u8>::dangling().as_ptr());
        }

        self.offset_data_ptr::<u8>()
    }

    fn offset_data_ptr<T>(&self) -> Result<*const T, Error> {
        ensure!(!self.data.is_null(), NullDataSnafu);

        let byte_offset =
            usize::try_from(self.byte_offset).map_err(|_| Error::ByteOffsetOverflow {
                byte_offset: self.byte_offset,
            })?;
        let data_addr = (self.data as usize)
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

        Ok(data_addr as *const T)
    }

    pub fn data_ptr(&self) -> *const c_void {
        self.data as *const c_void
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

        let actual = tensor.strides_or_compact().unwrap();

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

        let actual = tensor.strides_or_compact().unwrap();

        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(&*actual, &[12, 4, 1]);
    }

    #[test]
    fn strides_or_compact_keeps_scalar_strides_empty() {
        let tensor = DLTensor::default();

        let actual = tensor.strides_or_compact().unwrap();

        assert!(matches!(actual, Cow::Borrowed(_)));
        assert!(actual.is_empty());
    }
}
