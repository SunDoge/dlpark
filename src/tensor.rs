use crate::ffi::{DLDataType, DLDevice, DLTensor};
use snafu::{Snafu, ensure};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("shape pointer is null but ndim is {ndim}"))]
    NullShapePtr { ndim: i32 },

    #[snafu(display("ndim is negative: {ndim}"))]
    NegativeNdim { ndim: i32 },
}

impl Default for DLTensor {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            device: DLDevice::CPU,
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

    /// Returns the total number of elements in the tensor (product of all shape dimensions).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::shape`].
    pub fn num_elements(&self) -> Result<usize, Error> {
        Ok(self.shape()?.iter().product::<i64>() as usize)
    }

    /// Returns the total size of the tensor data in bytes.
    ///
    /// Computed as `num_elements × dtype.element_size()`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::num_elements`].
    pub fn num_bytes(&self) -> Result<usize, Error> {
        Ok(self.num_elements()? * self.dtype.element_size())
    }
}
