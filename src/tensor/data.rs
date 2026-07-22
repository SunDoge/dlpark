use super::*;
use crate::DlpackElement;
use snafu::ensure;
use std::{mem, os::raw::c_void};

impl DLTensor {
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
    pub unsafe fn cpu_slice<T: DlpackElement>(&self) -> Result<&[T], Error> {
        self.ensure_cpu()?;
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

    /// Returns compact CPU tensor data as its raw byte representation.
    ///
    /// Unlike [`Self::cpu_slice`], this does not require a Rust element
    /// type and therefore also supports packed sub-byte dtypes. The slice
    /// length is [`Self::num_bytes`].
    ///
    /// # Errors
    ///
    /// - [`Error::NotCpu`] if the tensor is not on CPU.
    /// - [`Error::NonCompactStrides`] if the tensor is not compact row-major.
    /// - Shape, pointer, and offset errors if the DLPack metadata cannot
    ///   satisfy Rust slice requirements.
    ///
    /// # Safety
    ///
    /// In addition to valid shape and strides metadata, the
    /// byte-offset-adjusted data pointer must reference [`Self::num_bytes`]
    /// initialized bytes that remain readable for the returned slice's
    /// lifetime.
    #[inline]
    pub unsafe fn cpu_bytes(&self) -> Result<&[u8], Error> {
        self.ensure_cpu()?;
        ensure!(unsafe { self.is_compact()? }, NonCompactStridesSnafu);
        let len = unsafe { self.num_bytes()? };
        let data = unsafe { self.offset_bytes_ptr()? };
        Ok(unsafe { std::slice::from_raw_parts(data, len) })
    }

    /// Returns the byte-offset-adjusted data pointer for typed consumers.
    ///
    /// This validates dtype, nullness for non-empty tensors, offset, and
    /// alignment without assuming a device or compact memory layout.
    /// # Safety
    ///
    /// The shape metadata must be readable, and for a non-empty tensor the
    /// byte-offset-adjusted address must lie within the device allocation.
    pub unsafe fn offset_data_ptr<T: DlpackElement>(&self) -> Result<*const T, Error> {
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

        unsafe { self.offset_ptr::<T>() }
    }

    /// Returns the byte-offset-adjusted data pointer without requiring a
    /// concrete Rust element type.
    ///
    /// Unlike [`Self::offset_data_ptr`], this does not check `dtype` against any
    /// particular type (there is none to check) and never fails on
    /// alignment — a `u8` pointer is trivially aligned. Useful for callers
    /// that dispatch on `self.dtype` themselves (e.g. against another
    /// library's own dtype enum) instead of a compile-time `T`.
    ///
    /// # Errors
    ///
    /// - [`Error::NullData`] if the data pointer is null for a non-empty tensor.
    /// - Errors while applying the tensor's byte offset to its data pointer.
    /// # Safety
    ///
    /// The shape metadata must be readable, and for a non-empty tensor the
    /// byte-offset-adjusted address must lie within the device allocation.
    pub unsafe fn offset_bytes_ptr(&self) -> Result<*const u8, Error> {
        if unsafe { self.num_bytes()? } == 0 {
            return Ok(std::ptr::NonNull::<u8>::dangling().as_ptr());
        }

        unsafe { self.offset_ptr::<u8>() }
    }

    unsafe fn offset_ptr<T>(&self) -> Result<*const T, Error> {
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

    #[inline]
    pub fn data_ptr(&self) -> *const c_void {
        self.data as *const c_void
    }

    pub(crate) fn ensure_cpu(&self) -> Result<(), Error> {
        ensure!(
            self.device.device_type == DLDeviceType::CPU,
            NotCpuSnafu {
                device_type: self.device.device_type
            }
        );
        Ok(())
    }
}
