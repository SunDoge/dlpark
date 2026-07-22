//! Validation, layout inspection, and CPU data access for raw DLPack tensors.

use crate::ffi::{DLDataType, DLDeviceType, DLTensor};
use snafu::Snafu;

mod data;
mod layout;

pub use layout::{compact_strides, compact_strides_array, is_compact_strides};

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

impl DLTensor {
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
    use crate::DlpackElement;
    use crate::ffi::{DLDataTypeCode, DLDevice};
    use std::borrow::Cow;

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
    fn cpu_slice_rejects_non_compact_strides_but_pointer_is_available() {
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
            unsafe { tensor.cpu_slice::<i32>() },
            Err(Error::NonCompactStrides)
        ));
        assert!(matches!(
            unsafe { tensor.cpu_bytes() },
            Err(Error::NonCompactStrides)
        ));
        assert_eq!(
            unsafe { tensor.offset_data_ptr::<i32>() }.unwrap(),
            data.as_ptr()
        );
    }

    #[test]
    fn cpu_bytes_supports_packed_sub_byte_dtype() {
        let data = [0x21u8, 0x03];
        let shape = [3i64];
        let strides = [1i64];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::CPU,
            ndim: 1,
            dtype: DLDataType {
                code: DLDataTypeCode::FLOAT4_E2M1FN,
                bits: 4,
                lanes: 1,
            },
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            ..DLTensor::default()
        };

        assert_eq!(unsafe { tensor.cpu_bytes() }.unwrap(), &data);
    }

    #[test]
    fn cpu_bytes_rejects_non_cpu_tensor() {
        let data = [1u8];
        let shape = [1i64];
        let strides = [1i64];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice::cuda(0),
            ndim: 1,
            dtype: u8::DTYPE,
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            ..DLTensor::default()
        };

        assert!(matches!(
            unsafe { tensor.cpu_bytes() },
            Err(Error::NotCpu { .. })
        ));
    }

    #[test]
    fn offset_pointers_are_device_agnostic_and_apply_byte_offset() {
        let data = [10u8, 20, 30];
        let shape = [2i64];
        let strides = [1i64];
        let tensor = DLTensor {
            data: data.as_ptr().cast_mut().cast(),
            device: DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: u8::DTYPE,
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            byte_offset: 1,
        };

        assert_eq!(tensor.data_ptr(), data.as_ptr().cast());
        assert_eq!(unsafe { tensor.offset_data_ptr::<u8>() }.unwrap(), unsafe {
            data.as_ptr().add(1)
        });
        assert_eq!(unsafe { tensor.offset_bytes_ptr() }.unwrap(), unsafe {
            data.as_ptr().add(1)
        });
        assert!(matches!(
            unsafe { tensor.cpu_slice::<u8>() },
            Err(Error::NotCpu { .. })
        ));
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
