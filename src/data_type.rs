use crate::ffi::{DLDataType, DLDataTypeCode};

/// Maps a Rust element type to its DLPack data type descriptor.
pub trait DlpackElement: 'static {
    const DTYPE: DLDataType;
}

macro_rules! impl_dlpack_element {
    ($ty:ty, $code:expr, $bits:expr) => {
        impl DlpackElement for $ty {
            const DTYPE: DLDataType = DLDataType {
                code: $code,
                bits: $bits,
                lanes: 1,
            };
        }
    };
}

impl_dlpack_element!(bool, DLDataTypeCode::BOOL, 8);

impl_dlpack_element!(i8, DLDataTypeCode::INT, 8);
impl_dlpack_element!(i16, DLDataTypeCode::INT, 16);
impl_dlpack_element!(i32, DLDataTypeCode::INT, 32);
impl_dlpack_element!(i64, DLDataTypeCode::INT, 64);

impl_dlpack_element!(u8, DLDataTypeCode::UINT, 8);
impl_dlpack_element!(u16, DLDataTypeCode::UINT, 16);
impl_dlpack_element!(u32, DLDataTypeCode::UINT, 32);
impl_dlpack_element!(u64, DLDataTypeCode::UINT, 64);

impl_dlpack_element!(f32, DLDataTypeCode::FLOAT, 32);
impl_dlpack_element!(f64, DLDataTypeCode::FLOAT, 64);

#[cfg(feature = "half")]
impl_dlpack_element!(half::f16, DLDataTypeCode::FLOAT, 16);

#[cfg(feature = "half")]
impl_dlpack_element!(half::bf16, DLDataTypeCode::BFLOAT, 16);

impl DLDataType {
    pub const fn scalar(code: DLDataTypeCode, bits: u8) -> Self {
        Self {
            code,
            bits,
            lanes: 1,
        }
    }

    pub const FLOAT32: Self = Self {
        code: DLDataTypeCode::FLOAT,
        bits: 32,
        lanes: 1,
    };

    pub const fn of<T: DlpackElement>() -> Self {
        T::DTYPE
    }

    pub fn matches(&self, other: Self) -> bool {
        self.code == other.code && self.bits == other.bits && self.lanes == other.lanes
    }

    pub fn is<T: DlpackElement>(&self) -> bool {
        self.matches(T::DTYPE)
    }

    /// Returns the size of a single element in bytes.
    ///
    /// Handles sub-byte types (e.g. INT4 with `bits = 4`) by rounding up to the nearest byte.
    /// For vectorized types (`lanes > 1`), the total bit width is `bits × lanes`.
    pub fn element_size(&self) -> usize {
        let total_bits = (self.bits as usize) * (self.lanes as usize);
        total_bits.div_ceil(8)
    }
}

impl Default for DLDataType {
    fn default() -> Self {
        Self {
            code: DLDataTypeCode(0),
            bits: 0,
            lanes: 0,
        }
    }
}
