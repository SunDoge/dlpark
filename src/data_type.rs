use crate::ffi::{DLDataType, DLDataTypeCode};

/// Maps a Rust element type to its DLPack data type descriptor.
pub trait DlpackElement: 'static {
    /// The scalar DLPack descriptor for this Rust element type.
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
    /// Constructs a scalar, single-lane data type descriptor.
    pub const fn scalar(code: DLDataTypeCode, bits: u8) -> Self {
        Self {
            code,
            bits,
            lanes: 1,
        }
    }

    /// Returns the descriptor registered for `T`.
    pub const fn of<T: DlpackElement>() -> Self {
        T::DTYPE
    }

    /// Returns whether code, bit width, and lane count all match.
    pub fn matches(&self, other: Self) -> bool {
        self.code == other.code && self.bits == other.bits && self.lanes == other.lanes
    }

    /// Returns whether this descriptor exactly represents `T`.
    pub fn is<T: DlpackElement>(&self) -> bool {
        self.matches(T::DTYPE)
    }

    /// Returns the size of a single element in bytes, rounded up.
    ///
    /// For vectorized types (`lanes > 1`), the total bit width is `bits × lanes`.
    ///
    /// This is only meaningful for byte-aligned dtypes (`bits` a multiple of
    /// 8 — true of every [`DlpackElement`] currently defined in this crate).
    /// For genuinely sub-byte packed dtypes (`bits < 8`, e.g. DLPack's
    /// 4-/6-bit float codes), several elements are packed per byte and this
    /// per-element rounding overcounts — use
    /// [`crate::ffi::DLTensor::num_bytes`] for a whole-tensor byte count
    /// that accounts for packing correctly instead of multiplying this by
    /// the element count.
    pub fn element_size(&self) -> usize {
        let total_bits = (self.bits as usize) * (self.lanes as usize);
        total_bits.div_ceil(8)
    }
}

impl DLDataTypeCode {
    /// Returns whether this data type code is defined by the bundled DLPack
    /// headers.
    pub const fn is_known(self) -> bool {
        const INT: u8 = DLDataTypeCode::INT.0;
        const FLOAT4_E2M1FN: u8 = DLDataTypeCode::FLOAT4_E2M1FN.0;

        matches!(self.0, INT..=FLOAT4_E2M1FN)
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

#[cfg(test)]
mod knownness_tests {
    use super::*;

    #[test]
    fn data_type_code_knownness_tracks_bundled_header() {
        for value in 0..=17 {
            assert!(DLDataTypeCode(value).is_known());
        }
        for value in [18, 99, u8::MAX] {
            assert!(!DLDataTypeCode(value).is_known());
        }
    }
}
