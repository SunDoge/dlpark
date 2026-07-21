use crate::ffi::{DLDataType, DLDataTypeCode};

/// Maps a Rust element type to its DLPack data type descriptor.
///
/// # Safety
///
/// `Self` must have exactly the size and alignment described by [`Self::DTYPE`],
/// and every initialized bit pattern permitted by that DLPack dtype must be a
/// valid value of `Self`. The dtype must be scalar (`lanes == 1`) and
/// byte-aligned.
pub unsafe trait DlpackElement: 'static {
    /// The scalar DLPack descriptor for this Rust element type.
    const DTYPE: DLDataType;
}

macro_rules! impl_dlpack_element {
    ($ty:ty, $code:expr, $bits:expr) => {
        unsafe impl DlpackElement for $ty {
            const DTYPE: DLDataType = DLDataType {
                code: $code,
                bits: $bits,
                lanes: 1,
            };
        }
    };
}

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

macro_rules! impl_data_type {
    ($name:ident, $code:expr, $bits:expr) => {
        impl DLDataType {
            #[doc = concat!("The scalar DLPack `", stringify!($name), "` data type.")]
            pub const $name: Self = Self {
                code: $code,
                bits: $bits,
                lanes: 1,
            };
        }
    };
}

impl_data_type!(BOOL, DLDataTypeCode::BOOL, 8);
impl_data_type!(I8, DLDataTypeCode::INT, 8);
impl_data_type!(I16, DLDataTypeCode::INT, 16);
impl_data_type!(I32, DLDataTypeCode::INT, 32);
impl_data_type!(I64, DLDataTypeCode::INT, 64);
impl_data_type!(U8, DLDataTypeCode::UINT, 8);
impl_data_type!(U16, DLDataTypeCode::UINT, 16);
impl_data_type!(U32, DLDataTypeCode::UINT, 32);
impl_data_type!(U64, DLDataTypeCode::UINT, 64);
impl_data_type!(F16, DLDataTypeCode::FLOAT, 16);
impl_data_type!(F32, DLDataTypeCode::FLOAT, 32);
impl_data_type!(F64, DLDataTypeCode::FLOAT, 64);
impl_data_type!(BF16, DLDataTypeCode::BFLOAT, 16);
impl_data_type!(C64, DLDataTypeCode::COMPLEX, 64);
impl_data_type!(C128, DLDataTypeCode::COMPLEX, 128);
impl_data_type!(F8E3M4, DLDataTypeCode::FLOAT8_E3M4, 8);
impl_data_type!(F8E4M3, DLDataTypeCode::FLOAT8_E4M3, 8);
impl_data_type!(F8E4M3B11FNUZ, DLDataTypeCode::FLOAT8_E4M3B11FNUZ, 8);
impl_data_type!(F8E4M3FN, DLDataTypeCode::FLOAT8_E4M3FN, 8);
impl_data_type!(F8E4M3FNUZ, DLDataTypeCode::FLOAT8_E4M3FNUZ, 8);
impl_data_type!(F8E5M2, DLDataTypeCode::FLOAT8_E5M2, 8);
impl_data_type!(F8E5M2FNUZ, DLDataTypeCode::FLOAT8_E5M2FNUZ, 8);
impl_data_type!(F8E8M0FNU, DLDataTypeCode::FLOAT8_E8M0FNU, 8);
impl_data_type!(F6E2M3FN, DLDataTypeCode::FLOAT6_E2M3FN, 6);
impl_data_type!(F6E3M2FN, DLDataTypeCode::FLOAT6_E3M2FN, 6);
impl_data_type!(F4E2M1FN, DLDataTypeCode::FLOAT4_E2M1FN, 4);

impl DLDataType {
    pub fn new(code: DLDataTypeCode, bits: u8, lanes: u16) -> Self {
        Self { code, bits, lanes }
    }

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

    #[test]
    fn bool_descriptor_uses_the_standard_storage_width() {
        assert_eq!(DLDataType::BOOL.code, DLDataTypeCode::BOOL);
        assert_eq!(DLDataType::BOOL.bits, 8);
        assert_eq!(DLDataType::BOOL.lanes, 1);
    }

    #[test]
    fn primitive_helpers_match_dlpack_elements() {
        assert!(DLDataType::I8.is::<i8>());
        assert!(DLDataType::I16.is::<i16>());
        assert!(DLDataType::I32.is::<i32>());
        assert!(DLDataType::I64.is::<i64>());
        assert!(DLDataType::U8.is::<u8>());
        assert!(DLDataType::U16.is::<u16>());
        assert!(DLDataType::U32.is::<u32>());
        assert!(DLDataType::U64.is::<u64>());
        assert!(DLDataType::F32.is::<f32>());
        assert!(DLDataType::F64.is::<f64>());

        #[cfg(feature = "half")]
        {
            assert!(DLDataType::F16.is::<half::f16>());
            assert!(DLDataType::BF16.is::<half::bf16>());
        }
    }
}
