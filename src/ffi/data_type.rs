#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DataTypeCode {
    /// signed integer
    Int = 0,
    /// unsigned integer
    UInt = 1,
    /// IEEE floating point
    Float = 2,
    /// Opaque handle type, reserved for testing purposes.
    /// Frameworks need to agree on the handle data type for the exchange to be
    /// well-defined.
    OpaqueHandle = 3,
    /// bfloat16
    Bfloat = 4,
    /// complex number
    /// (C/C++/Python layout: compact struct per complex number)
    Complex = 5,
    /// boolean
    Bool = 6,
    /// FP8 datatypes
    Float8E3m4 = 7,
    Float8E4m3 = 8,
    Float8E4m3b11fnuz = 9,
    Float8E4m3fn = 10,
    Float8E4m3fnuz = 11,
    Float8E5m2 = 12,
    Float8E5m2fnuz = 13,
    Float8E8m0fnu = 14,

    /// FP6 data types
    Float6E2m3fn = 15,
    Float6E3m2fn = 16,

    Float4E2m1fn = 17,
}

/// The data type the tensor can hold. The data type is assumed to follow the
/// native endian-ness. An explicit error message should be raised when
/// attempting to export an array with non-native endianness
/// Examples
/// - float: type_code = 2, bits = 32, lanes=1
/// - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
/// - int8: type_code = 0, bits = 8, lanes=1
/// - `std::complex<float>`: type_code = 5, bits = 64, lanes = 1
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DataType {
    /// Type code of base types.
    pub code: DataTypeCode,
    /// Number of bits, common choices are 8, 16, 32.
    pub bits: u8,
    /// Number of lanes in the type, used for vector types.
    pub lanes: u16,
}

impl From<(DataTypeCode, u8, u16)> for DataType {
    fn from(value: (DataTypeCode, u8, u16)) -> Self {
        Self {
            code: value.0,
            bits: value.1,
            lanes: value.2,
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        // Most deep learning framework create float32 tensor by default.
        Self::F32
    }
}

impl DataType {
    // Bfloat
    pub const BF16: Self = Self {
        code: DataTypeCode::Bfloat,
        bits: 16,
        lanes: 1,
    };
    // Bool
    pub const BOOL: Self = Self {
        code: DataTypeCode::Bool,
        bits: 8,
        lanes: 1,
    };
    // Float
    pub const F16: Self = Self {
        code: DataTypeCode::Float,
        bits: 16,
        lanes: 1,
    };
    pub const F32: Self = Self {
        code: DataTypeCode::Float,
        bits: 32,
        lanes: 1,
    };
    pub const F64: Self = Self {
        code: DataTypeCode::Float,
        bits: 64,
        lanes: 1,
    };
    pub const I128: Self = Self {
        code: DataTypeCode::Int,
        bits: 128,
        lanes: 1,
    };
    pub const I16: Self = Self {
        code: DataTypeCode::Int,
        bits: 16,
        lanes: 1,
    };
    pub const I32: Self = Self {
        code: DataTypeCode::Int,
        bits: 32,
        lanes: 1,
    };
    pub const I64: Self = Self {
        code: DataTypeCode::Int,
        bits: 64,
        lanes: 1,
    };
    // Int
    pub const I8: Self = Self {
        code: DataTypeCode::Int,
        bits: 8,
        lanes: 1,
    };
    pub const U128: Self = Self {
        code: DataTypeCode::UInt,
        bits: 128,
        lanes: 1,
    };
    pub const U16: Self = Self {
        code: DataTypeCode::UInt,
        bits: 16,
        lanes: 1,
    };
    pub const U32: Self = Self {
        code: DataTypeCode::UInt,
        bits: 32,
        lanes: 1,
    };
    pub const U64: Self = Self {
        code: DataTypeCode::UInt,
        bits: 64,
        lanes: 1,
    };
    // Uint
    pub const U8: Self = Self {
        code: DataTypeCode::UInt,
        bits: 8,
        lanes: 1,
    };
}

impl DataType {
    /// Calculate `DataType` size as (bits * lanes + 7) // 8
    pub fn size(&self) -> usize {
        (self.bits as u32 * self.lanes as u32).div_ceil(8) as usize
    }
}

