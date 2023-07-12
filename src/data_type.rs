use crate::ffi::{DataType, DataTypeCode};

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

    // Uint
    pub const U8: Self = Self {
        code: DataTypeCode::UInt,
        bits: 8,
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
    pub const U128: Self = Self {
        code: DataTypeCode::UInt,
        bits: 128,
        lanes: 1,
    };

    // Int
    pub const I8: Self = Self {
        code: DataTypeCode::Int,
        bits: 8,
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
    pub const I128: Self = Self {
        code: DataTypeCode::Int,
        bits: 128,
        lanes: 1,
    };

    // Bool
    pub const BOOL: Self = Self {
        code: DataTypeCode::Bool,
        bits: 8,
        lanes: 1,
    };

    // Bfloat
    pub const BF16: Self = Self {
        code: DataTypeCode::Bfloat,
        bits: 16,
        lanes: 1,
    };

    /// Calculate `DataType` size as (bits * lanes + 7) // 8
    pub fn size(&self) -> usize {
        ((self.bits as u32 * self.lanes as u32 + 7) / 8) as usize
    }
}
