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
        Self::F32
    }
}

impl DataType {
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
    pub const U8: Self = Self {
        code: DataTypeCode::UInt,
        bits: 8,
        lanes: 1,
    };
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
    pub const BOOL: Self = Self {
        code: DataTypeCode::Bool,
        bits: 8,
        lanes: 1,
    };

    pub fn size(&self) -> usize {
        ((self.bits as u32 * self.lanes as u32 + 7) / 8) as usize
    }
}
