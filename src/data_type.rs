use crate::ffi::{DLDataType, DLDataTypeCode};

impl DLDataType {
    pub const FLOAT32: Self = Self {
        code: DLDataTypeCode::FLOAT,
        bits: 32,
        lanes: 1,
    };
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
