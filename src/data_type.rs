use crate::dlpack::{DataType, DataTypeCode};

impl From<(DataTypeCode, u8, u16)> for DataType {
    fn from(value: (DataTypeCode, u8, u16)) -> Self {
        Self {
            code: value.0,
            bits: value.1,
            lanes: value.2,
        }
    }
}
