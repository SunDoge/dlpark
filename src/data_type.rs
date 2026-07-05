use crate::ffi::{DLDataType, DLDataTypeCode};

impl DLDataType {
    pub const FLOAT32: Self = Self {
        code: DLDataTypeCode::FLOAT,
        bits: 32,
        lanes: 1,
    };

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
