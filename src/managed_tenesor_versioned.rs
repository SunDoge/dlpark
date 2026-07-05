use crate::ffi::{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION, DLPackVersion};
use bitflags::bitflags;

impl Default for DLPackVersion {
    fn default() -> Self {
        Self {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        }
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DLPackFlags: u64 {
        const READ_ONLY = 1 << 0;
        const IS_COPIED = 1 << 1;
        const IS_SUBBYTE_TYPE_PADDED = 1 << 2;
    }
}
