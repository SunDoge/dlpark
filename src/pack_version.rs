use crate::ffi::{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION, DLPackVersion};

impl Default for DLPackVersion {
    fn default() -> Self {
        Self {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        }
    }
}
