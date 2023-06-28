use crate::ffi::{PackVersion, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

impl Default for PackVersion {
    fn default() -> Self {
        Self {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        }
    }
}


