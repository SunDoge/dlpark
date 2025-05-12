pub const DLPACK_MAJOR_VERSION: u32 = 1;
pub const DLPACK_MINOR_VERSION: u32 = 1;

/// The DLPack version.
#[repr(C)]
#[derive(Debug)]
pub struct PackVersion {
    /// DLPack major version.
    pub major: u32,
    /// DLPack minor version.
    pub minor: u32,
}

impl Default for PackVersion {
    fn default() -> Self {
        Self {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        }
    }
}

impl PackVersion {
    /// Create a new `PackVersion` instance.
    pub fn new(major: u32, minor: u32) -> Self {
        PackVersion { major, minor }
    }
}
