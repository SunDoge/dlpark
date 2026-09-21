use crate::ffi::DLPackVersion;
use snafu::Snafu;

/// The DLPack major version is incompatible with this build.
#[derive(Debug, Snafu)]
#[snafu(display("incompatible DLPack major version: expected {expected}, got {actual}"))]
pub struct VersionError {
    /// The expected major version (from the bundled headers).
    pub expected: u32,
    /// The actual major version declared by the managed tensor.
    pub actual: u32,
}

impl Default for DLPackVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl DLPackVersion {
    /// The DLPack version provided by the bundled headers.
    pub const CURRENT: Self = Self {
        major: crate::ffi::DLPACK_MAJOR_VERSION,
        minor: crate::ffi::DLPACK_MINOR_VERSION,
    };

    /// Returns whether two versions use a compatible ABI.
    pub const fn is_compatible_with(self, other: Self) -> bool {
        self.major == other.major
    }

    /// Ensures that this declared version is ABI-compatible with `expected`.
    pub fn ensure_compatible_with(self, expected: Self) -> Result<(), VersionError> {
        if self.is_compatible_with(expected) {
            Ok(())
        } else {
            Err(VersionError {
                expected: expected.major,
                actual: self.major,
            })
        }
    }

    /// Returns whether this version includes the requested feature level.
    pub const fn supports(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}
