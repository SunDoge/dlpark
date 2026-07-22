use crate::ffi::DLPackVersion;
use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(display("incompatible DLPack major version: expected {expected}, got {actual}"))]
pub struct VersionError {
    pub expected: u32,
    pub actual: u32,
}

pub(crate) fn validate_version(version: DLPackVersion) -> Result<(), VersionError> {
    if !version.is_compatible_with(DLPackVersion::CURRENT) {
        return Err(VersionError {
            expected: DLPackVersion::CURRENT.major,
            actual: version.major,
        });
    }
    Ok(())
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

    /// Returns whether this version includes the requested feature level.
    pub const fn supports(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl PartialEq for DLPackVersion {
    fn eq(&self, other: &Self) -> bool {
        self.major == other.major && self.minor == other.minor
    }
}

impl Eq for DLPackVersion {}

impl std::hash::Hash for DLPackVersion {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.major, state);
        std::hash::Hash::hash(&self.minor, state);
    }
}
