//! Conversion from externally supplied DLPack tensors.

/// Fallible conversion from an externally supplied DLPack tensor.
///
/// Implementations validate representable descriptor values such as device,
/// dtype, shape, and strides. They cannot validate whether foreign pointers
/// are readable or whether the producer still accesses the underlying data.
pub trait TryFromDlpack<D>: Sized {
    type Error;

    /// Converts `dlpack` into `Self`.
    ///
    /// # Safety
    ///
    /// The managed tensor, its descriptor, and every pointer accessed by the
    /// implementation must satisfy the DLPack memory, lifetime, and
    /// synchronization requirements. Implementations returning mutable access
    /// may document additional exclusivity requirements.
    unsafe fn try_from_dlpack(dlpack: D) -> Result<Self, Self::Error>;
}
