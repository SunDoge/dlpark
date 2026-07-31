//! Conversion from externally supplied DLPack tensors.

/// Fallible conversion from an externally supplied DLPack tensor.
///
/// Implementations validate representable descriptor values such as device,
/// dtype, shape, and strides. They cannot validate whether foreign pointers
/// are readable or whether the producer still accesses the underlying data.
///
/// # Stream synchronization
///
/// The `S` parameter carries the producer's stream handle so device-tensor
/// consumers can synchronize without a blocking host call: implementations
/// record a wait on their own stream for the producer's outstanding work
/// (e.g. `consumer_stream.join(producer_stream)` for cudarc). It is *not* the
/// consumer's own stream — the consumer creates that internally — and it is
/// *not* the Python `__dlpack__(stream=)` argument, which flows the other
/// direction and is handled by the [`crate::python`] layer. `S = ()` means no
/// producer stream is available: device-tensor consumers then leave
/// cross-stream synchronization to the caller, while CPU consumers ignore it.
pub trait TryFromDlpack<D, S = ()>: Sized {
    /// The error returned when validation or conversion fails.
    type Error;

    /// Converts `dlpack` into `Self`.
    ///
    /// `stream` is the producer's stream handle (`S = ()` when none is
    /// available). CPU consumers ignore it; device consumers use it to wait
    /// for the producer's outstanding work before exposing the data.
    ///
    /// # Safety
    ///
    /// The managed tensor, its descriptor, and every pointer accessed by the
    /// implementation must satisfy the DLPack memory, lifetime, and
    /// synchronization requirements. When `S = ()` for a device tensor, the
    /// caller must additionally ensure the data is synchronized for the
    /// consumer's device stream. Implementations returning mutable access may
    /// document additional exclusivity requirements.
    unsafe fn try_from_dlpack(dlpack: D, stream: S) -> Result<Self, Self::Error>;
}
