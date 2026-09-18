//! Conversion from externally supplied DLPack tensors.

/// Fallible conversion from an externally supplied DLPack tensor.
///
/// Implementations validate representable descriptor values such as device,
/// dtype, shape, and strides. They cannot validate whether foreign pointers
/// are readable or whether the producer still accesses the underlying data.
///
/// # Import context
///
/// `C` is a consumer-defined import context. It can carry anything required
/// to construct `Self`, such as a backend client or device context, an
/// allocator, device placement, conversion policy, or synchronization state.
/// A producer stream is a common context for device tensors, but it is not the
/// only supported shape. The same `Self` and `D` may have implementations for
/// multiple context types.
///
/// `C = ()` means that the implementation needs no additional context. Each
/// implementation documents any synchronization and runtime requirements it
/// places on its context.
///
/// This context is intentionally different from Python's
/// `__dlpack__(stream=...)` argument. Python passes a consumer stream to the
/// producer before export so the producer can make the tensor ready on that
/// stream. `TryFromDlpack` runs after an owning DLPack tensor already exists;
/// its context only supplies what the Rust consumer still needs to construct
/// its result. The optional Python interop layer handles the earlier export
/// negotiation.
pub trait TryFromDlpack<D, C = ()>: Sized {
    /// The error returned when validation or conversion fails.
    type Error;

    /// Converts `dlpack` into `Self`.
    ///
    /// The meaning of `context` is defined by the implementation. Device
    /// consumers commonly use it to carry a producer stream, backend client,
    /// allocator, placement choice, or import policy.
    ///
    /// # Safety
    ///
    /// The managed tensor, its descriptor, and every pointer accessed by the
    /// implementation must satisfy the DLPack memory, lifetime, and
    /// synchronization requirements, including all preconditions documented
    /// for `C`. Implementations returning mutable access may document
    /// additional exclusivity requirements.
    unsafe fn try_from_dlpack(dlpack: D, context: C) -> Result<Self, Self::Error>;
}
