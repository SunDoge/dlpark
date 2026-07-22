//! Conversions between DLPack and supported container libraries.
//!
//! Producer-side conversions return allocation-specific initialized values so
//! callers can adjust scalar fields before finishing the managed tensor.
//!
//! # Feature selection
//!
//! | Feature | Producer | Consumer | Data movement |
//! | --- | --- | --- | --- |
//! | `image` | boxed `ImageBuffer` | borrowed or owning `ImageBuffer` | zero-copy |
//! | `ndarray` | boxed owned array | `ArrayViewD` / `ArrayViewMutD` | zero-copy |
//! | `candle` | boxed CPU `Tensor` | owned CPU `Tensor` | export is zero-copy; import copies |
//! | `cudarc` | boxed `CudaSlice` | owning CUDA slice view | zero-copy |
//!
//! The `half` feature adds DLPack element implementations for the `half`
//! crate's 16-bit floating-point types. It is independent of these adapters.
//! Producer conversions require a `Box` because the container itself becomes
//! the stable, type-erased DLPack manager context; the library does not
//! implicitly allocate that box.

#[cfg(feature = "cudarc")]
/// CUDA device-buffer interop through `cudarc`.
pub mod cudarc;

#[cfg(feature = "image")]
/// HWC image interop through the `image` crate.
pub mod image;

#[cfg(feature = "ndarray")]
/// Dynamic-dimensional CPU array interop through `ndarray`.
pub mod ndarray;

#[cfg(feature = "candle")]
/// CPU tensor interop through `candle-core`.
pub mod candle;
