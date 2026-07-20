//! Conversions between DLPack and supported container libraries.
//!
//! Producer-side conversions return [`crate::Builder`] so callers can adjust
//! flags and metadata before selecting a managed tensor ABI.

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
