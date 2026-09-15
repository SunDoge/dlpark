//! Native device runtime resources used to coordinate DLPack exchange.
//!
//! Container-specific conversions live in [`crate::interop`]. This module
//! contains small native building blocks for code that does not use a tensor
//! container crate.

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
/// Minimal dynamically loaded CUDA Runtime stream and event support.
pub mod cuda;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
/// Shared Metal buffer support for Apple silicon.
pub mod metal;
