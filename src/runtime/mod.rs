//! Native device runtime resources used to coordinate DLPack exchange.
//!
//! Container-specific conversions live in [`crate::interop`]. This module
//! contains small native building blocks for code that does not use a tensor
//! container crate.

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
/// Minimal dynamically loaded CUDA Runtime stream and event support.
pub mod cuda;
