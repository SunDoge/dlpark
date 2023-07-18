mod data_type;
mod device;
mod dl_managed_tensor;
mod dl_managed_tensor_versioned;
mod dl_tensor;
mod manager_ctx;
mod pack_version;
mod tensor;

#[cfg(feature = "pyo3")]
mod python;

/// Raw bindings for DLPack.
pub mod ffi;

/// Imports the structs and traits for you to implement [`IntoDLPack`] and [`FromDLPack`].
pub mod prelude;

pub use crate::manager_ctx::{CowIntArray, ManagerCtx};
pub use crate::tensor::traits::{DLPack, FromDLPack, InferDtype, IntoDLPack, TensorView, ToTensor};
pub use crate::tensor::ManagedTensor;
