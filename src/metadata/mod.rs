//! Shape and stride allocation strategies for [`crate::Builder`].
//!
//! Copied metadata shares the managed tensor allocation. Borrowed metadata
//! avoids the copy but requires the caller to uphold its lifetime contract.

use crate::{OpaqueContext, managed_tensor::ManagedTensorBase};
use snafu::Snafu;
use std::{convert::Infallible, ptr::NonNull};

mod allocation;
mod array;
mod borrowed;
mod context;
mod raw;
mod slice;

pub use array::{CopiedArray, GenericArray};
pub use borrowed::{BorrowedArray, BorrowedSlice};
pub use context::FromContext;
pub(crate) use context::try_allocate_generic_from_context;
pub use slice::{CopiedSlice, GenericSlice};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Mismatched length of shape ({shape_len}) and strides ({strides_len})"))]
    MismatchedLength {
        shape_len: usize,
        strides_len: usize,
    },

    #[snafu(display("Dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        ndim: usize,
        source: std::num::TryFromIntError,
    },

    #[snafu(display("shape value at axis {axis} does not fit in i64"))]
    ShapeValueOverflow { axis: usize },

    #[snafu(display("stride value at axis {axis} does not fit in i64"))]
    StrideValueOverflow { axis: usize },
}

/// Allocates and initializes shape and stride storage for a managed tensor.
/// # Safety
///
/// Implementations must return a fully initialized `M` allocation compatible
/// with [`ManagedBox`](crate::ManagedBox). Its deleter must release both the
/// managed tensor and `ctx` exactly once, using the same allocation layout that
/// created the value.
pub unsafe trait Metadata {
    /// Error returned while validating or allocating this metadata.
    type Error;

    /// Validates metadata and allocates a managed tensor containing it.
    fn try_allocate<C, M>(self, ctx: C) -> Result<NonNull<M>, Self::Error>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;

    /// Allocates metadata storage without checking runtime invariants.
    ///
    /// # Safety
    ///
    /// The caller must uphold the metadata type's runtime invariants. For
    /// dynamic metadata this includes matching shape/strides lengths,
    /// `ndim <= i32::MAX`, and readable shape/strides storage for `ndim`
    /// values. Generic metadata additionally requires every value to convert
    /// successfully to `i64`. Violating these requirements may cause
    /// out-of-bounds reads or undefined behavior.
    unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;
}

/// Metadata whose rank and element types make allocation infallible.
/// # Safety
///
/// [`Self::allocate`] must uphold [`Metadata`]'s allocation contract and must
/// not fail for any input accepted by the implementation.
pub unsafe trait InfallibleMetadata: Metadata<Error = Infallible> {
    /// Allocates a managed tensor containing this metadata.
    fn allocate<C, M>(self, ctx: C) -> NonNull<M>
    where
        C: OpaqueContext,
        M: ManagedTensorBase;
}
