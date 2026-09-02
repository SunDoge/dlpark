//! Shape and stride metadata composed with managed tensor allocations.
//!
//! [`Fixed`](crate::metadata::Fixed) composes compile-time-rank shape and strides;
//! [`Dynamic`](crate::metadata::Dynamic) composes runtime-rank shape and strides.
//! Each part is wrapped in either [`Copied`](crate::metadata::Copied)
//! (values copied into the managed allocation; safe `prepare`) or
//! [`Borrowed`](crate::metadata::Borrowed) (caller-owned `i64` storage borrowed
//! zero-copy; unsafe `prepare_unchecked`, because the arrays must outlive the
//! managed tensor).
//!
//! `prepare::<M>()` returns a `PreparedFixed` / `PreparedDynamic`; calling
//! `initialize(ctx)` on it installs the context and deleter and returns an
//! [`crate::allocation::Initialized`]. `Copied` accepts any integer element
//! implementing `TryInto<i64>` (not just `i64`); an `i64` source takes a
//! `TypeId` fast path through `ptr::copy_nonoverlapping`.
//! [`Dynamic::compact`](crate::metadata::Dynamic::compact) computes and stores
//! explicit row-major strides, as required of non-scalar DLPack v1.2+ tensors.

use snafu::Snafu;
mod dynamic;
mod fixed;
mod storage;

pub use dynamic::{Compact, Dynamic, PreparedDynamic};
pub use fixed::{Fixed, PreparedFixed};
pub use storage::{Borrowed, Copied};

/// Errors raised while composing shape and stride metadata with an allocation.
#[derive(Debug, Snafu)]
pub enum Error {
    /// The underlying allocation failed.
    #[snafu(transparent)]
    Allocation {
        /// The underlying allocation error.
        source: crate::allocation::Error,
    },

    /// Shape and strides have different lengths.
    #[snafu(display("Mismatched length of shape ({shape_len}) and strides ({strides_len})"))]
    MismatchedLength {
        /// The number of shape dimensions.
        shape_len: usize,
        /// The number of stride entries.
        strides_len: usize,
    },

    /// The dimension count exceeds `i32::MAX`.
    #[snafu(display("Dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        /// The offending dimension count.
        ndim: usize,
        /// The underlying conversion error.
        source: std::num::TryFromIntError,
    },

    /// A shape value at the given axis does not fit in `i64`.
    #[snafu(display("shape value at axis {axis} does not fit in i64"))]
    ShapeValueOverflow {
        /// The axis of the offending value.
        axis: usize,
    },

    /// A shape value is negative and cannot describe a compact tensor.
    #[snafu(display("shape value at axis {axis} is negative: {value}"))]
    NegativeShapeValue {
        /// The axis of the offending value.
        axis: usize,
        /// The negative dimension.
        value: i64,
    },

    /// Computing compact row-major strides overflowed `i64`.
    #[snafu(display("compact strides overflow i64"))]
    CompactStrideOverflow,

    /// A stride value at the given axis does not fit in `i64`.
    #[snafu(display("stride value at axis {axis} does not fit in i64"))]
    StrideValueOverflow {
        /// The axis of the offending value.
        axis: usize,
    },
}
