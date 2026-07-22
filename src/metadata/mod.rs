//! Shape and stride metadata composed with managed tensor allocations.
//!
//! [`Fixed`] composes compile-time-rank shape and strides; [`Dynamic`] composes
//! runtime-rank shape and strides. Each part is wrapped in either [`Copied`]
//! (values copied into the managed allocation; safe `prepare`) or
//! [`Borrowed`] (caller-owned `i64` storage borrowed zero-copy; unsafe
//! `prepare_unchecked`, because the arrays must outlive the managed tensor).
//!
//! `prepare::<M>()` returns a `PreparedFixed` / `PreparedDynamic`; calling
//! `initialize(ctx)` on it installs the context and deleter and returns an
//! [`crate::allocation::Initialized`]. `Copied` accepts any integer element
//! implementing `TryInto<i64>` (not just `i64`); an `i64` source takes a
//! `TypeId` fast path through `ptr::copy_nonoverlapping`.

use snafu::Snafu;
mod dynamic;
mod fixed;
mod storage;

pub use dynamic::{Dynamic, PreparedDynamic};
pub use fixed::{Fixed, PreparedFixed};
pub use storage::{Borrowed, Copied};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Allocation { source: crate::allocation::Error },

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
