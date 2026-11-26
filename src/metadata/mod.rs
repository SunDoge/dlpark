//! Shape and stride metadata composed with managed tensor allocations.
//!
//! [`Fixed`](crate::metadata::Fixed) composes compile-time-rank shape and strides;
//! [`Dynamic`](crate::metadata::Dynamic) composes runtime-rank shape and strides.
//! The standard constructors copy values into the managed allocation. The
//! explicitly named `borrowed` constructors retain caller-owned `i64` storage
//! and therefore require unsafe `prepare_unchecked`: those arrays must outlive
//! the managed tensor. [`Copied`](crate::metadata::Copied) and
//! [`Borrowed`](crate::metadata::Borrowed) remain available with
//! `with_storage` for mixed or otherwise explicit storage policies.
//!
//! `prepare()` returns a versioned `PreparedFixed` / `PreparedDynamic`; calling
//! `initialize(ctx)` on it installs the context and deleter and returns an
//! [`crate::allocation::Initialized`]. Use `prepare_as::<M>()` when selecting
//! an ABI explicitly. Copied input accepts any integer element
//! implementing `TryInto<i64>` (not just `i64`); an `i64` source takes a
//! `TypeId` fast path through `ptr::copy_nonoverlapping`.
//! [`Dynamic::compact`](crate::metadata::Dynamic::compact) computes and stores
//! explicit row-major strides, as required of non-scalar DLPack v1.2+ tensors.
//!
//! ## Export an owned CPU buffer
//!
//! Preparation stores the shape and strides in the managed allocation.
//! Initialization installs the buffer owner as the context, which the managed
//! tensor's deleter releases.
//!
//! ```
//! use dlpark::{ffi::{DLDataType, DLDevice}, metadata::Fixed, versioned};
//!
//! let prepared = Fixed::new([2, 2], [2, 1]).prepare()?;
//! let mut values = Box::new([1_f32, 2.0, 3.0, 4.0]);
//! let data = values.as_mut_ptr().cast();
//! let mut initialized = prepared.initialize(values);
//! initialized
//!     .set_data(data)
//!     .set_dtype(DLDataType::F32)
//!     .set_device(DLDevice::CPU);
//! // SAFETY: the context owns the array and keeps its address valid.
//! // The shape and strides cover all four elements in row-major order.
//! let tensor: versioned::Dlpack = unsafe { initialized.finish() };
//! assert_eq!(tensor.validate()?.shape(), &[2, 2]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use snafu::Snafu;
mod dynamic;
mod fixed;
mod storage;

pub use dynamic::{Compact, Dynamic, PreparedDynamic};
pub use fixed::{Fixed, PreparedFixed};
pub use storage::{Borrowed, Copied};

/// Errors raised while composing shape and stride metadata with an allocation.
#[derive(Debug, Snafu)]
#[non_exhaustive]
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
