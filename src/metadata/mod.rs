//! Shape and stride metadata composed with managed tensor allocations.

use snafu::Snafu;
mod composed;
pub use composed::{Borrowed, Copied, Dynamic, Fixed, PreparedDynamic, PreparedFixed};

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
