use snafu::prelude::*;

use crate::utils::MemoryOrder;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Data type mismatch, bits {size}, expect {expected}"))]
    DataTypeSizeMismatch { size: usize, expected: usize },

    #[snafu(display("Shape mismatch, expected {expected:?}, actual {actual:?}"))]
    ShapeMismatch {
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    #[snafu(display("non contiguous"))]
    NonContiguous { shape: Vec<i64>, strides: Vec<i64> },

    #[snafu(display("not supported memory order {order}, expected {expected}"))]
    UnsupportedMemoryOrder {
        order: MemoryOrder,
        expected: MemoryOrder,
    },

    #[snafu(display("unsupported data type {name}"))]
    UnsupportedDataType { name: String },

    #[snafu(display("invalid dimensions, expected {expected}, actual {actual}"))]
    InvalidDimensions { expected: usize, actual: usize },

    #[snafu(display("invalid channels, expected {expected}, actual {actual}"))]
    InvalidChannels { expected: i64, actual: i64 },

}
