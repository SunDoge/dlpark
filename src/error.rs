use snafu::prelude::*;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Data type mismatch, bits {size}, expect {expected}"))]
    DataTypeSizeMismatch { size: usize, expected: u8 },

    #[snafu(display("non contiguous"))]
    NonContiguous { shape: Vec<i64>, strides: Vec<i64> },
}
