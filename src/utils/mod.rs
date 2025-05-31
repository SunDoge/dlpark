mod memory_order;

pub use memory_order::{
    MemoryOrder, is_column_major_contiguous, is_row_major_contiguous, make_column_major_strides,
    make_row_major_strides,
};
