#[derive(Debug, Eq, PartialEq)]
pub enum MemoryOrder {
    RowMajorContiguous,
    ColumnMajorContiguous,
    NonContiguous,
}

pub fn make_row_major_strides(shape: &[i64]) -> Vec<i64> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn make_column_major_strides(shape: &[i64]) -> Vec<i64> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in 0..rank - 1 {
        strides[i + 1] = strides[i] * shape[i];
    }
    strides
}

impl std::fmt::Display for MemoryOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryOrder::RowMajorContiguous => write!(f, "RowMajorContiguous"),
            MemoryOrder::ColumnMajorContiguous => write!(f, "ColumnMajorContiguous"),
            MemoryOrder::NonContiguous => write!(f, "NonContiguous"),
        }
    }
}

impl MemoryOrder {
    pub fn new(shape: &[i64], strides: &[i64]) -> Self {
        if is_row_major_contiguous(shape, strides) {
            MemoryOrder::RowMajorContiguous
        } else if is_column_major_contiguous(shape, strides) {
            MemoryOrder::ColumnMajorContiguous
        } else {
            MemoryOrder::NonContiguous
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            MemoryOrder::RowMajorContiguous | MemoryOrder::ColumnMajorContiguous => true,
            MemoryOrder::NonContiguous => false,
        }
    }
}

pub fn is_row_major_contiguous(shape: &[i64], strides: &[i64]) -> bool {
    let mut expected = 1;
    for (dim_size, stride) in shape.iter().zip(strides.iter()).rev() {
        if *dim_size == 0 {
            continue;
        }
        if *stride != expected {
            return false;
        }
        expected *= dim_size;
    }
    true
}

pub fn is_column_major_contiguous(shape: &[i64], strides: &[i64]) -> bool {
    let mut expected = 1;
    for (dim_size, stride) in shape.iter().zip(strides.iter()) {
        if *dim_size == 0 {
            continue;
        }
        if *stride != expected {
            return false;
        }
        expected *= dim_size;
    }
    true
}

// Generated by copilot.
#[cfg(test)]
mod tests {
    use super::*;

    // test make_contiguous_strides
    #[test]
    fn test_row_major_strides() {
        let shape = vec![1, 2, 3];
        let strides = make_row_major_strides(&shape);
        assert_eq!(strides, vec![6, 3, 1]);
        assert!(is_row_major_contiguous(&shape, &strides));
    }

    // test make_column_major_strides
    #[test]
    fn test_column_major_strides() {
        let shape = vec![1, 2, 3];
        let strides = make_column_major_strides(&shape);
        assert_eq!(strides, vec![1, 1, 2]);
        assert!(is_column_major_contiguous(&shape, &strides));
    }
}
