pub trait MemoryLayout {
    fn shape_ptr(&self) -> *mut i64;
    fn strides_ptr(&self) -> *mut i64;
    fn ndim(&self) -> i32;
}

pub struct ContiguousLayout(Box<[i64]>);
pub struct StridedLayout(Box<[i64]>);
pub struct BorrowedLayout {
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
}

impl MemoryLayout for ContiguousLayout {
    fn shape_ptr(&self) -> *mut i64 {
        self.0.as_ptr() as *mut i64
    }

    fn strides_ptr(&self) -> *mut i64 {
        std::ptr::null_mut()
    }

    fn ndim(&self) -> i32 {
        self.0.len() as i32
    }
}

impl MemoryLayout for StridedLayout {
    fn shape_ptr(&self) -> *mut i64 {
        self.0.as_ptr() as *mut i64
    }

    fn strides_ptr(&self) -> *mut i64 {
        unsafe { self.0.as_ptr().add(self.0.len() / 2) as *mut i64 }
    }

    fn ndim(&self) -> i32 {
        self.0.len() as i32 / 2
    }
}

impl MemoryLayout for BorrowedLayout {
    fn shape_ptr(&self) -> *mut i64 {
        self.shape
    }

    fn strides_ptr(&self) -> *mut i64 {
        self.strides
    }

    fn ndim(&self) -> i32 {
        self.ndim
    }
}
