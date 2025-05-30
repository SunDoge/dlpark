use crate::ffi::{self, Tensor};

pub trait TensorLike<L>
where
    L: MemoryLayout,
{
    fn data_ptr(&self) -> *mut std::ffi::c_void;
    fn memory_layout(&self) -> L;
    fn device(&self) -> ffi::Device;
    fn data_type(&self) -> ffi::DataType;
    fn byte_offset(&self) -> u64;
}

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

impl ContiguousLayout {
    pub fn new(shape: Vec<i64>) -> Self {
        Self(shape.into_boxed_slice())
    }

    pub fn new_with_ndim(ndim: usize) -> Self {
        Self(Vec::with_capacity(ndim).into_boxed_slice())
    }

    pub fn shape_mut(&mut self) -> &mut [i64] {
        &mut self.0
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

impl StridedLayout {
    pub fn new(shape_and_strides: Vec<i64>) -> Self {
        Self(shape_and_strides.into_boxed_slice())
    }

    pub fn new_with_ndim(ndim: usize) -> Self {
        Self(Vec::with_capacity(ndim * 2).into_boxed_slice())
    }

    pub fn shape_mut(&mut self) -> &mut [i64] {
        let num_dimensions = self.0.len() / 2;
        &mut self.0[..num_dimensions]
    }

    pub fn strides_mut(&mut self) -> &mut [i64] {
        let num_dimensions = self.0.len() / 2;
        &mut self.0[num_dimensions..]
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

impl BorrowedLayout {
    pub fn new(shape: *mut i64, strides: *mut i64, ndim: i32) -> Self {
        Self {
            shape,
            strides,
            ndim,
        }
    }

    pub fn from_slice(shape: &[i64], strides: Option<&[i64]>) -> Self {
        let num_dimensions = shape.len();
        Self {
            shape: shape.as_ptr() as *mut i64,
            strides: strides
                .map(|s| s.as_ptr() as *mut i64)
                .unwrap_or(std::ptr::null_mut()),
            ndim: num_dimensions as i32,
        }
    }
}

impl Tensor {
    pub fn update<T, L>(&mut self, t: &T, layout: &L)
    where
        T: TensorLike<L>,
        L: MemoryLayout,
    {
        self.data = t.data_ptr();
        self.device = t.device();
        self.dtype = t.data_type();
        self.byte_offset = t.byte_offset();
        self.ndim = layout.ndim();
        self.shape = layout.shape_ptr();
        self.strides = layout.strides_ptr();
    }
}
