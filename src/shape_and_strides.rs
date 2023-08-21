use std::ptr::NonNull;

/// If the shape or strides of Tensor is vec of i64, then it should be borrowed to avoid copy.
/// The lifetime should be 'static since we don't managed its memory.
/// Otherwise, we should copy the data and convert its type to i64 and managed it ourselves.
#[derive(Debug)]
pub enum ShapeAndStrides {
    Contiguous(Box<[i64]>),  // Shape only
    WithStrides(Box<[i64]>), // [Shape | Strides]
    Borrowed {
        shape: NonNull<i64>,
        strides: Option<NonNull<i64>>,
        len: usize,
    },
}

impl ShapeAndStrides {
    pub fn new_contiguous<'a, I>(shape: I) -> Self
    where
        I: IntoIterator<Item = &'a i64>,
    {
        let buf: Vec<i64> = shape.into_iter().copied().collect();
        Self::Contiguous(buf.into_boxed_slice())
    }

    pub fn new_with_strides<'a, I>(shape: I, strides: I) -> Self
    where
        I: IntoIterator<Item = &'a i64>,
    {
        let shape: Vec<&i64> = shape.into_iter().collect();
        let strides: Vec<&i64> = strides.into_iter().collect();
        assert_eq!(shape.len(), strides.len());
        let mut buf: Vec<i64> = Vec::with_capacity(shape.len() + strides.len());
        buf.extend(shape);
        buf.extend(strides);
        Self::WithStrides(buf.into_boxed_slice())
    }

    pub fn new_contiguous_with_strides<'a, I>(shape: I) -> Self
    where
        I: IntoIterator<Item = &'a i64>,
    {
        let shape: Vec<&i64> = shape.into_iter().collect();
        let len = shape.len();
        let mut buf: Vec<i64> = vec![0; len * 2];
        let mut stride = 1;
        for i in (0..len).rev() {
            buf[i] = *shape[i];
            buf[i + len] = stride;
            stride *= buf[i];
        }
        Self::WithStrides(buf.into_boxed_slice())
    }

    pub fn new_borrowed(shape: &[i64], strides: Option<&[i64]>) -> Self {
        if let Some(ref strides) = strides {
            assert_eq!(shape.len(), strides.len());
        }
        let len = shape.len();
        let shape = NonNull::from(&shape[0]);
        let strides = strides.map(|s| NonNull::from(&s[0]));
        Self::Borrowed {
            shape,
            strides,
            len,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Contiguous(ref v) => v.len(),
            Self::WithStrides(ref v) => v.len() / 2,
            Self::Borrowed { len, .. } => *len,
        }
    }

    pub fn ndim(&self) -> i32 {
        self.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn shape(&self) -> &[i64] {
        match self {
            Self::Contiguous(ref v) => v.as_ref(),
            Self::WithStrides(ref v) => &v[0..self.len()],
            Self::Borrowed { shape, .. } => unsafe {
                std::slice::from_raw_parts(shape.as_ptr(), self.len())
            },
        }
    }

    pub fn shape_ptr(&self) -> *mut i64 {
        match self {
            Self::Contiguous(ref v) => v.as_ptr() as *mut i64,
            Self::WithStrides(ref v) => v.as_ptr() as *mut i64,
            Self::Borrowed { shape, .. } => shape.as_ptr(),
        }
    }

    pub fn strides(&self) -> Option<&[i64]> {
        match self {
            Self::Contiguous(_) => None,
            Self::WithStrides(ref v) => Some(&v[self.len()..]),
            Self::Borrowed { strides, .. } => {
                strides.map(|s| unsafe { std::slice::from_raw_parts(s.as_ptr(), self.len()) })
            }
        }
    }

    /// Return nullptr if strides is None.
    pub fn strides_ptr(&self) -> *mut i64 {
        match self {
            Self::Contiguous(_) => std::ptr::null_mut(),
            Self::WithStrides(ref v) => &v[self.len()] as *const i64 as *mut i64,
            Self::Borrowed { strides, .. } => match strides {
                Some(strides) => strides.as_ptr(),
                None => std::ptr::null_mut(),
            },
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::Contiguous(_) => true,
            Self::Borrowed { strides: None, .. } => true,
            Self::WithStrides { .. }
            | Self::Borrowed {
                strides: Some(_), ..
            } => {
                let shape = self.shape();
                let strides = self.strides().unwrap();
                let mut expected_stride = 1;
                for (&dim, &stride) in shape.iter().rev().zip(strides.iter().rev()) {
                    if stride != expected_stride {
                        return false;
                    }
                    expected_stride *= dim;
                }
                true
            }
        }
    }
}
