use crate::Result;
use crate::error::{DataTypeSizeMismatchSnafu, NonContiguousSnafu};
use crate::ffi::DataType;
use crate::ffi::Tensor;
use crate::utils::MemoryOrder;
use snafu::ensure;
use std::ffi::c_void;

pub trait TensorView {
    fn dl_tensor(&self) -> &Tensor;

    fn data_ptr(&self) -> *mut c_void {
        self.dl_tensor().data
    }

    fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.dl_tensor().shape, self.num_dimensions()) }
    }

    fn strides(&self) -> Option<&[i64]> {
        if self.dl_tensor().strides.is_null() {
            None
        } else {
            Some(unsafe {
                std::slice::from_raw_parts(self.dl_tensor().strides, self.num_dimensions())
            })
        }
    }

    fn num_dimensions(&self) -> usize {
        self.dl_tensor().ndim as usize
    }

    fn num_elements(&self) -> usize {
        self.shape().iter().product::<i64>() as usize
    }

    fn data_type(&self) -> &DataType {
        &self.dl_tensor().dtype
    }

    fn byte_offset(&self) -> usize {
        self.dl_tensor().byte_offset as usize
    }

    fn as_slice_untyped(&self) -> &[u8] {
        let length = self.num_elements() * self.data_type().size();
        unsafe {
            std::slice::from_raw_parts(self.data_ptr().add(self.byte_offset()).cast(), length)
        }
    }

    unsafe fn as_slice<A>(&self) -> Result<&[A]> {
        let size = std::mem::size_of::<A>();
        let expected = self.data_type().size();
        ensure!(
            size == expected,
            DataTypeSizeMismatchSnafu { size, expected }
        );

        let s = unsafe {
            std::slice::from_raw_parts(
                self.data_ptr().add(self.byte_offset()).cast(),
                self.num_elements(),
            )
        };
        Ok(s)
    }

    fn memory_order(&self) -> MemoryOrder {
        match (self.shape(), self.strides()) {
            (_, None) => MemoryOrder::RowMajorContiguous,
            (shape, Some(strides)) => MemoryOrder::new(shape, strides),
        }
    }

    fn as_slice_contiguous<A>(&self) -> Result<&[A]> {
        ensure!(
            self.memory_order().is_contiguous(),
            NonContiguousSnafu {
                shape: self.shape(),
                strides: self.strides().expect("must have strides")
            }
        );
        unsafe { self.as_slice::<A>() }
    }
}

impl TensorView for Tensor {
    fn dl_tensor(&self) -> &Tensor {
        self
    }
}
