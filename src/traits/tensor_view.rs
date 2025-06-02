use crate::Result;
use crate::error::{DataTypeSizeMismatchSnafu, NonContiguousSnafu};
use crate::ffi::DataType;
use crate::ffi::Tensor;
use crate::utils::MemoryOrder;
use snafu::ensure;
use std::ffi::c_void;

/// A trait that provides a view into a DLPack tensor's data and metadata.
///
/// This trait defines methods to access and manipulate tensor data in a safe way,
/// providing access to the tensor's shape, strides, data type, and raw data.
/// It serves as a common interface for different tensor implementations.
pub trait TensorView {
    /// Returns a reference to the underlying DLPack tensor.
    ///
    /// This is the primary method that implementations must provide to access
    /// the raw tensor data structure.
    fn dl_tensor(&self) -> &Tensor;

    /// Returns a raw pointer to the tensor's data.
    ///
    /// This provides direct access to the tensor's memory location, which can be
    /// useful for low-level operations or FFI calls.
    fn data_ptr(&self) -> *mut c_void {
        self.dl_tensor().data
    }

    /// Returns a slice containing the tensor's shape dimensions.
    ///
    /// The shape array contains the size of each dimension of the tensor.
    /// For example, a 2x3 tensor would have shape [2, 3].
    fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.dl_tensor().shape, self.num_dimensions()) }
    }

    /// Returns an optional slice containing the tensor's strides.
    ///
    /// Strides define the number of elements to skip in memory when moving
    /// along each dimension. Returns None if the tensor is contiguous.
    ///
    /// # Returns
    /// * `Some(&[i64])` - The strides array if the tensor has custom strides
    /// * `None` - If the tensor is contiguous (strides are null)
    fn strides(&self) -> Option<&[i64]> {
        if self.dl_tensor().strides.is_null() {
            None
        } else {
            Some(unsafe {
                std::slice::from_raw_parts(self.dl_tensor().strides, self.num_dimensions())
            })
        }
    }

    /// Returns the number of dimensions in the tensor.
    ///
    /// This is equivalent to the length of the shape array.
    fn num_dimensions(&self) -> usize {
        self.dl_tensor().ndim as usize
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is calculated as the product of all dimensions in the shape.
    fn num_elements(&self) -> usize {
        self.shape().iter().product::<i64>() as usize
    }

    /// Returns a reference to the tensor's data type information.
    ///
    /// The data type contains information about the element type and size.
    fn data_type(&self) -> &DataType {
        &self.dl_tensor().dtype
    }

    /// Returns the byte offset from the start of the data pointer.
    ///
    /// This offset is used when the tensor is a view into a larger array,
    /// indicating where the tensor's data begins within that array.
    fn byte_offset(&self) -> usize {
        self.dl_tensor().byte_offset as usize
    }

    /// Returns a byte slice view of the tensor's data.
    ///
    /// This provides raw access to the tensor's memory as a byte slice,
    /// regardless of the actual data type. The length is calculated based
    /// on the number of elements and the size of each element.
    fn as_slice_untyped(&self) -> &[u8] {
        let length = self.num_elements() * self.data_type().size();
        unsafe {
            std::slice::from_raw_parts(self.data_ptr().add(self.byte_offset()).cast(), length)
        }
    }

    /// Returns a typed slice view of the tensor's data.
    ///
    /// # Arguments
    /// * `A` - The type to view the data as
    ///
    /// # Safety
    /// The caller must ensure that type `A` matches the tensor's data type.
    ///
    /// # Returns
    /// * `Result<&[A]>` - A slice of type `A` if the types match, or an error
    ///   if there's a size mismatch between the requested type and the tensor's type
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

    /// Determines the memory layout order of the tensor.
    ///
    /// Returns the memory order based on the tensor's shape and strides.
    /// If strides are not provided, assumes row-major contiguous layout.
    ///
    /// # Returns
    /// * `MemoryOrder` - The memory layout of the tensor
    fn memory_order(&self) -> MemoryOrder {
        match (self.shape(), self.strides()) {
            (_, None) => MemoryOrder::RowMajorContiguous,
            (shape, Some(strides)) => MemoryOrder::new(shape, strides),
        }
    }

    /// Returns a typed slice view of the tensor's data, ensuring it's contiguous.
    ///
    /// # Arguments
    /// * `A` - The type to view the data as
    ///
    /// # Returns
    /// * `Result<&[A]>` - A slice of type `A` if the tensor is contiguous, or an error
    ///   if the tensor is not contiguous or if there's a type mismatch
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

/// Implementation of TensorView for the raw Tensor type.
///
/// This allows direct access to Tensor data through the TensorView trait
/// without requiring a wrapper type.
impl TensorView for Tensor {
    /// Returns a reference to self since we're already a Tensor.
    fn dl_tensor(&self) -> &Tensor {
        self
    }
}
