use crate::ManagedTensor;
use crate::ffi::DLTensor;
use std::ptr::NonNull;

pub struct Dlpack<M: ManagedTensor>(NonNull<M>);

impl<M> Dlpack<M>
where
    M: ManagedTensor,
{
    pub fn new(ptr: *mut M) -> Option<Self> {
        NonNull::new(ptr).map(Dlpack)
    }

    /// Create a new `Dlpack` instance from a raw pointer without checking if it is null.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `ptr` is not null and points to a valid `M`.
    pub unsafe fn new_unchecked(ptr: *mut M) -> Self {
        Self(unsafe { NonNull::new_unchecked(ptr) })
    }

    pub fn dl_tensor(&self) -> &DLTensor {
        unsafe { self.0.as_ref().get_dltensor() }
    }

    /// Returns the shape of the tensor as a slice.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::shape`] for error conditions.
    pub fn shape(&self) -> Result<&[i64], crate::tensor::Error> {
        self.dl_tensor().shape()
    }

    /// Returns the strides of the tensor as a slice, or `None` for compact row-major layout.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::strides`] for error conditions.
    pub fn strides(&self) -> Result<Option<&[i64]>, crate::tensor::Error> {
        self.dl_tensor().strides()
    }

    /// Returns the total number of elements in the tensor (product of all shape dimensions).
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::num_elements`] for error conditions.
    pub fn num_elements(&self) -> Result<usize, crate::tensor::Error> {
        self.dl_tensor().num_elements()
    }

    /// Returns the total size of the tensor data in bytes.
    ///
    /// # Errors
    ///
    /// See [`crate::ffi::DLTensor::num_bytes`] for error conditions.
    pub fn num_bytes(&self) -> Result<usize, crate::tensor::Error> {
        self.dl_tensor().num_bytes()
    }

    /// Consumes the `Dlpack`, returning the wrapped raw pointer.
    ///
    /// The caller takes ownership of the memory and is responsible for calling the FFI deleter later.
    pub fn into_raw(self) -> *mut M {
        let ptr = self.0.as_ptr();
        std::mem::forget(self);
        ptr
    }

    /// Returns the wrapped raw pointer without consuming the `Dlpack`.
    ///
    /// The `Dlpack` still owns the memory and will automatically drop it when it goes out of scope.
    pub fn as_ptr(&self) -> *mut M {
        self.0.as_ptr()
    }
}

impl<M> Drop for Dlpack<M>
where
    M: ManagedTensor,
{
    fn drop(&mut self) {
        unsafe {
            M::call_deleter(self.0.as_ptr());
        }
    }
}
