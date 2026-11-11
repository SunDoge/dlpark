use super::{Error, compact_strides, is_compact_strides};
use crate::{
    DlpackElement, DlpackFlags,
    ffi::{DLDataType, DLDevice, DLTensor},
};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::rc::Rc;

/// A structurally validated view of a raw DLPack tensor descriptor.
///
/// Shape and stride metadata is safe to inspect for this value's lifetime.
/// DLPack does not report the bounds of the underlying data allocation, so
/// dereferencing the data pointer remains unsafe. This type carries no owner
/// provenance and is therefore deliberately not `Send` or `Sync`.
///
/// ```compile_fail
/// use dlpark::RawTensorRef;
/// fn assert_send<T: Send>() {}
/// assert_send::<RawTensorRef<'static>>();
/// ```
pub struct RawTensorRef<'a> {
    tensor: &'a DLTensor,
    shape: &'a [i64],
    strides: Option<&'a [i64]>,
    num_elements: usize,
    num_bytes: usize,
    _not_thread_safe: PhantomData<Rc<()>>,
}

/// An exclusively borrowed, structurally validated raw DLPack descriptor.
///
/// Exclusivity applies to the descriptor, not necessarily to its data
/// allocation. Mutable data access therefore remains unsafe. This type carries
/// no owner provenance and is deliberately not `Send` or `Sync`.
///
/// ```compile_fail
/// use dlpark::RawTensorMut;
/// fn assert_send<T: Send>() {}
/// assert_send::<RawTensorMut<'static>>();
/// ```
pub struct RawTensorMut<'a> {
    inner: RawTensorRef<'a>,
    _exclusive: PhantomData<&'a mut DLTensor>,
}

/// A validated tensor view borrowed from an owning [`crate::Managed`] handle.
///
/// This is the normal validated view returned by [`crate::Managed::validate`].
/// It is safe to share and move between threads because its managed owner
/// guarantees immutable metadata and cross-thread destruction.
pub struct TensorRef<'a> {
    inner: RawTensorRef<'a>,
}

/// An exclusively borrowed validated tensor view from [`crate::Managed`].
///
/// This is returned by [`crate::Managed::validate_mut`]. The view may be moved
/// to another thread, but it is not shareable because it represents exclusive
/// access to the managed tensor descriptor.
pub struct TensorMut<'a> {
    inner: RawTensorMut<'a>,
}

// SAFETY: these wrappers can only be constructed by Managed, whose contract
// guarantees immutable descriptor metadata and permits its allocation deleter
// to run on any thread. Data-pointer dereferences remain unsafe.
unsafe impl Send for TensorRef<'_> {}
unsafe impl Sync for TensorRef<'_> {}
unsafe impl Send for TensorMut<'_> {}

/// A shared interface for validated managed and raw tensor views.
///
/// Generic consumers can accept this trait when they only inspect descriptor
/// metadata or perform explicitly unsafe data-pointer access.
pub trait TensorView {
    /// Returns the underlying structurally validated descriptor view.
    fn as_raw_tensor(&self) -> &RawTensorRef<'_>;
}

impl<'a> RawTensorRef<'a> {
    /// Validates the descriptor's readable metadata and arithmetic invariants.
    /// Unknown dtype codes and device types remain valid for forward
    /// compatibility, but dtype bit width and lane count must be nonzero.
    ///
    /// # Safety
    ///
    /// `tensor` and its shape and optional strides pointers must remain readable
    /// and immutable for `'a`.
    pub unsafe fn from_raw(tensor: &'a DLTensor) -> Result<Self, Error> {
        if tensor.dtype.bits == 0 {
            return Err(Error::ZeroDtypeBits);
        }
        if tensor.dtype.lanes == 0 {
            return Err(Error::ZeroDtypeLanes);
        }
        let shape = unsafe { tensor.shape()? };
        let strides = unsafe { tensor.strides()? };
        let num_elements = unsafe { tensor.num_elements()? };
        let num_bytes = unsafe { tensor.num_bytes()? };
        Ok(Self {
            tensor,
            shape,
            strides,
            num_elements,
            num_bytes,
            _not_thread_safe: PhantomData,
        })
    }

    /// Returns the tensor's device.
    pub fn device(&self) -> DLDevice {
        self.tensor.device
    }

    /// Returns the tensor's element dtype.
    pub fn dtype(&self) -> DLDataType {
        self.tensor.dtype
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the shape as a slice of dimension sizes.
    pub fn shape(&self) -> &'a [i64] {
        self.shape
    }

    /// Returns the explicit strides, or `None` when DLPack stores them
    /// implicitly as compact row-major.
    pub fn strides(&self) -> Option<&'a [i64]> {
        self.strides
    }

    /// Returns the byte offset from the base data pointer.
    pub fn byte_offset(&self) -> u64 {
        self.tensor.byte_offset
    }

    /// Returns the total element count (product of the shape).
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Returns the total byte size, accounting for packed sub-byte dtypes.
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns explicit strides, or computed compact row-major strides when
    /// DLPack stores them implicitly.
    pub fn strides_or_compact(&self) -> Result<Cow<'a, [i64]>, Error> {
        match self.strides {
            Some(strides) => Ok(Cow::Borrowed(strides)),
            None if self.shape.is_empty() => Ok(Cow::Borrowed(&[])),
            None => Ok(Cow::Owned(compact_strides(self.shape)?)),
        }
    }

    /// Returns whether the strides describe compact row-major layout.
    pub fn is_compact(&self) -> Result<bool, Error> {
        is_compact_strides(self.shape, self.strides)
    }

    /// Returns the base data pointer without the byte offset applied.
    pub fn data_ptr(&self) -> *const std::ffi::c_void {
        self.tensor.data_ptr()
    }

    /// Returns the byte-offset-adjusted typed data pointer.
    ///
    /// # Safety
    ///
    /// For a non-empty tensor, the adjusted pointer must lie within the data
    /// allocation and be valid for the tensor's device.
    pub unsafe fn offset_data_ptr<T: DlpackElement>(&self) -> Result<*const T, Error> {
        unsafe { self.tensor.offset_data_ptr::<T>() }
    }

    /// Returns the byte-offset-adjusted untyped data pointer.
    ///
    /// # Safety
    ///
    /// For a non-empty tensor, the adjusted pointer must lie within the data
    /// allocation and be valid for the tensor's device.
    pub unsafe fn offset_bytes_ptr(&self) -> Result<*const u8, Error> {
        unsafe { self.tensor.offset_bytes_ptr() }
    }

    /// Borrows compact CPU data as typed elements.
    ///
    /// # Safety
    ///
    /// The adjusted data pointer must reference `num_elements()` initialized
    /// values of `T` for the returned lifetime.
    pub unsafe fn cpu_slice<T: DlpackElement>(&self) -> Result<&'a [T], Error> {
        unsafe { self.tensor.cpu_slice::<T>() }
    }

    /// Borrows compact CPU storage as bytes.
    ///
    /// # Safety
    ///
    /// The adjusted data pointer must reference `num_bytes()` initialized bytes
    /// for the returned lifetime.
    pub unsafe fn cpu_bytes(&self) -> Result<&'a [u8], Error> {
        unsafe { self.tensor.cpu_bytes() }
    }
}

impl<'a> TensorRef<'a> {
    pub(crate) fn new(inner: RawTensorRef<'a>) -> Self {
        Self { inner }
    }
}

impl<'a> std::ops::Deref for TensorRef<'a> {
    type Target = RawTensorRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TensorView for TensorRef<'_> {
    fn as_raw_tensor(&self) -> &RawTensorRef<'_> {
        &self.inner
    }
}

impl TensorView for RawTensorRef<'_> {
    fn as_raw_tensor(&self) -> &RawTensorRef<'_> {
        self
    }
}

impl<'a> RawTensorMut<'a> {
    /// Validates a raw descriptor for exclusive descriptor access.
    ///
    /// `READ_ONLY` tensors are rejected. Exclusivity of the descriptor does
    /// not prove exclusivity of the underlying allocation.
    ///
    /// # Safety
    ///
    /// `tensor` and all metadata pointers it references must remain valid for
    /// `'a`, and no other reference may mutate the descriptor during `'a`.
    pub unsafe fn from_raw(tensor: &'a mut DLTensor, flags: DlpackFlags) -> Result<Self, Error> {
        if flags.contains(DlpackFlags::READ_ONLY) {
            return Err(Error::ReadOnly);
        }
        let inner = unsafe { RawTensorRef::from_raw(tensor) }?;
        Ok(Self {
            inner,
            _exclusive: PhantomData,
        })
    }

    /// Borrows compact CPU data as mutable typed elements.
    ///
    /// # Safety
    ///
    /// No other reference may access the data allocation for the returned
    /// lifetime, and the adjusted pointer must cover `num_elements()` values.
    pub unsafe fn cpu_slice_mut<T: DlpackElement>(&mut self) -> Result<&mut [T], Error> {
        self.inner.tensor.ensure_cpu()?;
        if !self.inner.is_compact()? {
            return Err(Error::NonCompactStrides);
        }
        let data = unsafe { self.inner.offset_data_ptr::<T>()? }.cast_mut();
        Ok(unsafe { std::slice::from_raw_parts_mut(data, self.inner.num_elements) })
    }

    /// Borrows compact CPU storage as mutable bytes.
    ///
    /// # Safety
    ///
    /// No other reference may access the data allocation for the returned
    /// lifetime, and the adjusted pointer must cover `num_bytes()` bytes.
    pub unsafe fn cpu_bytes_mut(&mut self) -> Result<&mut [u8], Error> {
        self.inner.tensor.ensure_cpu()?;
        if !self.inner.is_compact()? {
            return Err(Error::NonCompactStrides);
        }
        let data = unsafe { self.inner.offset_bytes_ptr()? }.cast_mut();
        Ok(unsafe { std::slice::from_raw_parts_mut(data, self.inner.num_bytes) })
    }
}

impl<'a> TensorMut<'a> {
    pub(crate) fn new(inner: RawTensorMut<'a>) -> Self {
        Self { inner }
    }
}

impl<'a> std::ops::Deref for TensorMut<'a> {
    type Target = RawTensorMut<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for TensorMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl TensorView for TensorMut<'_> {
    fn as_raw_tensor(&self) -> &RawTensorRef<'_> {
        &self.inner.inner
    }
}

impl TensorView for RawTensorMut<'_> {
    fn as_raw_tensor(&self) -> &RawTensorRef<'_> {
        &self.inner
    }
}

impl<'a> std::ops::Deref for RawTensorMut<'a> {
    type Target = RawTensorRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn managed_views_have_cross_thread_traits() {
        assert_send_sync::<TensorRef<'static>>();
        assert_send::<TensorMut<'static>>();
    }

    #[test]
    fn tensor_view_unifies_raw_and_managed_views() {
        fn rank(view: &impl TensorView) -> usize {
            view.as_raw_tensor().ndim()
        }

        let tensor = DLTensor {
            dtype: DLDataType::U8,
            ..DLTensor::default()
        };
        let raw = unsafe { RawTensorRef::from_raw(&tensor) }.unwrap();
        assert_eq!(rank(&raw), 0);
    }

    #[test]
    fn rejects_negative_rank() {
        let tensor = DLTensor {
            ndim: -1,
            dtype: DLDataType::U8,
            ..DLTensor::default()
        };
        let error = match unsafe { RawTensorRef::from_raw(&tensor) } {
            Ok(_) => panic!("negative rank must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::NegativeNdim { ndim: -1 }));
    }

    #[test]
    fn rejects_null_shape_for_nonzero_rank() {
        let tensor = DLTensor {
            ndim: 1,
            dtype: DLDataType::U8,
            ..DLTensor::default()
        };
        let error = match unsafe { RawTensorRef::from_raw(&tensor) } {
            Ok(_) => panic!("null shape must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::NullShapePtr { ndim: 1 }));
    }

    #[test]
    fn exposes_validated_metadata() {
        let shape = [2_i64, 3];
        let strides = [3_i64, 1];
        let tensor = DLTensor {
            ndim: 2,
            dtype: DLDataType::U8,
            shape: shape.as_ptr().cast_mut(),
            strides: strides.as_ptr().cast_mut(),
            ..DLTensor::default()
        };

        let tensor = unsafe { RawTensorRef::from_raw(&tensor) }.unwrap();
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.strides(), Some(strides.as_slice()));
        assert_eq!(tensor.num_elements(), 6);
    }

    #[test]
    fn rejects_zero_width_dtype() {
        let tensor = DLTensor {
            dtype: DLDataType {
                code: crate::ffi::DLDataTypeCode::UINT,
                bits: 0,
                lanes: 1,
            },
            ..DLTensor::default()
        };

        assert!(matches!(
            unsafe { RawTensorRef::from_raw(&tensor) },
            Err(Error::ZeroDtypeBits)
        ));
    }

    #[test]
    fn rejects_zero_lane_dtype() {
        let tensor = DLTensor {
            dtype: DLDataType {
                code: crate::ffi::DLDataTypeCode::UINT,
                bits: 8,
                lanes: 0,
            },
            ..DLTensor::default()
        };

        assert!(matches!(
            unsafe { RawTensorRef::from_raw(&tensor) },
            Err(Error::ZeroDtypeLanes)
        ));
    }
}
