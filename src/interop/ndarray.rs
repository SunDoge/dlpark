//! Zero-copy CPU interop for owned and borrowed `ndarray` arrays.
//!
//! Owned arrays convert into [`crate::Builder`] values. Shape and stride
//! values are copied into the final managed tensor allocation only when the
//! builder is built, while the array allocation remains owned by its context.

use crate::{
    Builder, DlpackElement, DlpackFlags,
    dlpack::ManagedBox,
    ffi::DLDevice,
    managed_tensor::ManagedTensorBase,
    metadata::{self, Metadata},
};
use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Dimension, IxDyn, OwnedRepr, ShapeBuilder};
use snafu::{ResultExt, Snafu, ensure};
use std::os::raw::c_void;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("DLPack stride {axis} is negative: {value}"))]
    NegativeStride { axis: usize, value: i64 },

    #[snafu(display("DLPack stride {axis} with value {value} does not fit in usize"))]
    DlpackStrideOverflow { axis: usize, value: i64 },

    #[snafu(display("strided ndarray view span overflows usize"))]
    SpanOverflow,

    #[snafu(display("failed to build ndarray shape"))]
    Shape { source: ndarray::ShapeError },

    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },
}

/// Copies an ndarray's runtime shape and strides into the managed tensor
/// allocation when its [`Builder`] is built.
///
/// The pointers refer into the boxed ndarray owned by the same builder. Moving
/// the builder does not move that allocation, and the fields are private so
/// this relationship can only be constructed by the conversion below.
#[derive(Debug)]
pub struct NdarrayMetadata {
    shape: *const usize,
    strides: *const isize,
    ndim: usize,
}

impl Metadata for NdarrayMetadata {
    type Error = metadata::Error;

    fn try_allocate<C, M>(self, ctx: C) -> Result<std::ptr::NonNull<M>, Self::Error>
    where
        C: crate::OpaqueContext,
        M: ManagedTensorBase,
    {
        // SAFETY: construction boxes the ndarray before capturing these
        // pointers, and the builder still owns that box while allocating.
        let shape = unsafe { std::slice::from_raw_parts(self.shape, self.ndim) };
        let strides = unsafe { std::slice::from_raw_parts(self.strides, self.ndim) };
        metadata::GenericSlice::new(shape, strides).try_allocate(ctx)
    }

    unsafe fn allocate_unchecked<C, M>(self, ctx: C) -> std::ptr::NonNull<M>
    where
        C: crate::OpaqueContext,
        M: ManagedTensorBase,
    {
        // SAFETY: the pointer lifetime follows from the same invariant as
        // try_allocate; the caller supplies GenericSlice's unchecked
        // conversion and length invariants.
        let shape = unsafe { std::slice::from_raw_parts(self.shape, self.ndim) };
        let strides = unsafe { std::slice::from_raw_parts(self.strides, self.ndim) };
        unsafe { metadata::GenericSlice::new(shape, strides).allocate_unchecked(ctx) }
    }
}

impl<T, D> From<ArrayBase<OwnedRepr<T>, D>>
    for Builder<Box<ArrayBase<OwnedRepr<T>, D>>, NdarrayMetadata>
where
    T: DlpackElement + Send,
    D: Dimension,
{
    fn from(array: ArrayBase<OwnedRepr<T>, D>) -> Self {
        let array = Box::new(array);
        let data_ptr = if array.is_empty() {
            std::ptr::null_mut()
        } else {
            array.as_ptr() as *mut c_void
        };
        let metadata = NdarrayMetadata {
            shape: array.shape().as_ptr(),
            strides: array.strides().as_ptr(),
            ndim: array.ndim(),
        };
        let builder = Builder::new(array, metadata)
            .data(data_ptr)
            .dtype(T::DTYPE)
            .device(DLDevice::CPU);

        // SAFETY: the owned ndarray was moved into the builder and has no
        // remaining aliases to its storage.
        unsafe { builder.flags_unchecked(DlpackFlags::IS_COPIED) }
    }
}

impl<'a, T, M> TryFrom<&'a ManagedBox<M>> for ArrayViewD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a ManagedBox<M>) -> Result<Self, Self::Error> {
        array_view_from_dlpack(dlpack)
    }
}

impl<'a, T, M> TryFrom<&'a mut ManagedBox<M>> for ArrayViewMutD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a mut ManagedBox<M>) -> Result<Self, Self::Error> {
        array_view_from_dlpack_mut(dlpack)
    }
}

pub fn array_view_from_dlpack<'a, T, M>(
    dlpack: &'a ManagedBox<M>,
) -> Result<ArrayViewD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    let tensor = dlpack.tensor();
    let (shape, strides) = shape_and_strides(tensor)?;
    let ptr = tensor.cpu_data_ptr::<T>()?;
    let span_len = strided_span_len(&shape, &strides)?;
    let data = unsafe { std::slice::from_raw_parts(ptr, span_len) };

    ArrayViewD::from_shape(IxDyn(&shape).strides(IxDyn(&strides)), data).context(ShapeSnafu)
}

/// Returns a mutable ndarray view into a DLPack tensor's CPU data, without proving exclusivity.
///
/// This rejects versioned tensors carrying [`DlpackFlags::READ_ONLY`].
/// Legacy tensors have no flags and are treated as writable.
///
/// # Safety
///
/// The caller must ensure that no other references access the underlying
/// data for the lifetime of the returned view. Exclusive access to this
/// `ManagedBox` alone does not prove that the producer has no aliases.
/// Prefer [`array_view_from_dlpack_mut`], which additionally requires
/// [`DlpackFlags::IS_COPIED`] and needs no `unsafe` block.
pub unsafe fn array_view_from_dlpack_mut_unchecked<'a, T, M>(
    dlpack: &'a mut ManagedBox<M>,
) -> Result<ArrayViewMutD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    if dlpack.flags().contains(DlpackFlags::READ_ONLY) {
        return Err(crate::tensor::Error::ReadOnly.into());
    }

    let tensor = dlpack.tensor();
    let (shape, strides) = shape_and_strides(tensor)?;
    let ptr = tensor.cpu_data_ptr::<T>()?.cast_mut();
    let span_len = strided_span_len(&shape, &strides)?;
    let data = unsafe { std::slice::from_raw_parts_mut(ptr, span_len) };

    ArrayViewMutD::from_shape(IxDyn(&shape).strides(IxDyn(&strides)), data).context(ShapeSnafu)
}

/// Returns a mutable ndarray view into a DLPack tensor's CPU data.
///
/// This rejects tensors carrying [`DlpackFlags::READ_ONLY`], and requires
/// [`DlpackFlags::IS_COPIED`] to be set: per the DLPack spec, a tensor
/// copied specifically for this export has no other live aliases, which is
/// what makes this safe to call without an `unsafe` block. Legacy tensors
/// have no flags field and so can never satisfy `IS_COPIED`; use
/// [`array_view_from_dlpack_mut_unchecked`] for those.
pub fn array_view_from_dlpack_mut<'a, T, M>(
    dlpack: &'a mut ManagedBox<M>,
) -> Result<ArrayViewMutD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    if !dlpack.flags().contains(DlpackFlags::IS_COPIED) {
        return Err(crate::tensor::Error::NotCopied.into());
    }

    unsafe { array_view_from_dlpack_mut_unchecked(dlpack) }
}

fn shape_and_strides(tensor: &crate::ffi::DLTensor) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let shape = tensor
        .shape()?
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim < 0 {
                return Err(crate::tensor::Error::NegativeDimension { axis, value: dim }.into());
            }
            usize::try_from(dim).map_err(|_| Error::SpanOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let strides = tensor
        .strides_or_compact()?
        .iter()
        .enumerate()
        .map(|(axis, &stride)| {
            ensure!(
                stride >= 0,
                NegativeStrideSnafu {
                    axis,
                    value: stride
                }
            );
            usize::try_from(stride).map_err(|_| Error::DlpackStrideOverflow {
                axis,
                value: stride,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((shape, strides))
}

fn strided_span_len(shape: &[usize], strides: &[usize]) -> Result<usize, Error> {
    if shape.is_empty() {
        return Ok(1);
    }
    if shape.contains(&0) {
        return Ok(0);
    }

    shape
        .iter()
        .zip(strides)
        .try_fold(1usize, |span, (&dim, &stride)| {
            let axis_span = (dim - 1).checked_mul(stride).ok_or(Error::SpanOverflow)?;
            span.checked_add(axis_span).ok_or(Error::SpanOverflow)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ffi::{DLManagedTensor, DLManagedTensorVersioned},
        legacy, versioned,
    };
    use ndarray::{Array, arr2};

    /// A `[[1, 2, 3], [4, 5, 6]]` legacy tensor. Legacy tensors have no flags
    /// field, so this is always writable via the `_unchecked` accessors and
    /// never satisfies `IS_COPIED`.
    fn legacy_2x3_dlpack() -> legacy::Dlpack {
        Builder::from(arr2(&[[1i32, 2, 3], [4, 5, 6]]))
            .try_build::<DLManagedTensor>()
            .unwrap()
    }

    /// The transpose of [`legacy_2x3_dlpack`]'s array, i.e. non-compact strides.
    fn legacy_3x2_transposed_dlpack() -> legacy::Dlpack {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        Builder::from(array.reversed_axes().to_owned())
            .try_build::<DLManagedTensor>()
            .unwrap()
    }

    /// A `[[1, 2, 3], [4, 5, 6]]` versioned tensor carrying the given flags.
    fn versioned_2x3_dlpack(flags: DlpackFlags) -> versioned::Dlpack {
        Builder::from(arr2(&[[1i32, 2, 3], [4, 5, 6]]))
            .flags(flags)
            .unwrap()
            .try_build::<DLManagedTensorVersioned>()
            .unwrap()
    }

    #[test]
    fn owned_ndarray_to_legacy_dlpack_keeps_layout_and_data() {
        let dlpack = legacy_2x3_dlpack();

        assert_eq!(dlpack.shape().unwrap(), &[2, 3]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[3, 1]);
        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_keeps_layout_and_data() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack = Builder::from(array).try_build().unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(dlpack.cpu_data_slice::<f32>().unwrap(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_sets_is_copied() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack = Builder::from(array).try_build().unwrap();

        assert_eq!(dlpack.flags(), DlpackFlags::IS_COPIED);
    }

    #[test]
    fn ndarray_builder_allows_setting_read_only_safely() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack = Builder::from(array)
            .insert_flags(DlpackFlags::READ_ONLY)
            .unwrap()
            .try_build()
            .unwrap();

        assert_eq!(
            dlpack.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_allows_unsafe_mutation() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let mut dlpack: versioned::Dlpack = Builder::from(array).try_build().unwrap();

        unsafe {
            dlpack.cpu_data_slice_mut_unchecked::<f32>().unwrap()[1] = 42.;
        }

        assert_eq!(dlpack.cpu_data_slice::<f32>().unwrap(), &[1., 42., 3., 4.]);
    }

    #[test]
    fn owned_arrayd_to_dlpack_keeps_dynamic_shape() {
        let array = arr2(&[[1i32, 2], [3, 4]]).into_dyn();
        let dlpack: legacy::Dlpack = Builder::from(array).try_build().unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_is_zero_copy() {
        let dlpack = legacy_2x3_dlpack();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_preserves_strides() {
        let dlpack = legacy_3x2_transposed_dlpack();
        let view = array_view_from_dlpack::<i32, _>(&dlpack).unwrap();

        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.strides(), &[1, 3]);
        assert_eq!(view[[2, 1]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_mut_ndarray_view_unchecked_writes_through() {
        let mut dlpack = legacy_2x3_dlpack();
        let mut view =
            unsafe { array_view_from_dlpack_mut_unchecked::<i32, _>(&mut dlpack).unwrap() };

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        view[[1, 2]] = 42;

        assert_eq!(
            dlpack.cpu_data_slice::<i32>().unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn borrowed_dlpack_to_mut_ndarray_view_unchecked_preserves_strides() {
        let mut dlpack = legacy_3x2_transposed_dlpack();
        let mut view =
            unsafe { array_view_from_dlpack_mut_unchecked::<i32, _>(&mut dlpack).unwrap() };

        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.strides(), &[1, 3]);
        view[[2, 1]] = 42;

        let view = array_view_from_dlpack::<i32, _>(&dlpack).unwrap();
        assert_eq!(view[[2, 1]], 42);
    }

    #[test]
    fn mut_ndarray_view_unchecked_rejects_read_only_tensor() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::READ_ONLY);

        let error =
            unsafe { array_view_from_dlpack_mut_unchecked::<i32, _>(&mut dlpack) }.unwrap_err();

        assert!(matches!(
            error,
            Error::Tensor {
                source: crate::tensor::Error::ReadOnly
            }
        ));
    }

    #[test]
    fn mut_ndarray_view_updates_is_copied_tensor_without_unsafe() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED);

        let mut view = array_view_from_dlpack_mut::<i32, _>(&mut dlpack).unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            dlpack.cpu_data_slice::<i32>().unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn mut_ndarray_view_rejects_tensor_without_is_copied() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::empty());

        let error = array_view_from_dlpack_mut::<i32, _>(&mut dlpack).unwrap_err();

        assert!(matches!(
            error,
            Error::Tensor {
                source: crate::tensor::Error::NotCopied
            }
        ));
    }

    #[test]
    fn mut_ndarray_view_rejects_legacy_tensor_as_never_copied() {
        let mut dlpack = legacy_2x3_dlpack();

        let error = array_view_from_dlpack_mut::<i32, _>(&mut dlpack).unwrap_err();

        assert!(matches!(
            error,
            Error::Tensor {
                source: crate::tensor::Error::NotCopied
            }
        ));
    }

    #[test]
    fn try_into_mut_ndarray_view_requires_is_copied() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED);

        let mut view = ArrayViewMutD::<i32>::try_from(&mut dlpack).unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            dlpack.cpu_data_slice::<i32>().unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn sliced_owned_ndarray_to_dlpack_exports_non_standard_strides() {
        let array = Array::from_shape_vec((2, 2).strides((4, 2)), (0i32..7).collect()).unwrap();
        let dlpack: legacy::Dlpack = Builder::from(array).try_build().unwrap();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[4, 2]);
        assert_eq!(view[[0, 1]], 2);
        assert_eq!(view[[1, 1]], 6);
    }
}
