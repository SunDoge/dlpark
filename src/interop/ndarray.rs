//! Zero-copy CPU interop for owned and borrowed `ndarray` arrays.
//!
//! Boxed owned arrays convert into [`crate::allocation::dynamic::Initialized`]
//! values. Shape and strides are copied into the managed allocation.
//!
//! ```
//! use dlpark::{Foreign, allocation::dynamic, ffi::DLManagedTensorVersioned};
//! use ndarray::{ArrayViewD, arr2};
//!
//! let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
//!     Box::new(arr2(&[[1_i32, 2], [3, 4]])).try_into()?;
//! let dlpack: Foreign<DLManagedTensorVersioned> =
//!     unsafe { initialized.finish() }.into_foreign();
//! let view = ArrayViewD::<i32>::try_from(&dlpack)?;
//! assert_eq!(view[[1, 0]], 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    DlpackElement, DlpackFlags, Foreign,
    allocation::dynamic,
    ffi::DLDevice,
    managed_tensor::ManagedTensorBase,
    metadata::{Copied, Dynamic},
};
use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Dimension, IxDyn, OwnedRepr, ShapeBuilder};
use snafu::{Snafu, ensure};
use std::os::raw::c_void;

#[derive(Debug, Snafu)]
/// Errors produced while validating a DLPack tensor as an ndarray view.
pub enum Error {
    /// An ndarray view cannot represent a negative DLPack stride.
    #[snafu(display("DLPack stride {axis} is negative: {value}"))]
    NegativeStride { axis: usize, value: i64 },

    /// A DLPack stride cannot be represented by ndarray's `usize` stride.
    #[snafu(display("DLPack stride {axis} with value {value} does not fit in usize"))]
    DlpackStrideOverflow { axis: usize, value: i64 },

    /// The address span described by the shape and strides overflowed.
    #[snafu(display("strided ndarray view span overflows usize"))]
    SpanOverflow,

    /// ndarray rejected the converted shape and strides.
    #[snafu(display("failed to build ndarray shape"))]
    Shape { source: ndarray::ShapeError },

    /// The underlying DLPack tensor failed validation.
    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },
}

/// Converts a boxed owned ndarray into an initialized DLPack allocation.
///
/// The array is not copied. Its shape and strides are converted to DLPack's
/// `i64` representation before the boxed array becomes the manager context.
/// The resulting allocation starts with [`DlpackFlags::IS_COPIED`] because ownership has been
/// transferred and no ndarray aliases remain.
impl<T, D, M> TryFrom<Box<ArrayBase<OwnedRepr<T>, D>>> for dynamic::Initialized<M>
where
    T: DlpackElement + Send,
    D: Dimension,
    M: ManagedTensorBase,
{
    type Error = crate::metadata::Error;

    fn try_from(array: Box<ArrayBase<OwnedRepr<T>, D>>) -> Result<Self, Self::Error> {
        let data_ptr = if array.is_empty() {
            std::ptr::null_mut()
        } else {
            array.as_ptr() as *mut c_void
        };
        let prepared =
            Dynamic::new(Copied(array.shape()), Copied(array.strides())).prepare::<M>()?;
        let mut initialized = prepared.initialize(array)?;
        initialized.set_data(data_ptr);
        initialized.set_dtype(T::DTYPE);
        initialized.set_device(DLDevice::CPU);

        // SAFETY: ownership of the ndarray is transferred into the builder.
        initialized.set_flags_unchecked(DlpackFlags::IS_COPIED);
        Ok(initialized)
    }
}

impl<'a, T, M> TryFrom<&'a Foreign<M>> for ArrayViewD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a Foreign<M>) -> Result<Self, Self::Error> {
        let tensor = unsafe { dlpack.tensor() };
        let (shape, strides) = shape_and_strides(tensor)?;
        let ptr = unsafe { tensor.offset_data_ptr::<T>()? };
        validate_strided_span(&shape, &strides)?;
        Ok(unsafe { ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), ptr) })
    }
}

impl<'a, T, M> TryFrom<&'a mut Foreign<M>> for ArrayViewMutD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a mut Foreign<M>) -> Result<Self, Self::Error> {
        if !dlpack.flags().contains(DlpackFlags::IS_COPIED) {
            return Err(crate::tensor::Error::NotCopied.into());
        }

        unsafe { array_view_from_dlpack_mut_unchecked(dlpack) }
    }
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
/// `Foreign` alone does not prove that the producer has no aliases.
/// Prefer [`ArrayViewMutD::try_from`], which additionally requires
/// [`DlpackFlags::IS_COPIED`] and needs no `unsafe` block.
pub unsafe fn array_view_from_dlpack_mut_unchecked<'a, T, M>(
    dlpack: &'a mut Foreign<M>,
) -> Result<ArrayViewMutD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    if dlpack.flags().contains(DlpackFlags::READ_ONLY) {
        return Err(crate::tensor::Error::ReadOnly.into());
    }

    let tensor = unsafe { dlpack.tensor() };
    let (shape, strides) = shape_and_strides(tensor)?;
    validate_non_overlapping(&shape, &strides)?;
    let ptr = unsafe { tensor.offset_data_ptr::<T>()? }.cast_mut();
    validate_strided_span(&shape, &strides)?;
    Ok(unsafe { ArrayViewMutD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), ptr) })
}

fn shape_and_strides(tensor: &crate::ffi::DLTensor) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let shape = unsafe { tensor.shape()? }
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim < 0 {
                return Err(crate::tensor::Error::NegativeDimension { axis, value: dim }.into());
            }
            usize::try_from(dim).map_err(|_| Error::SpanOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let strides = unsafe { tensor.strides_or_compact()? }
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

fn validate_strided_span(shape: &[usize], strides: &[usize]) -> Result<(), Error> {
    if shape.is_empty() {
        return Ok(());
    }
    if shape.contains(&0) {
        return Ok(());
    }

    shape
        .iter()
        .zip(strides)
        .try_fold(1usize, |span, (&dim, &stride)| {
            let axis_span = (dim - 1).checked_mul(stride).ok_or(Error::SpanOverflow)?;
            span.checked_add(axis_span).ok_or(Error::SpanOverflow)
        })
        .map(|_| ())
}

fn validate_non_overlapping(shape: &[usize], strides: &[usize]) -> Result<(), Error> {
    let mut axes = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(dim, _)| dim > 1)
        .collect::<Vec<_>>();
    axes.sort_unstable_by_key(|&(_, stride)| stride);

    let mut required_stride = 1usize;
    for (dim, stride) in axes {
        if stride < required_stride {
            return Err(Error::Shape {
                source: ndarray::ShapeError::from_kind(ndarray::ErrorKind::Unsupported),
            });
        }
        required_stride = stride.checked_mul(dim).ok_or(Error::SpanOverflow)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Local, ManagedTensorBase, legacy, test_support, versioned};
    use ndarray::{Array, arr2};

    fn managed_array<T, D, M>(array: ArrayBase<OwnedRepr<T>, D>) -> Local<M>
    where
        T: DlpackElement + Send,
        D: Dimension,
        M: ManagedTensorBase,
    {
        let initialized: dynamic::Initialized<M> = Box::new(array).try_into().unwrap();
        unsafe { initialized.finish() }
    }

    fn managed_array_with_flags<T, D, M>(
        array: ArrayBase<OwnedRepr<T>, D>,
        flags: DlpackFlags,
    ) -> Local<M>
    where
        T: DlpackElement + Send,
        D: Dimension,
        M: ManagedTensorBase,
    {
        let mut initialized: dynamic::Initialized<M> = Box::new(array).try_into().unwrap();
        initialized.set_flags_unchecked(flags);
        unsafe { initialized.finish() }
    }

    /// A `[[1, 2, 3], [4, 5, 6]]` legacy tensor. Legacy tensors have no flags
    /// field, so this is always writable via the `_unchecked` accessors and
    /// never satisfies `IS_COPIED`.
    fn legacy_2x3_dlpack() -> legacy::Dlpack {
        managed_array(arr2(&[[1i32, 2, 3], [4, 5, 6]]))
    }

    /// The transpose of [`legacy_2x3_dlpack`]'s array, i.e. non-compact strides.
    fn legacy_3x2_transposed_dlpack() -> legacy::Dlpack {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        managed_array(array.reversed_axes().to_owned())
    }

    /// A `[[1, 2, 3], [4, 5, 6]]` versioned tensor carrying the given flags.
    fn versioned_2x3_dlpack(flags: DlpackFlags) -> versioned::Dlpack {
        managed_array_with_flags(arr2(&[[1i32, 2, 3], [4, 5, 6]]), flags)
    }

    #[test]
    fn owned_ndarray_to_legacy_dlpack_keeps_layout_and_data() {
        let dlpack = legacy_2x3_dlpack();

        assert_eq!(dlpack.shape().unwrap(), &[2, 3]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[3, 1]);
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_keeps_layout_and_data() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack = managed_array(array);

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<f32>() }.unwrap(),
            &[1., 2., 3., 4.]
        );
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_sets_is_copied() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack = managed_array(array);

        assert_eq!(dlpack.flags(), DlpackFlags::IS_COPIED);
    }

    #[test]
    fn ndarray_builder_allows_setting_read_only_safely() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: versioned::Dlpack =
            managed_array_with_flags(array, DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY);

        assert_eq!(
            dlpack.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_allows_unsafe_mutation() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let mut dlpack: versioned::Dlpack = managed_array(array);

        unsafe {
            dlpack.cpu_slice_mut_unchecked::<f32>().unwrap()[1] = 42.;
        }

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<f32>() }.unwrap(),
            &[1., 42., 3., 4.]
        );
    }

    #[test]
    fn owned_arrayd_to_dlpack_keeps_dynamic_shape() {
        let array = arr2(&[[1i32, 2], [3, 4]]).into_dyn();
        let dlpack: legacy::Dlpack = managed_array(array);

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_is_zero_copy() {
        let dlpack = legacy_2x3_dlpack().into_foreign();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_preserves_strides() {
        let dlpack = legacy_3x2_transposed_dlpack().into_foreign();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.strides(), &[1, 3]);
        assert_eq!(view[[2, 1]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_mut_ndarray_view_unchecked_writes_through() {
        let mut dlpack = legacy_2x3_dlpack().into_foreign();
        let mut view =
            unsafe { array_view_from_dlpack_mut_unchecked::<i32, _>(&mut dlpack).unwrap() };

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        view[[1, 2]] = 42;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn borrowed_dlpack_to_mut_ndarray_view_unchecked_preserves_strides() {
        let mut dlpack = legacy_3x2_transposed_dlpack().into_foreign();
        let mut view =
            unsafe { array_view_from_dlpack_mut_unchecked::<i32, _>(&mut dlpack).unwrap() };

        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.strides(), &[1, 3]);
        view[[2, 1]] = 42;

        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();
        assert_eq!(view[[2, 1]], 42);
    }

    #[test]
    fn mut_ndarray_view_unchecked_rejects_read_only_tensor() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::READ_ONLY).into_foreign();

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
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED).into_foreign();

        let mut view = ArrayViewMutD::<i32>::try_from(&mut dlpack).unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn mut_ndarray_view_rejects_tensor_without_is_copied() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::empty()).into_foreign();

        let error = ArrayViewMutD::<i32>::try_from(&mut dlpack).unwrap_err();

        assert!(matches!(
            error,
            Error::Tensor {
                source: crate::tensor::Error::NotCopied
            }
        ));
    }

    #[test]
    fn mut_ndarray_view_rejects_overlapping_strides() {
        let data = Box::new(vec![1i32, 2]);
        let data_ptr = data.as_ptr().cast_mut().cast();
        let mut dlpack = test_support::fixed_tensor::<_, crate::ffi::DLManagedTensorVersioned, 2>(
            data,
            data_ptr,
            <i32 as DlpackElement>::DTYPE,
            DLDevice::CPU,
            [2, 2],
            [0, 1],
            DlpackFlags::IS_COPIED,
        )
        .into_foreign();

        assert!(matches!(
            ArrayViewMutD::<i32>::try_from(&mut dlpack),
            Err(Error::Shape { .. })
        ));
    }

    #[test]
    fn mut_ndarray_view_rejects_legacy_tensor_as_never_copied() {
        let mut dlpack = legacy_2x3_dlpack().into_foreign();

        let error = ArrayViewMutD::<i32>::try_from(&mut dlpack).unwrap_err();

        assert!(matches!(
            error,
            Error::Tensor {
                source: crate::tensor::Error::NotCopied
            }
        ));
    }

    #[test]
    fn try_into_mut_ndarray_view_requires_is_copied() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED).into_foreign();

        let mut view = ArrayViewMutD::<i32>::try_from(&mut dlpack).unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn sliced_owned_ndarray_to_dlpack_exports_non_standard_strides() {
        let array = Array::from_shape_vec((2, 2).strides((4, 2)), (0i32..7).collect()).unwrap();
        let dlpack: legacy::Dlpack = managed_array(array);
        let dlpack = dlpack.into_foreign();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[4, 2]);
        assert_eq!(view[[0, 1]], 2);
        assert_eq!(view[[1, 1]], 6);
    }
}
