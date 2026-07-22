use super::{Error, NegativeStrideSnafu};
use crate::{DlpackElement, DlpackFlags, Foreign, ManagedTensorBase, TryFromDlpack};
use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn, ShapeBuilder};
use snafu::ensure;

impl<'a, T, M> TryFromDlpack<&'a Foreign<M>> for ArrayViewD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: &'a Foreign<M>) -> Result<Self, Self::Error> {
        let tensor = unsafe { dlpack.tensor() };
        let (shape, strides) = shape_and_strides(tensor)?;
        let ptr = unsafe { tensor.offset_data_ptr::<T>()? };
        validate_strided_span(&shape, &strides)?;
        Ok(unsafe { ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), ptr) })
    }
}

impl<'a, T, M> TryFromDlpack<&'a mut Foreign<M>> for ArrayViewMutD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    /// # Safety
    ///
    /// In addition to the trait-level requirements, no other reference may
    /// access the tensor data for the returned view's lifetime.
    unsafe fn try_from_dlpack(dlpack: &'a mut Foreign<M>) -> Result<Self, Self::Error> {
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
/// This is the implementation used by [`TryFromDlpack`]; callers must prove
/// exclusivity because foreign flags alone cannot establish Rust aliasing.
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
