use super::{Error, NegativeStrideSnafu};
use crate::{DlpackElement, Managed, ManagedTensorBase, TryFromDlpack};
use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn, ShapeBuilder};
use snafu::ensure;

impl<'a, T, M> TryFromDlpack<&'a Managed<M>, ()> for ArrayViewD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: &'a Managed<M>, _context: ()) -> Result<Self, Self::Error> {
        let tensor = dlpack.validate()?;
        let (shape, strides) = shape_and_strides(&tensor)?;
        let ptr = unsafe { tensor.offset_data_ptr::<T>()? };
        validate_strided_span(&shape, &strides)?;
        Ok(unsafe { ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), ptr) })
    }
}

impl<'a, T, M> TryFromDlpack<&'a mut Managed<M>, ()> for ArrayViewMutD<'a, T>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    type Error = Error;

    /// # Safety
    ///
    /// In addition to the trait-level requirements, no other reference may
    /// access the tensor data for the returned view's lifetime.
    unsafe fn try_from_dlpack(
        dlpack: &'a mut Managed<M>,
        _context: (),
    ) -> Result<Self, Self::Error> {
        unsafe { array_view_from_dlpack_mut_unchecked(dlpack) }
    }
}

/// Returns a mutable ndarray view into a DLPack tensor's CPU data, without proving exclusivity.
///
/// This rejects versioned tensors carrying `READ_ONLY`.
///
/// # Safety
///
/// The caller must ensure that no other references access the underlying
/// data for the lifetime of the returned view. Exclusive access to this
/// `Managed` alone does not prove that the producer has no aliases.
/// This is the implementation used by [`TryFromDlpack`]; callers must prove
/// exclusivity because DLPack flags alone cannot establish Rust aliasing.
pub unsafe fn array_view_from_dlpack_mut_unchecked<'a, T, M>(
    dlpack: &'a mut Managed<M>,
) -> Result<ArrayViewMutD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    let tensor = dlpack.validate_mut()?;
    let (shape, strides) = shape_and_strides(&tensor)?;
    validate_non_overlapping(&shape, &strides)?;
    let ptr = unsafe { tensor.offset_data_ptr::<T>()? }.cast_mut();
    validate_strided_span(&shape, &strides)?;
    Ok(unsafe { ArrayViewMutD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), ptr) })
}

fn shape_and_strides(
    tensor: &crate::tensor::TensorRef<'_>,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let shape = tensor
        .shape()
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
            usize::try_from(stride).map_err(|_| Error::StrideOverflow {
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
    // Fast path: the conservative stride-growth heuristic accepts every
    // layout that real frameworks produce (compact, sliced, broadcast).
    if conservative_non_overlapping(shape, strides)? {
        return Ok(());
    }
    // Rescue path: the heuristic is only sufficient, not necessary — e.g.
    // shape [2, 2] with strides [6, 5] touches {0, 5, 6, 11}, genuinely
    // distinct addresses, yet fails it. Fall back to an exact check.
    match exact_non_overlapping(shape, strides)? {
        true => Ok(()),
        false => Err(Error::Shape {
            source: ndarray::ShapeError::from_kind(ndarray::ErrorKind::Unsupported),
        }),
    }
}

/// Sufficient non-overlap check: ascending strides must each clear the index
/// footprint of all smaller-stride axes combined.
fn conservative_non_overlapping(shape: &[usize], strides: &[usize]) -> Result<bool, Error> {
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
            return Ok(false);
        }
        required_stride = stride.checked_mul(dim).ok_or(Error::SpanOverflow)?;
    }
    Ok(true)
}

/// Exact non-overlap check by construction of the element-offset set.
///
/// The covered offsets start as `{0}` and grow one axis at a time: for an
/// axis with stride `s` and length `d`, `d - 1` copies of the covered range
/// are added, shifted by `k * s`. Two distinct index tuples share an offset
/// exactly when some copy collides with existing coverage, so tracking the
/// covered set as disjoint intervals detects every overlap. The interval
/// count is capped; past the cap the layout is rejected conservatively
/// rather than spending unbounded memory on adversarial strides.
fn exact_non_overlapping(shape: &[usize], strides: &[usize]) -> Result<bool, Error> {
    const MAX_INTERVALS: usize = 1 << 16;

    let mut axes = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(dim, _)| dim > 1)
        .collect::<Vec<_>>();
    axes.sort_unstable_by_key(|&(_, stride)| stride);

    // Inclusive-start, exclusive-end element offsets covered so far.
    let mut covered: Vec<(usize, usize)> = vec![(0, 1)];
    for (dim, stride) in axes {
        let base = covered.clone();
        for k in 1..dim {
            let shift = k.checked_mul(stride).ok_or(Error::SpanOverflow)?;
            for &(lo, hi) in &base {
                let (lo, hi) = (lo.checked_add(shift), hi.checked_add(shift));
                let (Some(lo), Some(hi)) = (lo, hi) else {
                    return Err(Error::SpanOverflow);
                };
                covered.push((lo, hi));
                if covered.len() > MAX_INTERVALS {
                    return Ok(false);
                }
            }
        }
        covered.sort_unstable();
        // Any adjacent overlap proves two index tuples share an element.
        if covered.windows(2).any(|w| w[0].1 > w[1].0) {
            return Ok(false);
        }
        // Merge runs back into disjoint intervals to bound growth.
        covered.dedup_by(|second, first| {
            if first.1 >= second.0 {
                first.1 = first.1.max(second.1);
                true
            } else {
                false
            }
        });
    }
    Ok(true)
}
