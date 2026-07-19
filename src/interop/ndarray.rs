use crate::{
    DlpackElement, dlpack::ManagedBox, ffi::DLDevice, managed_tensor::ManagedTensorBase, metadata,
};
use ndarray::{ArrayBase, ArrayViewD, Dimension, IxDyn, OwnedRepr, ShapeBuilder};
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

    #[snafu(transparent)]
    Builder { source: crate::builder::Error },
}

impl<T, D, M> TryFrom<ArrayBase<OwnedRepr<T>, D>> for ManagedBox<M>
where
    T: DlpackElement + Send,
    D: Dimension,
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(array: ArrayBase<OwnedRepr<T>, D>) -> Result<Self, Self::Error> {
        dlpack_from_ndarray(array)
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

pub fn dlpack_from_ndarray<T, D, M>(
    array: ArrayBase<OwnedRepr<T>, D>,
) -> Result<ManagedBox<M>, Error>
where
    T: DlpackElement + Send,
    D: Dimension,
    M: ManagedTensorBase,
{
    // Read metadata after the array has moved into its stable Box, then copy it
    // directly into the final managed-tensor allocation.
    let data_ptr = if array.is_empty() {
        std::ptr::null_mut()
    } else {
        array.as_ptr() as *mut c_void
    };
    let mut managed = metadata::allocate_generic_slice_from_context::<_, M, usize, isize, _>(
        Box::new(array),
        |array| (array.shape(), array.strides()),
    )?;
    let tensor = unsafe { managed.as_mut() }.tensor_mut();
    tensor.data = data_ptr;
    tensor.dtype = T::DTYPE;
    tensor.device = DLDevice::CPU;

    Ok(unsafe { ManagedBox::new_unchecked(managed.as_ptr()) })
}

pub fn array_view_from_dlpack<'a, T, M>(
    dlpack: &'a ManagedBox<M>,
) -> Result<ArrayViewD<'a, T>, Error>
where
    T: DlpackElement,
    M: ManagedTensorBase,
{
    let tensor = dlpack.tensor();
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
    let ptr = tensor.cpu_data_ptr::<T>()?;
    let span_len = strided_span_len(&shape, &strides)?;
    let data = unsafe { std::slice::from_raw_parts(ptr, span_len) };

    ArrayViewD::from_shape(IxDyn(&shape).strides(IxDyn(&strides)), data).context(ShapeSnafu)
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
    use crate::{legacy, versioned};
    use ndarray::{Array, arr2};

    #[test]
    fn owned_ndarray_to_legacy_dlpack_keeps_layout_and_data() {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        let dlpack = legacy::Dlpack::try_from(array).unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 3]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[3, 1]);
        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_keeps_layout_and_data() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack = versioned::Dlpack::try_from(array).unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(dlpack.cpu_data_slice::<f32>().unwrap(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn owned_arrayd_to_dlpack_keeps_dynamic_shape() {
        let array = arr2(&[[1i32, 2], [3, 4]]).into_dyn();
        let dlpack = legacy::Dlpack::try_from(array).unwrap();

        assert_eq!(dlpack.shape().unwrap(), &[2, 2]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[2, 1]);
        assert_eq!(dlpack.cpu_data_slice::<i32>().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_is_zero_copy() {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        let dlpack = legacy::Dlpack::try_from(array).unwrap();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_preserves_strides() {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        let transposed = array.reversed_axes().to_owned();
        let dlpack = legacy::Dlpack::try_from(transposed).unwrap();
        let view = array_view_from_dlpack::<i32, _>(&dlpack).unwrap();

        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.strides(), &[1, 3]);
        assert_eq!(view[[2, 1]], 6);
    }

    #[test]
    fn sliced_owned_ndarray_to_dlpack_exports_non_standard_strides() {
        let array = Array::from_shape_vec((2, 2).strides((4, 2)), (0i32..7).collect()).unwrap();
        let dlpack = legacy::Dlpack::try_from(array).unwrap();
        let view = ArrayViewD::<i32>::try_from(&dlpack).unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[4, 2]);
        assert_eq!(view[[0, 1]], 2);
        assert_eq!(view[[1, 1]], 6);
    }
}
