//! Zero-copy CPU interop for owned and borrowed `ndarray` arrays.
//!
//! Boxed owned arrays convert into [`crate::allocation::dynamic::Initialized`]
//! values. Shape and strides are copied into the managed allocation.
//!
//! ```
//! use dlpark::{Foreign, TryFromDlpack, allocation::dynamic, ffi::DLManagedTensorVersioned};
//! use ndarray::{ArrayViewD, arr2};
//!
//! let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
//!     Box::new(arr2(&[[1_i32, 2], [3, 4]])).try_into()?;
//! let dlpack: Foreign<DLManagedTensorVersioned> =
//!     unsafe { initialized.finish() }.into_foreign();
//! let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&dlpack)? };
//! assert_eq!(view[[1, 0]], 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use snafu::Snafu;

mod consumer;
mod producer;

pub use consumer::array_view_from_dlpack_mut_unchecked;

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

#[cfg(test)]
use crate::{DlpackElement, DlpackFlags, TryFromDlpack, allocation::dynamic, ffi::DLDevice};
#[cfg(test)]
use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Dimension, OwnedRepr, ShapeBuilder};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Local, ManagedTensorBase, allocation::fixed::make_test_tensor};
    use ndarray::{Array, arr2};

    type LegacyDlpack = Local<crate::ffi::DLManagedTensor>;
    type VersionedDlpack = Local<crate::ffi::DLManagedTensorVersioned>;

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
    fn legacy_2x3_dlpack() -> LegacyDlpack {
        managed_array(arr2(&[[1i32, 2, 3], [4, 5, 6]]))
    }

    /// The transpose of [`legacy_2x3_dlpack`]'s array, i.e. non-compact strides.
    fn legacy_3x2_transposed_dlpack() -> LegacyDlpack {
        let array = arr2(&[[1i32, 2, 3], [4, 5, 6]]);
        managed_array(array.reversed_axes().to_owned())
    }

    /// A `[[1, 2, 3], [4, 5, 6]]` versioned tensor carrying the given flags.
    fn versioned_2x3_dlpack(flags: DlpackFlags) -> VersionedDlpack {
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
        let dlpack: VersionedDlpack = managed_array(array);

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
        let dlpack: VersionedDlpack = managed_array(array);

        assert_eq!(dlpack.flags(), DlpackFlags::IS_COPIED);
    }

    #[test]
    fn ndarray_builder_allows_setting_read_only_safely() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let dlpack: VersionedDlpack =
            managed_array_with_flags(array, DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY);

        assert_eq!(
            dlpack.flags(),
            DlpackFlags::IS_COPIED | DlpackFlags::READ_ONLY
        );
    }

    #[test]
    fn owned_ndarray_to_versioned_dlpack_allows_unsafe_mutation() {
        let array = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.]).unwrap();
        let mut dlpack: VersionedDlpack = managed_array(array);

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
        let dlpack: LegacyDlpack = managed_array(array);

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
        let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&dlpack) }.unwrap();

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn borrowed_dlpack_to_ndarray_view_preserves_strides() {
        let dlpack = legacy_3x2_transposed_dlpack().into_foreign();
        let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&dlpack) }.unwrap();

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

        let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&dlpack) }.unwrap();
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
    fn mut_ndarray_view_updates_is_copied_tensor() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED).into_foreign();

        let mut view = unsafe { ArrayViewMutD::<i32>::try_from_dlpack(&mut dlpack) }.unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn mut_ndarray_view_accepts_caller_proven_exclusivity_without_is_copied() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::empty()).into_foreign();

        let mut view = unsafe { ArrayViewMutD::<i32>::try_from_dlpack(&mut dlpack) }.unwrap();
        view[[1, 2]] = 42;
    }

    #[test]
    fn mut_ndarray_view_rejects_overlapping_strides() {
        let data = Box::new(vec![1i32, 2]);
        let data_ptr = data.as_ptr().cast_mut().cast();
        let mut dlpack = make_test_tensor::<_, crate::ffi::DLManagedTensorVersioned, 2>(
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
            unsafe { ArrayViewMutD::<i32>::try_from_dlpack(&mut dlpack) },
            Err(Error::Shape { .. })
        ));
    }

    #[test]
    fn mut_ndarray_view_accepts_caller_proven_legacy_exclusivity() {
        let mut dlpack = legacy_2x3_dlpack().into_foreign();

        let mut view = unsafe { ArrayViewMutD::<i32>::try_from_dlpack(&mut dlpack) }.unwrap();
        view[[1, 2]] = 42;
    }

    #[test]
    fn try_from_dlpack_mutates_with_caller_proven_exclusivity() {
        let mut dlpack = versioned_2x3_dlpack(DlpackFlags::IS_COPIED).into_foreign();

        let mut view = unsafe { ArrayViewMutD::<i32>::try_from_dlpack(&mut dlpack) }.unwrap();
        view[[1, 2]] = 42;

        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 42]
        );
    }

    #[test]
    fn sliced_owned_ndarray_to_dlpack_exports_non_standard_strides() {
        let array = Array::from_shape_vec((2, 2).strides((4, 2)), (0i32..7).collect()).unwrap();
        let dlpack: LegacyDlpack = managed_array(array);
        let dlpack = dlpack.into_foreign();
        let view = unsafe { ArrayViewD::<i32>::try_from_dlpack(&dlpack) }.unwrap();

        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[4, 2]);
        assert_eq!(view[[0, 1]], 2);
        assert_eq!(view[[1, 1]], 6);
    }
}
