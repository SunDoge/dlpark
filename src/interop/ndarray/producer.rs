use crate::{
    DlpackElement, DlpackFlags, ManagedTensorBase,
    allocation::dynamic,
    ffi::DLDevice,
    metadata::{Copied, Dynamic},
};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use std::os::raw::c_void;

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
