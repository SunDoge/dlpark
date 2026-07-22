use crate::{
    DlpackFlags, Local, ManagedTensorBase, OpaqueContext,
    ffi::{DLDataType, DLDevice},
    metadata::{Copied, Fixed},
};
use std::ffi::c_void;

pub(crate) fn fixed_tensor<C, M, const N: usize>(
    ctx: C,
    data: *mut c_void,
    dtype: DLDataType,
    device: DLDevice,
    shape: [i64; N],
    strides: [i64; N],
    flags: DlpackFlags,
) -> Local<M>
where
    C: OpaqueContext,
    M: ManagedTensorBase,
{
    let prepared = Fixed::new(Copied(shape), Copied(strides))
        .prepare::<M>()
        .unwrap();
    let mut initialized = prepared.initialize(ctx);
    initialized.set_data(data);
    initialized.set_dtype(dtype);
    initialized.set_device(device);
    initialized.set_flags_unchecked(flags);
    unsafe { initialized.finish() }
}
