use std::ffi::c_void;

use crate::ffi::ManagedTensor;

pub trait IntoDlpack {
    fn into_dlpack(self) -> Box<ManagedTensor>;
}

pub trait ToDlpack {
    fn to_dlpack(&self) -> Box<ManagedTensor>;
}

pub trait ToDlpackMut {
    fn to_dlpack_mut(&mut self) -> Box<ManagedTensor>;
}

pub trait FromDlpack {
    fn from_dlpack(pack: &ManagedTensor) -> Self;
}

pub trait FromDlpackMut {
    fn from_dlpack_mut(pack: &mut ManagedTensor) -> Self;
}

// pub trait TensorLike {
//     fn data_ptr(&self) -> *mut c_void;
//     fn shape(&self) -> &[i64];
//     fn strides(&self) -> Option<&[i64]>;
//     fn device(&self) -> Device;
//     fn dtype(&self) -> DataType;
//     fn byte_offset(&self) -> u64;
// }
