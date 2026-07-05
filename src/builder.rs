use crate::{
    context::OpaqueContext,
    ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor},
    managed_tensor::ManagedTensor,
    DlpackFlags,
};
use snafu::{ResultExt, Snafu, ensure};
use std::os::raw::c_void;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Mismatched length of shape ({shape_len}) and strides ({strides_len})"))]
    MismatchedLength {
        shape_len: usize,
        strides_len: usize,
    },

    #[snafu(display("Dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        ndim: usize,
        source: std::num::TryFromIntError,
    },

    #[snafu(display("Negative dimension count ({ndim}) is invalid"))]
    NegativeNdim { ndim: i32 },
}

#[repr(C)]
pub struct DlpackBox<M, const N: usize> {
    managed_tensor: M,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

unsafe extern "C" fn static_deleter<const N: usize, C: OpaqueContext, M: ManagedTensor>(
    dlmt: *mut M,
) {
    if dlmt.is_null() {
        return;
    }
    unsafe {
        let b = Box::from_raw(dlmt as *mut DlpackBox<M, N>);
        C::drop_raw(b.managed_tensor.manager_ctx_ptr());
        // Box drop will automatically drop shape and strides heap allocations
    }
}

impl<M: ManagedTensor, const N: usize> DlpackBox<M, N> {
    pub fn data(mut self, ptr: *mut c_void) -> Self {
        self.managed_tensor.get_dltensor_mut().data = ptr;
        self
    }

    pub fn device(mut self, device: DLDevice) -> Self {
        self.managed_tensor.get_dltensor_mut().device = device;
        self
    }

    pub fn dtype(mut self, dtype: DLDataType) -> Self {
        self.managed_tensor.get_dltensor_mut().dtype = dtype;
        self
    }

    pub fn byte_offset(mut self, byte_offset: u64) -> Self {
        self.managed_tensor.get_dltensor_mut().byte_offset = byte_offset;
        self
    }
}

impl<const N: usize> DlpackBox<DLManagedTensor, N> {
    pub fn with_slice_layout<C, T>(ctx: C, shape: &[T], strides: &[T]) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), N, "shape length must match N");
        assert_eq!(strides.len(), N, "strides length must match N");
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim: N as i32,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<N, C, _>),
        };

        boxed
    }

    pub fn with_array_layout<C, T>(ctx: C, shape: [T; N], strides: [T; N]) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim: N as i32,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<N, C, _>),
        };

        boxed
    }
}

impl<const N: usize> DlpackBox<DLManagedTensorVersioned, N> {
    pub fn with_slice_layout<C, T>(ctx: C, shape: &[T], strides: &[T]) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), N, "shape length must match N");
        assert_eq!(strides.len(), N, "strides length must match N");
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<N, C, _>),
            flags: DlpackFlags::empty(),
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim: N as i32,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
        };

        boxed
    }

    pub fn with_array_layout<C, T>(ctx: C, shape: [T; N], strides: [T; N]) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<N, C, _>),
            flags: DlpackFlags::empty(),
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim: N as i32,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
        };

        boxed
    }

    pub fn flags(mut self, flags: DlpackFlags) -> Self {
        self.managed_tensor.flags = flags;
        self
    }
}

impl DlpackBox<DLManagedTensor, 0> {
    pub fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<Box<Self>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });
        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: Box::default(),
            strides: Box::default(),
        });

        boxed.managed_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<0, C, _>),
        };

        Ok(boxed)
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<Box<Self>, Error>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        ensure!(
            shape.len() == strides.len(),
            MismatchedLengthSnafu {
                shape_len: shape.len(),
                strides_len: strides.len()
            }
        );
        let ndim_usize = shape.len();
        let ndim: i32 = ndim_usize
            .try_into()
            .context(NdimOverflowSnafu { ndim: ndim_usize })?;

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<0, C, _>),
        };

        Ok(boxed)
    }
}

impl DlpackBox<DLManagedTensorVersioned, 0> {
    pub fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<Box<Self>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });
        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: Box::default(),
            strides: Box::default(),
        });

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<0, C, _>),
            flags: DlpackFlags::empty(),
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
        };

        Ok(boxed)
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<Box<Self>, Error>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        ensure!(
            shape.len() == strides.len(),
            MismatchedLengthSnafu {
                shape_len: shape.len(),
                strides_len: strides.len(),
            }
        );
        let ndim_usize = shape.len();
        let ndim: i32 = ndim_usize
            .try_into()
            .context(NdimOverflowSnafu { ndim: ndim_usize })?;

        let shape_boxed: Box<[i64]> = shape.iter().map(|&s| s.into()).collect();
        let strides_boxed: Box<[i64]> = strides.iter().map(|&s| s.into()).collect();

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_boxed,
            strides: strides_boxed,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<0, C, _>),
            flags: DlpackFlags::empty(),
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(),
                device: DLDevice::CPU,
                ndim,
                dtype: DLDataType::default(),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: 0,
            },
        };

        Ok(boxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dlpack;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone)]
    struct TestContext {
        drop_count: Arc<AtomicUsize>,
    }

    impl OpaqueContext for TestContext {
        type Target = Arc<AtomicUsize>;

        fn into_raw(self) -> *mut c_void {
            let boxed = Box::new(self);
            Box::into_raw(boxed) as *mut c_void
        }

        unsafe fn drop_raw(raw: *mut c_void) {
            if !raw.is_null() {
                let boxed = unsafe { Box::from_raw(raw as *mut TestContext) };
                boxed.drop_count.fetch_add(1, Ordering::SeqCst);
            }
        }

        unsafe fn as_ref<'a>(raw: *mut c_void) -> &'a Self::Target {
            let r = unsafe { &*(raw as *mut TestContext) };
            &r.drop_count
        }
    }

    #[test]
    fn test_array_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let boxed = DlpackBox::<DLManagedTensor, 3>::with_array_layout(ctx, [1, 2, 3], [6, 3, 1]);

        let raw_box = Box::into_raw(boxed);
        unsafe {
            let dlpack = Dlpack::new(raw_box as *mut DLManagedTensor).unwrap();
            assert_eq!(dlpack.dl_tensor().ndim, 3);
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 1);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 2);
            assert_eq!(*dlpack.dl_tensor().shape.add(2), 3);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 6);
            assert_eq!(*dlpack.dl_tensor().strides.add(1), 3);
            assert_eq!(*dlpack.dl_tensor().strides.add(2), 1);

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            drop(dlpack); // should call deleter, which frees the box and drops context
            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn test_slice_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let boxed = DlpackBox::<DLManagedTensor, 3>::with_slice_layout(ctx, &[2, 4, 8], &[32, 8, 1]);

        let raw_box = Box::into_raw(boxed);
        let dlpack = Dlpack::new(raw_box as *mut DLManagedTensor).unwrap();
        assert_eq!(dlpack.dl_tensor().ndim, 3);
        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_dynamic_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let boxed = DlpackBox::<DLManagedTensor, 0>::with_dynamic_layout(ctx, &[3, 5], &[5, 1]).unwrap();

        let raw_box = Box::into_raw(boxed);
        unsafe {
            let dlpack = Dlpack::new(raw_box as *mut DLManagedTensor).unwrap();
            assert_eq!(dlpack.dl_tensor().ndim, 2);
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 3);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 5);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 5);
            assert_eq!(*dlpack.dl_tensor().strides.add(1), 1);

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            drop(dlpack);
            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn test_pointer_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let mut shape = [10, 20];
        let mut strides = [20, 1];

        let boxed = DlpackBox::<DLManagedTensor, 0>::with_pointer_layout(
            ctx,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            2,
        ).unwrap();

        let raw_box = Box::into_raw(boxed);
        unsafe {
            let dlpack = Dlpack::new(raw_box as *mut DLManagedTensor).unwrap();
            assert_eq!(dlpack.dl_tensor().ndim, 2);
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 10);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 20);

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            drop(dlpack);
            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        }
    }
}
