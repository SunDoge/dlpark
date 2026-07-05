use crate::{
    DlpackFlags,
    context::OpaqueContext,
    dlpack::Dlpack,
    ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor},
    managed_tensor::ManagedTensor,
};
use snafu::{ResultExt, Snafu, ensure};
use std::{os::raw::c_void, ptr::NonNull};

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

pub struct DlpackBuilder<M: ManagedTensor, const N: usize>(NonNull<DlpackBox<M, N>>);

unsafe impl<M: ManagedTensor + Send, const N: usize> Send for DlpackBuilder<M, N> {}
unsafe impl<M: ManagedTensor + Sync, const N: usize> Sync for DlpackBuilder<M, N> {}

impl<M: ManagedTensor, const N: usize> DlpackBuilder<M, N> {
    pub fn new(ptr: NonNull<DlpackBox<M, N>>) -> Self {
        Self(ptr)
    }

    pub fn into_raw(b: Self) -> *mut DlpackBox<M, N> {
        let ptr = b.0.as_ptr();
        std::mem::forget(b);
        ptr
    }

    pub fn build(self) -> Dlpack<M> {
        let raw = Self::into_raw(self);
        unsafe { Dlpack::new_unchecked(raw as *mut M) }
    }
}

impl<M: ManagedTensor, const N: usize> std::ops::Deref for DlpackBuilder<M, N> {
    type Target = DlpackBox<M, N>;
    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl<M: ManagedTensor, const N: usize> std::ops::DerefMut for DlpackBuilder<M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

impl<M: ManagedTensor, const N: usize> Drop for DlpackBuilder<M, N> {
    fn drop(&mut self) {
        unsafe {
            M::call_deleter(self.0.as_ptr() as *mut M);
        }
    }
}

#[repr(C)]
pub struct DlpackBox<M, const N: usize> {
    managed_tensor: M,
    shape: [i64; N],
    strides: [i64; N],
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
        // Box drop will automatically run drop_in_place and free memory
    }
}

unsafe extern "C" fn dynamic_deleter<C: OpaqueContext, M: ManagedTensor>(dlmt: *mut M) {
    if dlmt.is_null() {
        return;
    }
    unsafe {
        let b = NonNull::new_unchecked(dlmt as *mut DlpackBox<M, 0>);
        let ndim = b.as_ref().managed_tensor.get_dltensor().ndim;
        let ndim_usize = if ndim < 0 { 0 } else { ndim as usize };
        let total_size = size_of::<DlpackBox<M, 0>>() + 2 * ndim_usize * size_of::<i64>();
        let layout = std::alloc::Layout::from_size_align_unchecked(total_size, 8);
        C::drop_raw(b.as_ref().managed_tensor.manager_ctx_ptr());
        // Defensive drop of any struct fields
        std::ptr::drop_in_place(b.as_ptr());
        std::alloc::dealloc(b.as_ptr() as *mut u8, layout);
    }
}

impl<M: ManagedTensor, const N: usize> DlpackBuilder<M, N> {
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

impl<const N: usize> DlpackBuilder<DLManagedTensor, N> {
    pub fn with_slice_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> DlpackBuilder<DLManagedTensor, N>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), N, "shape length must match N");
        assert_eq!(strides.len(), N, "strides length must match N");
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let mut shape_arr = [0i64; N];
        for (i, s) in shape.iter().enumerate() {
            shape_arr[i] = (*s).into();
        }

        let mut strides_arr = [0i64; N];
        for (i, s) in strides.iter().enumerate() {
            strides_arr[i] = (*s).into();
        }

        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_arr,
            strides: strides_arr,
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            (*raw_ptr).managed_tensor = DLManagedTensor {
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

            DlpackBuilder::new(NonNull::new_unchecked(raw_ptr))
        }
    }

    pub fn with_array_layout<C, T>(
        ctx: C,
        shape: [T; N],
        strides: [T; N],
    ) -> DlpackBuilder<DLManagedTensor, N>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape.map(|x| x.into()),
            strides: strides.map(|x| x.into()),
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            (*raw_ptr).managed_tensor = DLManagedTensor {
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

            DlpackBuilder::new(NonNull::new_unchecked(raw_ptr))
        }
    }
}

impl<const N: usize> DlpackBuilder<DLManagedTensorVersioned, N> {
    pub fn with_slice_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> DlpackBuilder<DLManagedTensorVersioned, N>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), N, "shape length must match N");
        assert_eq!(strides.len(), N, "strides length must match N");
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let mut shape_arr = [0i64; N];
        for (i, s) in shape.iter().enumerate() {
            shape_arr[i] = (*s).into();
        }

        let mut strides_arr = [0i64; N];
        for (i, s) in strides.iter().enumerate() {
            strides_arr[i] = (*s).into();
        }

        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_arr,
            strides: strides_arr,
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            (*raw_ptr).managed_tensor = DLManagedTensorVersioned {
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

            DlpackBuilder::new(NonNull::new_unchecked(raw_ptr))
        }
    }

    pub fn with_array_layout<C, T>(
        ctx: C,
        shape: [T; N],
        strides: [T; N],
    ) -> DlpackBuilder<DLManagedTensorVersioned, N>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape.map(|x| x.into()),
            strides: strides.map(|x| x.into()),
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            (*raw_ptr).managed_tensor = DLManagedTensorVersioned {
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

            DlpackBuilder::new(NonNull::new_unchecked(raw_ptr))
        }
    }

    pub fn flags(mut self, flags: DlpackFlags) -> Self {
        self.managed_tensor.flags = flags;
        self
    }
}

impl DlpackBuilder<DLManagedTensor, 0> {
    /// # Safety
    ///
    /// TODO
    pub unsafe fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<DlpackBuilder<DLManagedTensor, 0>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });
        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: [],
            strides: [],
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            (*raw_ptr).managed_tensor = DLManagedTensor {
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

            Ok(DlpackBuilder::new(NonNull::new_unchecked(raw_ptr)))
        }
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<DlpackBuilder<DLManagedTensor, 0>, Error>
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

        let total_size =
            size_of::<DlpackBox<DLManagedTensor, 0>>() + 2 * ndim_usize * size_of::<i64>();
        let layout = std::alloc::Layout::from_size_align(total_size, 8).unwrap();

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut DlpackBox<DLManagedTensor, 0>;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let shape_ptr =
                (ptr as *mut u8).add(size_of::<DlpackBox<DLManagedTensor, 0>>()) as *mut i64;
            let strides_ptr = shape_ptr.add(ndim_usize);

            for (i, s) in shape.iter().enumerate() {
                std::ptr::write(shape_ptr.add(i), (*s).into());
            }
            for (i, s) in strides.iter().enumerate() {
                std::ptr::write(strides_ptr.add(i), (*s).into());
            }

            let managed_tensor = DLManagedTensor {
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
                deleter: Some(dynamic_deleter::<C, _>),
            };

            std::ptr::write(
                std::ptr::addr_of_mut!((*ptr).managed_tensor),
                managed_tensor,
            );

            Ok(DlpackBuilder::new(NonNull::new_unchecked(ptr)))
        }
    }
}

impl DlpackBuilder<DLManagedTensorVersioned, 0> {
    pub fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<DlpackBuilder<DLManagedTensorVersioned, 0>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });
        let boxed = Box::new(DlpackBox {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: [],
            strides: [],
        });

        let raw_ptr = Box::into_raw(boxed);

        unsafe {
            (*raw_ptr).managed_tensor = DLManagedTensorVersioned {
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

            Ok(DlpackBuilder::new(NonNull::new_unchecked(raw_ptr)))
        }
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<DlpackBuilder<DLManagedTensorVersioned, 0>, Error>
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

        let total_size =
            size_of::<DlpackBox<DLManagedTensorVersioned, 0>>() + 2 * ndim_usize * size_of::<i64>();
        let layout = std::alloc::Layout::from_size_align(total_size, 8).unwrap();

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut DlpackBox<DLManagedTensorVersioned, 0>;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let shape_ptr = (ptr as *mut u8)
                .add(size_of::<DlpackBox<DLManagedTensorVersioned, 0>>())
                as *mut i64;
            let strides_ptr = shape_ptr.add(ndim_usize);

            for (i, s) in shape.iter().enumerate() {
                std::ptr::write(shape_ptr.add(i), (*s).into());
            }
            for (i, s) in strides.iter().enumerate() {
                std::ptr::write(strides_ptr.add(i), (*s).into());
            }

            let managed_tensor = DLManagedTensorVersioned {
                version: crate::ffi::DLPackVersion::default(),
                manager_ctx: ctx.into_raw(),
                deleter: Some(dynamic_deleter::<C, _>),
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

            std::ptr::write(
                std::ptr::addr_of_mut!((*ptr).managed_tensor),
                managed_tensor,
            );

            Ok(DlpackBuilder::new(NonNull::new_unchecked(ptr)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct TestContext {
        drop_count: Arc<AtomicUsize>,
    }

    impl OpaqueContext for TestContext {
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
    }

    #[test]
    fn test_array_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let dlpack =
            DlpackBuilder::<DLManagedTensor, 3>::with_array_layout(ctx, [1, 2, 3], [6, 3, 1])
                .build();

        assert_eq!(dlpack.dl_tensor().ndim, 3);
        unsafe {
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 1);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 2);
            assert_eq!(*dlpack.dl_tensor().shape.add(2), 3);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 6);
            assert_eq!(*dlpack.dl_tensor().strides.add(1), 3);
            assert_eq!(*dlpack.dl_tensor().strides.add(2), 1);
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack); // should call deleter, which frees the box and drops context
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_slice_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let dlpack =
            DlpackBuilder::<DLManagedTensor, 3>::with_slice_layout(ctx, &[2, 4, 8], &[32, 8, 1])
                .build();

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

        let dlpack =
            DlpackBuilder::<DLManagedTensor, 0>::with_dynamic_layout(ctx, &[3, 5], &[5, 1])
                .unwrap()
                .build();

        assert_eq!(dlpack.dl_tensor().ndim, 2);
        unsafe {
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 3);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 5);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 5);
            assert_eq!(*dlpack.dl_tensor().strides.add(1), 1);
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_pointer_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let mut shape = [10, 20];
        let mut strides = [20, 1];

        let dlpack = unsafe {
            DlpackBuilder::<DLManagedTensor, 0>::with_pointer_layout(
                ctx,
                shape.as_mut_ptr(),
                strides.as_mut_ptr(),
                2,
            )
            .unwrap()
            .build()
        };

        assert_eq!(dlpack.dl_tensor().ndim, 2);
        unsafe {
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 10);
            assert_eq!(*dlpack.dl_tensor().strides.add(0), 20);
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }
}
