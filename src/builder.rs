use crate::{
    context::OpaqueContext,
    ffi::{self, DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor},
    managed_tensor::AsManagedTensor,
    Flags,
};
use std::{alloc::Layout, os::raw::c_void, ptr::NonNull};

#[repr(C)]
pub struct DlpackBox<M, const N: usize> {
    managed_tensor: M,
    shape: [i64; N],
    strides: [i64; N],
}

unsafe extern "C" fn static_deleter<const N: usize, C: OpaqueContext, M: AsManagedTensor>(
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

unsafe extern "C" fn dynamic_deleter<C: OpaqueContext, M: AsManagedTensor>(dlmt: *mut M) {
    if dlmt.is_null() {
        return;
    }
    unsafe {
        let b = NonNull::new_unchecked(dlmt as *mut DlpackBox<M, 0>);
        let ndim = b.as_ref().managed_tensor.get_dltensor().ndim;
        assert!(ndim >= 0, "ndim must be non-negative in deleter");
        let ndim_usize = ndim as usize;
        let total_size = size_of::<DlpackBox<M, 0>>() + 2 * ndim_usize * size_of::<i64>();
        let layout = Layout::from_size_align_unchecked(total_size, 8);
        C::drop_raw(b.as_ref().managed_tensor.manager_ctx_ptr());
        // Defensive drop of any struct fields
        std::ptr::drop_in_place(b.as_ptr());
        std::alloc::dealloc(b.as_ptr() as *mut u8, layout);
    };
}

impl<M: AsManagedTensor, const N: usize> DlpackBox<M, N> {
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
    pub fn with_owned_layout<C, T>(ctx: C, shape: &[T], strides: &[T]) -> Box<Self>
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

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_arr,
            strides: strides_arr,
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
    pub fn with_owned_layout<C, T>(ctx: C, shape: &[T], strides: &[T]) -> Box<Self>
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

        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: shape_arr,
            strides: strides_arr,
        });

        let shape_ptr = boxed.shape.as_mut_ptr();
        let strides_ptr = boxed.strides.as_mut_ptr();

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<N, C, _>),
            flags: Flags::empty(),
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

    pub fn flags(mut self, flags: Flags) -> Self {
        self.managed_tensor.flags = flags;
        self
    }
}

impl DlpackBox<DLManagedTensor, 0> {
    pub fn with_borrowed_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Box<Self>
    where
        C: OpaqueContext,
    {
        assert!(ndim >= 0, "ndim must be non-negative");
        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: [],
            strides: [],
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

        boxed
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), strides.len(), "shape and strides must have the same length");
        let ndim_usize = shape.len();
        assert!(ndim_usize <= i32::MAX as usize, "ndim must fit in i32");
        let ndim = ndim_usize as i32;

        let total_size = size_of::<Self>() + 2 * ndim_usize * size_of::<i64>();
        let layout = Layout::from_size_align(total_size, 8).unwrap();

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let shape_ptr = (ptr as *mut u8).add(size_of::<Self>()) as *mut i64;
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

            std::ptr::write(std::ptr::addr_of_mut!((*ptr).managed_tensor), managed_tensor);

            Box::from_raw(ptr)
        }
    }
}

impl DlpackBox<DLManagedTensorVersioned, 0> {
    pub fn with_borrowed_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Box<Self>
    where
        C: OpaqueContext,
    {
        assert!(ndim >= 0, "ndim must be non-negative");
        let mut boxed = Box::new(Self {
            managed_tensor: unsafe { std::mem::zeroed() },
            shape: [],
            strides: [],
        });

        boxed.managed_tensor = DLManagedTensorVersioned {
            version: crate::ffi::DLPackVersion::default(),
            manager_ctx: ctx.into_raw(),
            deleter: Some(static_deleter::<0, C, _>),
            flags: Flags::empty(),
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

        boxed
    }

    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Box<Self>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert_eq!(shape.len(), strides.len(), "shape and strides must have the same length");
        let ndim_usize = shape.len();
        assert!(ndim_usize <= i32::MAX as usize, "ndim must fit in i32");
        let ndim = ndim_usize as i32;

        let total_size = size_of::<Self>() + 2 * ndim_usize * size_of::<i64>();
        let layout = Layout::from_size_align(total_size, 8).unwrap();

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let shape_ptr = (ptr as *mut u8).add(size_of::<Self>()) as *mut i64;
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
                flags: Flags::empty(),
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

            std::ptr::write(std::ptr::addr_of_mut!((*ptr).managed_tensor), managed_tensor);

            Box::from_raw(ptr)
        }
    }
}
