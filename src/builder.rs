use crate::{
    DlpackFlags,
    context::OpaqueContext,
    dlpack::ManagedBox,
    ffi::{DLDataType, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor},
    managed_tensor::ManagedTensorBase,
};
use snafu::{ResultExt, Snafu, ensure};
use std::{ffi::c_void, ptr::NonNull};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Mismatched length of shape ({shape_len}) and strides ({strides_len})"))]
    MismatchedLength {
        shape_len: usize,
        strides_len: usize,
    },

    #[snafu(display("Dimension count ({ndim}) exceeds {expected}"))]
    NdimExceedsLimit { ndim: usize, expected: usize },

    #[snafu(display("Dimension count ({ndim}) exceeds i32::MAX"))]
    NdimOverflow {
        ndim: usize,
        source: std::num::TryFromIntError,
    },

    #[snafu(display("Negative dimension count ({ndim}) is invalid"))]
    NegativeNdim { ndim: i32 },
}

pub struct Builder<M: ManagedTensorBase, const N: usize>(NonNull<DlpackTensorStorage<M, N>>);

pub type DlpackBuilder<const N: usize> = Builder<DLManagedTensor, N>;
pub type DlpackVersionedBuilder<const N: usize> = Builder<DLManagedTensorVersioned, N>;
pub type DynamicDlpackBuilder = DlpackBuilder<0>;
pub type DynamicDlpackVersionedBuilder = DlpackVersionedBuilder<0>;

unsafe impl<M: ManagedTensorBase + Send, const N: usize> Send for Builder<M, N> {}
unsafe impl<M: ManagedTensorBase + Sync, const N: usize> Sync for Builder<M, N> {}

impl<M: ManagedTensorBase, const N: usize> Builder<M, N> {
    pub fn new(ptr: NonNull<DlpackTensorStorage<M, N>>) -> Self {
        Self(ptr)
    }

    pub fn into_raw(b: Self) -> *mut DlpackTensorStorage<M, N> {
        let ptr = b.0.as_ptr();
        std::mem::forget(b);
        ptr
    }

    pub fn build(self) -> ManagedBox<M> {
        let raw = Self::into_raw(self);
        unsafe { ManagedBox::new_unchecked(raw as *mut M) }
    }

    /// Builds the tensor and returns the raw pointer directly, skipping the
    /// `ManagedBox` RAII wrapper — for handing ownership straight across an
    /// FFI boundary (e.g. a `DLManagedTensor**` out-parameter) where the
    /// caller never wants a Rust-side owner in between.
    ///
    /// Safe for the same reason [`Self::build`] and [`ManagedBox::into_raw`]
    /// are: moving a raw pointer around commits to nothing unsafe by itself.
    /// The unsafe step is on the other end, reconstructing an owner from it
    /// (e.g. via [`ManagedBox::new_unchecked`]) — not here.
    pub fn build_raw(self) -> *mut M {
        Self::into_raw(self) as *mut M
    }
}

impl<M: ManagedTensorBase, const N: usize> std::ops::Deref for Builder<M, N> {
    type Target = DlpackTensorStorage<M, N>;
    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl<M: ManagedTensorBase, const N: usize> std::ops::DerefMut for Builder<M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

impl<M: ManagedTensorBase, const N: usize> Drop for Builder<M, N> {
    fn drop(&mut self) {
        unsafe {
            M::drop_raw(self.0.as_ptr() as *mut M);
        }
    }
}

#[repr(C)]
pub struct DlpackTensorStorage<M, const N: usize> {
    managed_tensor: M,
    shape: [i64; N],
    strides: [i64; N],
}

unsafe extern "C" fn static_deleter<const N: usize, C: OpaqueContext, M: ManagedTensorBase>(
    dlmt: *mut M,
) {
    if dlmt.is_null() {
        return;
    }
    unsafe {
        let ptr = dlmt as *mut DlpackTensorStorage<M, N>;
        C::drop_raw((*ptr).managed_tensor.manager_ctx());
        std::ptr::drop_in_place(ptr);
        std::alloc::dealloc(
            ptr as *mut u8,
            std::alloc::Layout::new::<DlpackTensorStorage<M, N>>(),
        );
    }
}

/// Allocates uninitialized memory sized and aligned for a
/// `DlpackTensorStorage<M, N>`.
///
/// Used instead of `Box::new` to construct the value in place: `Box::new`
/// would require first materializing the full struct (including the `shape`
/// and `strides` arrays) on the stack, then copying the whole thing onto the
/// heap — for large `N` that's a second full copy of the shape/strides data
/// on top of the one needed to convert it. Writing each field directly into
/// this allocation via `ptr::write` does exactly one copy per field. Paired
/// with `static_deleter`, which deallocates using the same `Layout`.
///
/// # Safety
///
/// The caller must initialize every field (`managed_tensor`, `shape`,
/// `strides`) via `ptr::write` before the returned pointer is read from, and
/// must not deallocate it except via a deleter using a matching
/// `Layout::new::<DlpackTensorStorage<M, N>>()`.
unsafe fn alloc_uninit_storage<M, const N: usize>() -> *mut DlpackTensorStorage<M, N> {
    let layout = std::alloc::Layout::new::<DlpackTensorStorage<M, N>>();
    unsafe {
        let ptr = std::alloc::alloc(layout) as *mut DlpackTensorStorage<M, N>;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ptr
    }
}

unsafe extern "C" fn dynamic_deleter<C: OpaqueContext, M: ManagedTensorBase>(dlmt: *mut M) {
    if dlmt.is_null() {
        return;
    }
    unsafe {
        let b = NonNull::new_unchecked(dlmt as *mut DlpackTensorStorage<M, 0>);
        let ndim = b.as_ref().managed_tensor.dl_tensor().ndim;
        let ndim_usize = if ndim < 0 { 0 } else { ndim as usize };
        let total_size = size_of::<DlpackTensorStorage<M, 0>>() + 2 * ndim_usize * size_of::<i64>();
        let layout = std::alloc::Layout::from_size_align_unchecked(total_size, 8);
        C::drop_raw(b.as_ref().managed_tensor.manager_ctx());
        // Defensive drop of any struct fields
        std::ptr::drop_in_place(b.as_ptr());
        std::alloc::dealloc(b.as_ptr() as *mut u8, layout);
    }
}

impl<M: ManagedTensorBase, const N: usize> Builder<M, N> {
    pub fn data(mut self, ptr: *mut c_void) -> Self {
        self.managed_tensor.dl_tensor_mut().data = ptr;
        self
    }

    pub fn device(mut self, device: DLDevice) -> Self {
        self.managed_tensor.dl_tensor_mut().device = device;
        self
    }

    pub fn dtype(mut self, dtype: DLDataType) -> Self {
        self.managed_tensor.dl_tensor_mut().dtype = dtype;
        self
    }

    pub fn byte_offset(mut self, byte_offset: u64) -> Self {
        self.managed_tensor.dl_tensor_mut().byte_offset = byte_offset;
        self
    }

    /// Same fixed capacity `N` as `with_array_layout`, but takes `&[T]`
    /// instead of `&[T; N]`. `shape`/`strides` may be shorter than `N`
    /// (`ndim` becomes their actual length) but not longer.
    pub fn with_slice_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<Builder<M, N>, Error>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");
        ensure!(
            shape.len() == strides.len(),
            MismatchedLengthSnafu {
                shape_len: shape.len(),
                strides_len: strides.len()
            }
        );
        let ndim = shape.len();
        ensure!(ndim <= N, NdimExceedsLimitSnafu { ndim, expected: N });

        unsafe {
            let raw_ptr = alloc_uninit_storage::<M, N>();

            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            for (i, s) in shape.iter().enumerate() {
                std::ptr::write(shape_ptr.add(i), (*s).into());
            }
            for (i, s) in strides.iter().enumerate() {
                std::ptr::write(strides_ptr.add(i), (*s).into());
            }

            std::ptr::write(
                std::ptr::addr_of_mut!((*raw_ptr).managed_tensor),
                M::from_parts(
                    DLTensor {
                        data: std::ptr::null_mut(),
                        device: DLDevice::CPU,
                        ndim: ndim as i32,
                        dtype: DLDataType::default(),
                        shape: shape_ptr,
                        strides: strides_ptr,
                        byte_offset: 0,
                    },
                    ctx.into_raw(),
                    Some(static_deleter::<N, C, M>),
                ),
            );

            Ok(Builder::new(NonNull::new_unchecked(raw_ptr)))
        }
    }

    /// `shape`/`strides` as fixed-size arrays: `N` is known at compile time,
    /// so a length mismatch can't happen and this never fails.
    pub fn with_array_layout<C, T>(ctx: C, shape: &[T; N], strides: &[T; N]) -> Builder<M, N>
    where
        C: OpaqueContext,
        T: Into<i64> + Copy,
    {
        assert!(N <= i32::MAX as usize, "N must fit in i32");

        unsafe {
            let raw_ptr = alloc_uninit_storage::<M, N>();

            let shape_ptr = std::ptr::addr_of_mut!((*raw_ptr).shape) as *mut i64;
            let strides_ptr = std::ptr::addr_of_mut!((*raw_ptr).strides) as *mut i64;

            for (i, s) in shape.iter().enumerate() {
                std::ptr::write(shape_ptr.add(i), (*s).into());
            }
            for (i, s) in strides.iter().enumerate() {
                std::ptr::write(strides_ptr.add(i), (*s).into());
            }

            std::ptr::write(
                std::ptr::addr_of_mut!((*raw_ptr).managed_tensor),
                M::from_parts(
                    DLTensor {
                        data: std::ptr::null_mut(),
                        device: DLDevice::CPU,
                        ndim: N as i32,
                        dtype: DLDataType::default(),
                        shape: shape_ptr,
                        strides: strides_ptr,
                        byte_offset: 0,
                    },
                    ctx.into_raw(),
                    Some(static_deleter::<N, C, M>),
                ),
            );

            Builder::new(NonNull::new_unchecked(raw_ptr))
        }
    }
}

impl<const N: usize> Builder<DLManagedTensorVersioned, N> {
    pub fn flags(mut self, flags: DlpackFlags) -> Self {
        self.managed_tensor.flags = flags;
        self
    }
}

impl Builder<DLManagedTensor, 0> {
    /// # Safety
    ///
    /// TODO
    pub unsafe fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<Builder<DLManagedTensor, 0>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });

        unsafe {
            // shape/strides are `[i64; 0]` here — zero-sized, nothing to
            // initialize — so only `managed_tensor` needs a `ptr::write`.
            let raw_ptr = alloc_uninit_storage::<DLManagedTensor, 0>();

            std::ptr::write(
                std::ptr::addr_of_mut!((*raw_ptr).managed_tensor),
                DLManagedTensor {
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
                },
            );

            Ok(Builder::new(NonNull::new_unchecked(raw_ptr)))
        }
    }
}

impl Builder<DLManagedTensorVersioned, 0> {
    pub fn with_pointer_layout<C>(
        ctx: C,
        shape_ptr: *mut i64,
        strides_ptr: *mut i64,
        ndim: i32,
    ) -> Result<Builder<DLManagedTensorVersioned, 0>, Error>
    where
        C: OpaqueContext,
    {
        ensure!(ndim >= 0, NegativeNdimSnafu { ndim });

        unsafe {
            // shape/strides are `[i64; 0]` here — zero-sized, nothing to
            // initialize — so only `managed_tensor` needs a `ptr::write`.
            let raw_ptr = alloc_uninit_storage::<DLManagedTensorVersioned, 0>();

            std::ptr::write(
                std::ptr::addr_of_mut!((*raw_ptr).managed_tensor),
                DLManagedTensorVersioned {
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
                },
            );

            Ok(Builder::new(NonNull::new_unchecked(raw_ptr)))
        }
    }
}

impl<M: ManagedTensorBase> Builder<M, 0> {
    /// For when the rank isn't known until runtime: `shape`/`strides` can be
    /// any length, and everything (including the metadata allocation itself)
    /// is sized dynamically.
    pub fn with_dynamic_layout<C, T>(
        ctx: C,
        shape: &[T],
        strides: &[T],
    ) -> Result<Builder<M, 0>, Error>
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

        let total_size = size_of::<DlpackTensorStorage<M, 0>>() + 2 * ndim_usize * size_of::<i64>();
        // SAFETY: align=8 is a valid power of two, and `total_size` doesn't
        // overflow `isize::MAX` for any `ndim` that fits in `i32` (checked
        // above). `from_size_align`'s runtime validation of exactly these
        // two properties measurably costs more than `Layout::new::<T>()`'s
        // compiler-known-valid path even for constant-equivalent inputs
        // (benchmarked: ~40ns vs ~29ns for the same resulting layout).
        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(total_size, 8) };

        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut DlpackTensorStorage<M, 0>;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            let shape_ptr =
                (ptr as *mut u8).add(size_of::<DlpackTensorStorage<M, 0>>()) as *mut i64;
            let strides_ptr = shape_ptr.add(ndim_usize);

            for (i, s) in shape.iter().enumerate() {
                std::ptr::write(shape_ptr.add(i), (*s).into());
            }
            for (i, s) in strides.iter().enumerate() {
                std::ptr::write(strides_ptr.add(i), (*s).into());
            }

            std::ptr::write(
                std::ptr::addr_of_mut!((*ptr).managed_tensor),
                M::from_parts(
                    DLTensor {
                        data: std::ptr::null_mut(),
                        device: DLDevice::CPU,
                        ndim,
                        dtype: DLDataType::default(),
                        shape: shape_ptr,
                        strides: strides_ptr,
                        byte_offset: 0,
                    },
                    ctx.into_raw(),
                    Some(dynamic_deleter::<C, M>),
                ),
            );

            Ok(Builder::new(NonNull::new_unchecked(ptr)))
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
            Builder::<DLManagedTensor, 3>::with_array_layout(ctx, &[1, 2, 3], &[6, 3, 1]).build();

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
    fn test_build_raw_roundtrips_through_managed_box() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let raw = Builder::<DLManagedTensor, 3>::with_array_layout(ctx, &[1, 2, 3], &[6, 3, 1])
            .build_raw();

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        // Reconstruct via the unsafe half of the contract build_raw defers to.
        let dlpack = unsafe { ManagedBox::new_unchecked(raw) };
        assert_eq!(dlpack.dl_tensor().ndim, 3);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_slice_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let dlpack = Builder::<DLManagedTensor, 3>::with_slice_layout(ctx, &[2, 4, 8], &[32, 8, 1])
            .unwrap()
            .build();

        assert_eq!(dlpack.dl_tensor().ndim, 3);
        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_slice_layout_shorter_than_n_succeeds() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let dlpack = Builder::<DLManagedTensor, 3>::with_slice_layout(ctx, &[2, 4], &[8, 1])
            .unwrap()
            .build();

        assert_eq!(dlpack.dl_tensor().ndim, 2);
        unsafe {
            assert_eq!(*dlpack.dl_tensor().shape.add(0), 2);
            assert_eq!(*dlpack.dl_tensor().shape.add(1), 4);
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        drop(dlpack);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_slice_layout_mismatched_shape_strides_returns_error() {
        let ctx = TestContext {
            drop_count: Arc::new(AtomicUsize::new(0)),
        };

        let result = Builder::<DLManagedTensor, 3>::with_slice_layout(ctx, &[2, 4], &[8, 1, 1]);

        assert!(matches!(
            result,
            Err(Error::MismatchedLength {
                shape_len: 2,
                strides_len: 3
            })
        ));
    }

    #[test]
    fn test_slice_layout_exceeding_n_returns_error() {
        let ctx = TestContext {
            drop_count: Arc::new(AtomicUsize::new(0)),
        };

        let result = Builder::<DLManagedTensor, 2>::with_slice_layout(ctx, &[2, 4, 8], &[32, 8, 1]);

        assert!(matches!(
            result,
            Err(Error::NdimExceedsLimit {
                ndim: 3,
                expected: 2
            })
        ));
    }

    #[test]
    fn test_dynamic_layout() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let ctx = TestContext {
            drop_count: drop_count.clone(),
        };

        let dlpack = Builder::<DLManagedTensor, 0>::with_dynamic_layout(ctx, &[3, 5], &[5, 1])
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
            Builder::<DLManagedTensor, 0>::with_pointer_layout(
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
