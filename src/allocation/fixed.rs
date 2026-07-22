//! Fixed-rank metadata allocation without generic const expressions.

use super::{Error, allocate, empty_tensor};
use crate::{ManagedBox, ManagedTensorBase, OpaqueContext};
use std::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

/// Storage selected by fixed-rank shape or strides metadata.
///
/// # Safety
///
/// `initialize` must fully initialize a value at `storage`. `Storage` must be
/// suitable for embedding directly in the managed tensor allocation.
pub unsafe trait Storage<const N: usize> {
    type Value: Copy;

    /// Initializes the inline storage at `storage`.
    ///
    /// # Safety
    ///
    /// `storage` must be non-null, aligned, and writable for one uninitialized
    /// value of `Self::Value`.
    unsafe fn initialize(storage: *mut Self::Value);
}

/// Borrowed metadata which requires no inline storage.
pub struct Borrowed;

unsafe impl<const N: usize> Storage<N> for Borrowed {
    type Value = ();

    unsafe fn initialize(storage: *mut Self::Value) {
        unsafe { storage.write(()) };
    }
}

/// Metadata stored inline as `[i64; N]`.
pub struct Copied;

unsafe impl<const N: usize> Storage<N> for Copied {
    type Value = [i64; N];

    unsafe fn initialize(storage: *mut Self::Value) {
        unsafe { storage.write([0; N]) };
    }
}

/// An uninitialized fixed-rank managed tensor allocation.
pub struct Allocation<M, const N: usize, Shape: Storage<N> = Copied, Strides: Storage<N> = Copied> {
    managed: NonNull<M>,
    shape: NonNull<Shape::Value>,
    strides: NonNull<Strides::Value>,
    layout: Layout,
}

impl<M, const N: usize, Shape, Strides> Allocation<M, N, Shape, Strides>
where
    M: ManagedTensorBase,
    Shape: Storage<N>,
    Strides: Storage<N>,
{
    /// Allocates the managed tensor and selected metadata storage.
    pub fn allocate() -> Result<Self, Error> {
        i32::try_from(N).map_err(|_| Error::NdimOverflow { ndim: N })?;
        let parts = allocation_parts::<M, N, Shape, Strides>()?;
        let managed = allocate::<M>(parts.layout);
        unsafe {
            let base = managed.as_ptr().cast::<u8>();
            let shape = base.add(parts.shape).cast::<Shape::Value>();
            let strides = base.add(parts.strides).cast::<Strides::Value>();
            Shape::initialize(shape);
            Strides::initialize(strides);
            Ok(Self {
                managed,
                shape: NonNull::new_unchecked(shape),
                strides: NonNull::new_unchecked(strides),
                layout: parts.layout,
            })
        }
    }

    /// Returns the storage selected for shape metadata.
    pub fn shape_storage_mut(&mut self) -> &mut Shape::Value {
        unsafe { self.shape.as_mut() }
    }

    /// Returns the storage selected for strides metadata.
    pub fn strides_storage_mut(&mut self) -> &mut Strides::Value {
        unsafe { self.strides.as_mut() }
    }

    /// Initializes the managed tensor and installs its context and deleter.
    pub fn initialize<C: OpaqueContext>(self, ctx: C) -> Initialized<M, N, Shape, Strides> {
        let this = ManuallyDrop::new(self);
        unsafe {
            this.managed.as_ptr().write(M::from_parts(
                empty_tensor(N as i32),
                ctx.into_raw(),
                Some(drop_allocation::<C, M, N, Shape, Strides>),
            ));
            super::Initialized {
                managed: ManagedBox::new_unchecked(this.managed.as_ptr()),
                storage: Metadata {
                    shape: this.shape,
                    strides: this.strides,
                },
            }
        }
    }
}

impl<M, const N: usize, Shape: Storage<N>, Strides: Storage<N>> Drop
    for Allocation<M, N, Shape, Strides>
{
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.managed.as_ptr().cast(), self.layout) };
    }
}

/// Metadata retained by an initialized fixed-rank allocation.
pub struct Metadata<const N: usize, Shape: Storage<N> = Copied, Strides: Storage<N> = Copied> {
    shape: NonNull<Shape::Value>,
    strides: NonNull<Strides::Value>,
}

/// An initialized fixed-rank allocation.
pub type Initialized<M, const N: usize, Shape = Copied, Strides = Copied> =
    super::Initialized<M, Metadata<N, Shape, Strides>>;

impl<M, const N: usize, Shape, Strides> super::Initialized<M, Metadata<N, Shape, Strides>>
where
    M: ManagedTensorBase,
    Shape: Storage<N>,
    Strides: Storage<N>,
{
    /// Returns the selected shape storage.
    pub fn shape_storage_mut(&mut self) -> &mut Shape::Value {
        unsafe { self.storage.shape.as_mut() }
    }

    /// Returns the selected strides storage.
    pub fn strides_storage_mut(&mut self) -> &mut Strides::Value {
        unsafe { self.storage.strides.as_mut() }
    }
}

impl<M: ManagedTensorBase, const N: usize, Strides: Storage<N>> Allocation<M, N, Copied, Strides> {
    pub fn shape_mut(&mut self) -> &mut [i64; N] {
        self.shape_storage_mut()
    }
}

impl<M: ManagedTensorBase, const N: usize, Shape: Storage<N>> Allocation<M, N, Shape, Copied> {
    pub fn strides_mut(&mut self) -> &mut [i64; N] {
        self.strides_storage_mut()
    }
}

impl<M: ManagedTensorBase, const N: usize, Strides: Storage<N>>
    super::Initialized<M, Metadata<N, Copied, Strides>>
{
    pub fn shape_mut(&mut self) -> &mut [i64; N] {
        self.shape_storage_mut()
    }
}

impl<M: ManagedTensorBase, const N: usize, Shape: Storage<N>>
    super::Initialized<M, Metadata<N, Shape, Copied>>
{
    pub fn strides_mut(&mut self) -> &mut [i64; N] {
        self.strides_storage_mut()
    }
}

struct Parts {
    layout: Layout,
    shape: usize,
    strides: usize,
}

fn allocation_parts<M, const N: usize, Shape, Strides>() -> Result<Parts, Error>
where
    Shape: Storage<N>,
    Strides: Storage<N>,
{
    let (layout, shape) = Layout::new::<M>()
        .extend(Layout::new::<Shape::Value>())
        .map_err(|_| Error::LayoutOverflow)?;
    let (layout, strides) = layout
        .extend(Layout::new::<Strides::Value>())
        .map_err(|_| Error::LayoutOverflow)?;
    Ok(Parts {
        layout: layout.pad_to_align(),
        shape,
        strides,
    })
}

unsafe extern "C" fn drop_allocation<C, M, const N: usize, Shape, Strides>(managed: *mut M)
where
    C: OpaqueContext,
    M: ManagedTensorBase,
    Shape: Storage<N>,
    Strides: Storage<N>,
{
    if managed.is_null() {
        return;
    }
    unsafe {
        let parts =
            allocation_parts::<M, N, Shape, Strides>().unwrap_or_else(|_| std::process::abort());
        C::drop_raw((*managed).manager_ctx());
        std::ptr::drop_in_place(managed);
        std::alloc::dealloc(managed.cast(), parts.layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManagedTensorBase, ffi::DLManagedTensor};

    #[test]
    fn copied_shape_and_strides_use_independent_arrays() {
        let mut allocation = Allocation::<DLManagedTensor, 2>::allocate().unwrap();
        *allocation.shape_mut() = [2, 3];
        *allocation.strides_mut() = [3, 1];
        let mut initialized = allocation.initialize(Box::new(()));
        initialized.tensor_mut().shape = initialized.shape_mut().as_mut_ptr();
        initialized.tensor_mut().strides = initialized.strides_mut().as_mut_ptr();
        let tensor = unsafe { initialized.finish() };
        assert_eq!(tensor.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor.strides().unwrap().unwrap(), &[3, 1]);
    }

    #[test]
    fn borrowed_shape_allocates_only_strides() {
        let shape = [2_i64, 3];
        let mut allocation =
            Allocation::<DLManagedTensor, 2, Borrowed, Copied>::allocate().unwrap();
        *allocation.strides_mut() = [3, 1];
        let mut initialized = allocation.initialize(Box::new(()));
        initialized.tensor_mut().shape = shape.as_ptr().cast_mut();
        initialized.tensor_mut().strides = initialized.strides_mut().as_mut_ptr();
        let tensor = unsafe { initialized.finish() };
        assert_eq!(tensor.shape().unwrap(), &shape);
    }

    #[test]
    fn storage_types_determine_layout() {
        let one = allocation_parts::<DLManagedTensor, 2, Borrowed, Copied>()
            .unwrap()
            .layout;
        let two = allocation_parts::<DLManagedTensor, 2, Copied, Copied>()
            .unwrap()
            .layout;
        assert!(one.size() < two.size());
    }

    #[test]
    fn allocation_and_deleter_use_const_layout() {
        drop(Allocation::<DLManagedTensor, 2>::allocate().unwrap());
        let allocation = Allocation::<DLManagedTensor, 2>::allocate().unwrap();
        let initialized = allocation.initialize(Box::new(()));
        let tensor = unsafe { initialized.finish() };
        unsafe { (*tensor.as_ptr()).tensor_mut().ndim = i32::MAX };
        drop(tensor);
    }
}
